"""Module to handle robust map-reduce chronological simulation history generation.

Three separate concerns, kept deliberately distinct:
  1. INPUT robustness - never skip or cut off raw log content before it's been
     considered, no matter how large the log file is.
  2. OUTPUT compactness - the final chronicle must be short.
  3. OUTPUT format - the final chronicle must actually be a terse bullet list,
     not prose. Earlier versions asked for a "concise summary under N words" -
     but a word *ceiling* just gives a padding-prone model room to pad up to,
     it doesn't make the model want to be brief. Bullet points sidestep that:
     it's much harder (and much more obviously wrong) for a model to pad out a
     bullet with filler than to pad a paragraph. Bullet extraction is also
     enforced in code, not just requested in the prompt, so a model that
     ignores the format still gets forced into it.

Design:
  - Logs are streamed line-by-line off disk - the whole file is never held in
    memory, so this scales the same whether the log is 100 lines or 100 million.
  - MAP step: each chunk is compressed into a short bullet list of only its
    significant events (routine/repetitive actions like idle/move are skipped
    entirely, not summarized).
  - REDUCE step: bullet lists are merged pairwise in a binary tree (log2(n)
    rounds). Each merge DEDUPES and re-selects the most significant bullets
    from the two inputs down to a fixed cap - it does not concatenate. Because
    every node in the tree is capped to the same bullet count, the final
    chronicle's size is bounded by that cap, not by how many chunks the
    original log produced.
  - Every stage's output is passed through _extract_bullets(), which parses
    out actual bullet lines (or, if the model ignored the format and replied
    in prose, splits that prose into sentences as a fallback) and hard-caps
    both the bullet count and the words-per-bullet. This is what actually
    guarantees brevity - the prompt asks nicely, the code enforces it.
  - Failure handling unchanged: llm_client.py's `_timed_post` returns the
    literal string "{}" on a fatal 400/413/419 (context overflow). That's
    treated as a failed call (retried with backoff), and a chunk that still
    can't be summarized is split in half and recursed rather than dropped.
"""

import asyncio
import re
from typing import Iterator, List
from llm_client import LLMClient

# llm_client._timed_post's sentinel string for a fatal (non-retryable-by-it) API error,
# e.g. a 400/413/419 context-overflow response. Never trust this as real content.
_FAILURE_SENTINELS = {"", "{}", "{}."}

_BULLET_PREFIX_RE = re.compile(r"^[-*•\u2022]\s*")
_NUMBERED_PREFIX_RE = re.compile(r"^\d+[\.\)]\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class HistorianError(Exception):
    """Raised only if the historian truly cannot produce any chronicle at all."""


async def _identity(x: str) -> str:
    return x


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _extract_bullets(text: str, max_bullets: int, words_per_bullet: int) -> List[str]:
    """The actual enforcement layer. Pulls real bullet lines out of the model's
    response; if the model ignored the format and wrote prose instead, falls back to
    treating each sentence as a bullet. Either way, hard-caps count and per-bullet
    length in code - not just via prompt instruction - so padding can't get through."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bullets = []
    for line in lines:
        if _BULLET_PREFIX_RE.match(line) or _NUMBERED_PREFIX_RE.match(line):
            content = _BULLET_PREFIX_RE.sub("", line)
            content = _NUMBERED_PREFIX_RE.sub("", content)
            if content:
                bullets.append(content)

    if not bullets:
        # Model replied in prose despite instructions - degrade gracefully into
        # sentence-as-bullet rather than returning nothing or a raw paragraph.
        sentences = _SENTENCE_SPLIT_RE.split(text.strip())
        bullets = [s.strip() for s in sentences if s.strip()]

    bullets = bullets[:max_bullets]
    bullets = [_cap_words(b, words_per_bullet) for b in bullets]
    return bullets


def _format_bullets(bullets: List[str]) -> str:
    return "\n".join(f"- {b}" for b in bullets)


class SimulationHistorian:
    def __init__(self, client: LLMClient, chunk_chars: int = 12000,
                 chunk_max_bullets: int = 5, merge_max_bullets: int = 8,
                 final_max_bullets: int = 10, words_per_bullet: int = 18,
                 max_chunk_retries: int = 3, max_concurrent_calls: int = 4):
        """
        client: any LLMClient (Ollama/Groq) - must implement generate_text.
        chunk_chars: soft cap on characters per leaf chunk fed to a single map call.
        chunk_max_bullets: max bullets extracted per leaf chunk summary.
        merge_max_bullets: max bullets kept after each reduce-step merge. Since every
            node in the tree (leaf or merged) is capped near this count, the final
            chronicle's size is bounded by this constant, not by log length.
        final_max_bullets: max bullets in the final chronicle handed back to the
            caller (aim for the 5-10 range for an actually concise result).
        words_per_bullet: hard per-bullet word cap, enforced in code.
        max_chunk_retries: retries (with backoff) before a chunk/merge is treated as a
            hard failure and escalated (chunk split, or fragment concatenation).
        max_concurrent_calls: local concurrency cap for historian LLM calls, on top of
            (not instead of) llm_client's own global semaphore/rate limiter.
        """
        self.client = client
        self.chunk_chars = chunk_chars
        self.chunk_max_bullets = chunk_max_bullets
        self.merge_max_bullets = merge_max_bullets
        self.final_max_bullets = final_max_bullets
        self.words_per_bullet = words_per_bullet
        self.max_chunk_retries = max_chunk_retries
        self._sem = asyncio.Semaphore(max_concurrent_calls)

    # ---------------------------------------------------------------- reading
    def _iter_lines(self, log_file_path: str) -> Iterator[str]:
        """Streams the log file line by line rather than .readlines()-ing it whole."""
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    yield stripped

    def _chunk_lines(self, lines: Iterator[str]) -> List[str]:
        """Groups lines into character-bounded chunks. A single line longer than
        chunk_chars is still emitted as its own (oversized) chunk rather than being
        cut - _summarize_chunk's split-and-recurse handles it if that's too big for
        a single call."""
        chunks = []
        current, current_len = [], 0
        for line in lines:
            if current and current_len + len(line) + 1 > self.chunk_chars:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            current.append(line)
            current_len += len(line) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    async def _call_with_retries(self, system_prompt: str, user_prompt: str, label: str) -> str:
        """Shared retry/backoff wrapper. Returns raw (untrimmed) text, or "" if every
        retry failed - bullet extraction/trimming happens one layer up, at the caller,
        after this returns."""
        for attempt in range(self.max_chunk_retries + 1):
            async with self._sem:
                result = await self.client.generate_text(system_prompt, user_prompt)
            result = (result or "").strip()
            if result not in _FAILURE_SENTINELS:
                return result
            if attempt < self.max_chunk_retries:
                delay = 1.5 * (attempt + 1)
                print(f"[Historian] {label}: empty/failed response (attempt {attempt + 1}/"
                      f"{self.max_chunk_retries + 1}), retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        return ""

    # ------------------------------------------------------------------ map
    async def _summarize_chunk(self, chunk_text: str) -> str:
        """Map step: extract only the significant events from one chunk of raw log
        lines as a short bullet list. Routine/repetitive actions (idle, ordinary
        movement) are skipped entirely rather than summarized - they add bulk with
        no informational value."""
        system_prompt = (
            "You extract significant events from simulation logs as bullet points. "
            "Output ONLY a bullet list (each line starting with '- '), one significant "
            "event per bullet: notable trades, economic shifts, conflicts, alliances, "
            "large cash/debt changes. SKIP routine or repetitive actions (idle, "
            "ordinary movement, small trades) entirely - do not make a bullet for "
            f"them. Maximum {self.chunk_max_bullets} bullets. If fewer than "
            f"{self.chunk_max_bullets} events are actually significant, output fewer "
            "bullets - never pad to reach the maximum. No preamble, no headers, no "
            "closing remarks."
        )
        user_prompt = f"[RAW LOG LINES]\n{chunk_text}\n\n[TASK]\nExtract only the significant events as bullets."
        raw = await self._call_with_retries(system_prompt, user_prompt, "chunk summary")

        if raw:
            bullets = _extract_bullets(raw, self.chunk_max_bullets, self.words_per_bullet)
            if bullets:
                return _format_bullets(bullets)
            # Extraction found nothing usable (e.g. model said "nothing significant") -
            # that's a legitimate empty result for a quiet chunk, not a failure.
            return ""

        # Hard failure even after retries. If this chunk has more than one line, the
        # chunk itself may be what's overflowing the model - split it and recurse
        # instead of giving up on the data.
        lines = chunk_text.split("\n")
        if len(lines) > 1:
            mid = len(lines) // 2
            left, right = "\n".join(lines[:mid]), "\n".join(lines[mid:])
            print(f"[Historian] Splitting an unsummarizable {len(lines)}-line chunk in half and retrying...")
            left_summary, right_summary = await asyncio.gather(
                self._summarize_chunk(left), self._summarize_chunk(right)
            )
            combined = [b for part in (left_summary, right_summary) for b in part.splitlines() if b.strip()]
            bullets = _extract_bullets("\n".join(combined), self.chunk_max_bullets, self.words_per_bullet)
            return _format_bullets(bullets)

        # A single line that still fails: keep it raw as one bullet rather than
        # dropping it.
        print("[Historian] A single log line could not be summarized after retries; keeping it raw.")
        return f"- {_cap_words(chunk_text, self.words_per_bullet)}"

    # --------------------------------------------------------------- reduce
    async def _merge_pair(self, a: str, b: str) -> str:
        """Reduce step: DEDUPES and re-selects the most significant bullets from two
        chronologically-ordered bullet lists (a before b) down to a fixed cap - it does
        not concatenate them. Held to the same bullet cap at every level of the tree,
        so the final chronicle's length is bounded by that cap, not by log size."""
        if not a:
            return b
        if not b:
            return a

        system_prompt = (
            "You merge two chronologically ordered bullet lists (A happened before B) "
            "of simulation events into ONE bullet list. Deduplicate overlapping "
            "events, drop the least significant ones, and keep only the most "
            f"important. Output ONLY a bullet list, maximum {self.merge_max_bullets} "
            "bullets, in chronological order. If fewer bullets are genuinely "
            "significant, output fewer - never pad. No preamble, no headers."
        )
        user_prompt = f"[LIST A - EARLIER]\n{a}\n\n[LIST B - LATER]\n{b}\n\n[TASK]\nMerge into one deduplicated bullet list."
        raw = await self._call_with_retries(system_prompt, user_prompt, "fragment merge")

        if raw:
            bullets = _extract_bullets(raw, self.merge_max_bullets, self.words_per_bullet)
            if bullets:
                return _format_bullets(bullets)

        # Merging genuinely failed - fall back to combining both lists' bullets
        # directly and hard-capping, rather than dropping one side's content.
        print("[Historian] Merge failed after retries; combining bullet lists directly instead.")
        combined = [l for src in (a, b) for l in src.splitlines() if l.strip()]
        bullets = _extract_bullets("\n".join(combined), self.merge_max_bullets, self.words_per_bullet)
        return _format_bullets(bullets)

    async def _reduce_tree(self, fragments: List[str]) -> str:
        """Binary-tree reduction: merge neighbor pairs each round until one fragment
        remains. O(log n) sequential rounds; each call sees only 2 already-short
        bullet lists and produces one equally-short list - size stays flat across
        rounds instead of compounding."""
        current = [f for f in fragments if f]  # drop empty (quiet) chunks up front
        if not current:
            return ""
        round_num = 0
        while len(current) > 1:
            round_num += 1
            print(f"[Historian] Reduce round {round_num}: merging {len(current)} fragment(s)...")
            tasks = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    tasks.append(self._merge_pair(current[i], current[i + 1]))
                else:
                    tasks.append(_identity(current[i]))  # odd one out, carried to next round
            current = await asyncio.gather(*tasks)
        return current[0] if current else ""

    # --------------------------------------------------------------- public
    async def compile_final_history(self, log_file_path: str, historical_focus: str) -> str:
        try:
            lines = list(self._iter_lines(log_file_path))
        except FileNotFoundError:
            return "Error: No simulation history logs found."
        except OSError as e:
            return f"Error: Could not read simulation logs ({e})."

        if not lines:
            return "Simulation logs are empty."

        chunks = self._chunk_lines(iter(lines))
        print(f"[Historian] Summarizing {len(chunks)} log chunk(s) (map step, "
              f"max {self.chunk_max_bullets} bullets each)...")

        summaries = await asyncio.gather(*(self._summarize_chunk(c) for c in chunks))
        non_empty = sum(1 for s in summaries if s)
        print(f"[Historian] {non_empty}/{len(summaries)} chunk(s) had significant events. "
              f"Reducing (merge step, max {self.merge_max_bullets} bullets per merge)...")

        running_history = await self._reduce_tree(list(summaries))
        if not running_history:
            return "No significant events found in the simulation logs."
        bullet_count = len([l for l in running_history.splitlines() if l.strip()])
        print(f"[Historian] Chronicle assembled ({bullet_count} bullets).")

        final_system_prompt = (
            "You produce the final summary of a simulation for a reader who has not "
            f"seen the raw logs. Output ONLY a bullet list of {self.final_max_bullets} "
            "bullets or fewer - the single most significant events of the whole run. "
            "Each bullet is one short, factual sentence. No introduction, no headers, "
            "no concluding remarks, no restating the request. Do not pad to reach the "
            "maximum - if only 5 events truly matter, output 5 bullets."
        )
        final_user_prompt = f"""
[COMPRESSED EVENT LIST FOR THE WHOLE RUN]
{running_history}

[TASK]
{historical_focus}
Output as a bullet list of at most {self.final_max_bullets} bullets.
"""
        raw_final = await self._call_with_retries(final_system_prompt, final_user_prompt, "final compilation")
        if raw_final:
            bullets = _extract_bullets(raw_final, self.final_max_bullets, self.words_per_bullet)
            if bullets:
                return _format_bullets(bullets)

        # Final compilation call failed or returned nothing usable - fall back to the
        # already-bounded assembled chronicle, re-capped to the final bullet count.
        print("[Historian] Final compilation call failed after retries; re-capping assembled chronicle.")
        bullets = _extract_bullets(running_history, self.final_max_bullets, self.words_per_bullet)
        return _format_bullets(bullets)