"""Multi-LLM Router. Dynamically creates the right client per agent using aiohttp."""
import json
import os
import time
import logging
import asyncio
import collections
import aiohttp
import config
import random
from aiolimiter import AsyncLimiter

log = logging.getLogger("llm_client")

_session: aiohttp.ClientSession = None
_semaphore: asyncio.Semaphore = None
_client_cache: dict = {}                 # llm_string -> LLMClient instance (reused across ticks/agents)
_embedding_cache = collections.OrderedDict()  # text -> vector, capped LRU
_stats = {
    "llm_calls": 0, "llm_time": 0.0, "embed_calls": 0, "embed_time": 0.0, "embed_cache_hits": 0,
    "json_parse_failures": 0, "rate_limit_hits": 0,
}
# Ollama is a local process with no real rate limit - only Groq's actual API quota needs throttling.
# Sharing one limiter across both providers meant local calls were stealing budget meant for Groq.
_groq_limiter = AsyncLimiter(config.GROQ_REQUESTS_PER_MINUTE, 60)

async def get_session() -> aiohttp.ClientSession:
    """Shared, connection-pooled session. Avoids a new TCP/TLS handshake per LLM/embedding call."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


def get_semaphore() -> asyncio.Semaphore:
    """Caps concurrent in-flight LLM requests so we don't flood local Ollama or hit rate limits."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_REQUESTS)
    return _semaphore


async def close_session():
    global _session
    if _session and not _session.closed:
        await _session.close()


def get_stats() -> dict:
    return dict(_stats)


class LLMClient:
    rate_limiter = None  # set per-provider subclass; None means "don't throttle" (local Ollama)

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict: raise NotImplementedError
    async def generate_text(self, system_prompt: str, user_prompt: str) -> str: raise NotImplementedError

    async def _timed_post(self, coro_fn, *args):
        """Wraps a provider's raw POST with the concurrency semaphore, timing stats, and
        retry/backoff. 429s and plain connection errors get separate retry budgets."""
        sem = get_semaphore()
        last_err = None
        transport_attempt = 0
        rate_limit_attempt = 0

        while True:
            start = time.perf_counter()
            try:
                if self.rate_limiter is not None:
                    async with self.rate_limiter:
                        async with sem:
                            result = await coro_fn(*args)
                else:
                    async with sem:
                        result = await coro_fn(*args)
                _stats["llm_calls"] += 1
                _stats["llm_time"] += time.perf_counter() - start
                return result

            except aiohttp.ClientResponseError as e:
                last_err = e
                if e.status == 429:
                    _stats["rate_limit_hits"] += 1
                    if rate_limit_attempt >= config.MAX_429_RETRIES:
                        log.error("[429] exhausted %d retries, giving up", config.MAX_429_RETRIES)
                        raise
                    rate_limit_attempt += 1
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    delay = float(retry_after) if retry_after else min((2 ** rate_limit_attempt) + random.uniform(0, 1), 60)
                    # In-progress retries are expected/routine - DEBUG only, not console noise.
                    log.debug("[429] %d/%d - waiting %.2fs before retry...", rate_limit_attempt, config.MAX_429_RETRIES, delay)
                    await asyncio.sleep(delay)
                    continue

                # Include 400 as a potential context limit error
                if e.status in [400, 413, 419]:
                    log.error("[%d] Fatal API Error. Message: %s", e.status, e.message)
                    return "{}"

                raise

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
                if transport_attempt >= config.LLM_TRANSPORT_RETRIES:
                    log.error("[transport] exhausted %d retries: %s", config.LLM_TRANSPORT_RETRIES, e)
                    raise last_err
                transport_attempt += 1
                log.debug("[transport] retry %d/%d after: %s", transport_attempt, config.LLM_TRANSPORT_RETRIES, e)
                await asyncio.sleep((2 ** transport_attempt) + random.uniform(0, 1))


class OllamaClient(LLMClient):
    def __init__(self, model):
        self.model = model
        self.url = f"{config.OLLAMA_HOST}/api/chat"

    async def _raw_post(self, payload) -> str:
        session = await get_session()
        async with session.post(self.url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["message"]["content"]

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "format": "json", "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE, "num_predict": config.LLM_MAX_TOKENS},
        }
        content = await self._timed_post(self._raw_post, payload)
        return _safe_parse(content)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "stream": False
        }
        return await self._timed_post(self._raw_post, payload)


class GroqClient(LLMClient):
    rate_limiter = _groq_limiter

    def __init__(self, model):
        self.model = model
        self.api_key = os.environ.get(config.GROQ_API_KEY_ENV)
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    async def _raw_post(self, payload, headers) -> str:
        session = await get_session()
        async with session.post(self.url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                # Capture the actual Groq error body
                error_body = await resp.text()
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=f"{resp.reason} - Details: {error_body}",
                    headers=resp.headers
                )
            
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
        
    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"}, "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        content = await self._timed_post(self._raw_post, payload, headers)
        return _safe_parse(content)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": config.LLM_TEMPERATURE
        }
        return await self._timed_post(self._raw_post, payload, headers)


def _extract_json_block(text: str) -> str | None:
    """Best-effort fallback: scan for the first balanced {...} block. Catches cases where
    a model prefixes/suffixes valid JSON with prose or a markdown fence despite
    response_format=json_object (common with smaller/faster instant models)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _safe_parse(content: str) -> dict:
    stripped = content.strip().strip("`")
    if stripped[:4].lower() == "json":
        stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    block = _extract_json_block(content)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    # Both attempts failed. This used to return {} with zero trace of what the model
    # actually said - log a preview so failures are diagnosable, and count it so the
    # end-of-run stats show the real parse-failure rate instead of it being invisible.
    _stats["json_parse_failures"] += 1
    preview = content.strip()[:400].replace("\n", " ")
    log.warning("JSON parse failed. Raw response preview: %r", preview)
    return {}


def get_client_for_agent(llm_string: str) -> LLMClient:
    """Returns a cached client per model string - avoids re-instantiating (and re-parsing
    env vars / building URLs) on every single decide() call."""
    if llm_string in _client_cache:
        return _client_cache[llm_string]
    if ":" not in llm_string:
        # A bare model name (e.g. "openai/gpt-oss-120b") with no "provider:" prefix used
        # to blow up here with "not enough values to unpack (expected 2, got 1)" from the
        # split below - a confusing error far from its real cause. Fail loudly and clearly
        # at the actual mistake instead.
        raise ValueError(
            f"Invalid llm string {llm_string!r}: expected 'provider:model' "
            f"(e.g. 'ollama:llama3.2:1b' or 'groq:openai/gpt-oss-120b')."
        )
    provider, model = llm_string.split(":", 1)
    client = GroqClient(model) if provider == "groq" else OllamaClient(model)
    _client_cache[llm_string] = client
    return client


async def get_embedding(text: str) -> list:
    """Generates a mathematical vector for a piece of text using local Ollama (non-blocking),
    with an in-memory cache so identical text never triggers a duplicate network round-trip."""
    if text in _embedding_cache:
        _stats["embed_cache_hits"] += 1
        _embedding_cache.move_to_end(text)
        return _embedding_cache[text]

    start = time.perf_counter()
    try:
        session = await get_session()
        payload = {"model": "nomic-embed-text", "prompt": text}
        sem = get_semaphore()
        async with sem:
            async with session.post(f"{config.OLLAMA_HOST}/api/embeddings", json=payload,
                                     timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                vector = data.get("embedding", [])
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        vector = []

    _stats["embed_calls"] += 1
    _stats["embed_time"] += time.perf_counter() - start

    if vector:
        _embedding_cache[text] = vector
        if len(_embedding_cache) > config.EMBEDDING_CACHE_SIZE:
            _embedding_cache.popitem(last=False)
    return vector