"""Rolling short-term memory buffer with periodic LLM-driven summarization."""
from models import AgentRuntimeState, MemoryEvent
from llm_client import LLMClient

SUMMARIZE_SYSTEM_PROMPT = (
    "You compress an agent's recent memory log into a short third-person narrative "
    "(3-5 sentences) capturing key relationships, betrayals, debts, and unresolved goals. "
    "Be factual and concise."
)


class MemoryManager:
    """Owns short-term buffer eviction and long-term summarization for all agents."""

    def __init__(self, llm_client: LLMClient, buffer_size: int = 20):
        self.llm_client = llm_client
        self.buffer_size = buffer_size

    def add_event(self, state: AgentRuntimeState, event: MemoryEvent) -> None:
        state.short_term_memory.append(event)

    def is_full(self, state: AgentRuntimeState) -> bool:
        return len(state.short_term_memory) >= self.buffer_size

    async def consolidate(self, state: AgentRuntimeState, llm_id: str) -> None:
        """Summarize the buffer into long_term_memory via LLM call, then clear it."""
        if not state.short_term_memory:
            return
        log = "\n".join(f"[t{e.tick}] {e.description}" for e in state.short_term_memory)
        prompt = (
            f"Existing long-term memory: {state.long_term_memory or 'None'}\n\n"
            f"New events to fold in:\n{log}\n\n"
            "Produce the updated long-term memory narrative."
        )
        summary = await self.llm_client.complete(
            llm_id,
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        state.long_term_memory = summary.strip()
        state.short_term_memory.clear()