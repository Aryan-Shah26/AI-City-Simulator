"""World orchestrator: tick loop, locations, and the shared Community Board."""
import asyncio
import logging

from models import ActionType, AgentAction, BoardPost, MemoryEvent
from economy import Economy
from memory import MemoryManager
from agent import Agent

log = logging.getLogger("world")


class CommunityBoard:
    """Async-safe public bulletin board agents can post to and read from."""

    def __init__(self):
        self._posts: list[BoardPost] = []
        self._lock = asyncio.Lock()

    async def post(self, post: BoardPost) -> None:
        async with self._lock:
            self._posts.append(post)

    async def read_recent(self, n: int = 10) -> list[BoardPost]:
        async with self._lock:
            return self._posts[-n:]


class World:
    """Owns all agents, the board, and drives the tick loop."""

    def __init__(self, agents: list[Agent], memory_mgr: MemoryManager, plan_interval: int = 5):
        self.agents = {a.config.id: a for a in agents}
        self.board = CommunityBoard()
        self.memory_mgr = memory_mgr
        self.plan_interval = plan_interval
        self.tick_num = 0

    async def run(self, num_ticks: int) -> None:
        for _ in range(num_ticks):
            self.tick_num += 1
            await self._run_tick()

    async def _run_tick(self) -> None:
        """One tick: periodic planning, concurrent action decisions, then resolution + memory consolidation."""
        log.info("── Tick %d ──", self.tick_num)

        for agent in self.agents.values():
            if self.tick_num - agent.state.last_plan_tick >= self.plan_interval:
                try:
                    await agent.plan(self.tick_num)
                    log.info("[%s] plan (t%d): %s", agent.config.name, self.tick_num, agent.state.current_plan)
                except Exception:
                    log.warning("[%s] planning failed, keeping previous plan", agent.config.name, exc_info=True)

        agents = list(self.agents.values())
        results = await asyncio.gather(
            *(a.decide_action(self.tick_num, self.board, agents) for a in agents),
            return_exceptions=True,
        )
        actions: list[AgentAction] = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                log.warning("[%s] decide_action failed (%s), defaulting to idle", agent.config.name, result)
                actions.append(AgentAction(action_type=ActionType.IDLE, reasoning="fallback: LLM call failed"))
            else:
                actions.append(result)

        for agent, action in zip(agents, actions):
            target_str = f" -> {action.target_agent_id}" if action.target_agent_id else ""
            deceit_str = " [DECEPTIVE]" if action.is_deceptive else ""
            log.info("[%s] action: %s%s%s", agent.config.name, action.action_type.value, target_str, deceit_str)
            log.info("[%s] thought: %s", agent.config.name, action.reasoning or "(none given)")
            await self._resolve_action(agent, action)

        for agent in agents:
            if self.memory_mgr.is_full(agent.state):
                log.info("[%s] consolidating memory", agent.config.name)
                await self.memory_mgr.consolidate(agent.state, agent.config.llm)

    async def _resolve_action(self, agent: Agent, action) -> None:
        target = self.agents.get(action.target_agent_id) if action.target_agent_id else None

        if action.action_type == ActionType.POST_BOARD and action.content:
            await self.board.post(BoardPost(tick=self.tick_num, author_id=agent.config.id, content=action.content))
            desc = f"{agent.config.name} posted to the board: {action.content}"
            log.info("  [board] %s", desc)
            self._log(agent.config.id, desc)

        elif action.action_type == ActionType.TRADE and target and action.offer:
            ok = Economy.apply_trade(agent.config, target.config, action.offer)
            desc = f"{agent.config.name} {'traded with' if ok else 'failed to trade with'} {target.config.name}."
            log.info("  [trade] %s offer=%s", desc, action.offer)
            self._log(agent.config.id, desc, witnesses=[target.config.id])
            self._log(target.config.id, desc, witnesses=[agent.config.id])

        elif action.action_type == ActionType.STEAL and target:
            _, caught, desc = Economy.attempt_steal(agent.config, agent.state, target.config, target.state, self.tick_num)
            log.info("  [steal] %s", desc)
            self._log(agent.config.id, desc)
            if caught:
                self._log(target.config.id, desc, witnesses=[agent.config.id])

        elif action.action_type == ActionType.CONVERSE and target:
            log.info("  [converse] %s <-> %s", agent.config.name, target.config.name)
            transcript = await agent.converse(target, self.tick_num)
            for line in transcript:
                log.info("    %s", line)

        elif action.action_type == ActionType.WORK:
            agent.config.financials.cash += 10
            desc = f"{agent.config.name} worked and earned income."
            log.info("  [work] %s (cash=%.1f)", desc, agent.config.financials.cash)
            self._log(agent.config.id, desc)

    def _log(self, agent_id: str, description: str, witnesses: list[str] | None = None) -> None:
        agent = self.agents.get(agent_id)
        if agent:
            self.memory_mgr.add_event(
                agent.state, MemoryEvent(tick=self.tick_num, description=description, witnessed_by=witnesses or [])
            )