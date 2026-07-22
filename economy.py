"""Resource transfer, trade resolution, and robbery mechanics."""
import random

from models import AgentConfig, AgentRuntimeState

STEAL_SUCCESS_CHANCE = 0.5
STEAL_CATCH_CHANCE = 0.6
STEAL_TAKE_FRACTION = 0.2
REPUTATION_PENALTY_CAUGHT_STEAL = 25
REPUTATION_PENALTY_CAUGHT_LIE = 15


class Economy:
    """Stateless resolver for trade and theft actions between two agents' financials."""

    @staticmethod
    def apply_trade(actor_cfg: AgentConfig, target_cfg: AgentConfig, offer: dict[str, float]) -> bool:
        """Apply actor's proposed deltas (target receives the inverse) if both sides can afford it."""
        actor_fin, target_fin = actor_cfg.financials, target_cfg.financials
        for key, delta in offer.items():
            if not hasattr(actor_fin, key):
                return False
            if getattr(actor_fin, key) + delta < 0 or getattr(target_fin, key) - delta < 0:
                return False
        for key, delta in offer.items():
            setattr(actor_fin, key, getattr(actor_fin, key) + delta)
            setattr(target_fin, key, getattr(target_fin, key) - delta)
        return True

    @staticmethod
    def attempt_steal(
        actor_cfg: AgentConfig,
        actor_state: AgentRuntimeState,
        target_cfg: AgentConfig,
        target_state: AgentRuntimeState,
        tick: int,
    ) -> tuple[bool, bool, str]:
        """Resolve a robbery attempt. Returns (success, caught, description)."""
        success = random.random() < STEAL_SUCCESS_CHANCE
        caught = random.random() < STEAL_CATCH_CHANCE

        if success:
            amount = target_cfg.financials.cash * STEAL_TAKE_FRACTION
            target_cfg.financials.cash -= amount
            actor_cfg.financials.cash += amount

        if caught:
            actor_state.reputation = max(0.0, actor_state.reputation - REPUTATION_PENALTY_CAUGHT_STEAL)
            target_state.opinions[actor_cfg.id] = max(-100.0, target_state.opinions.get(actor_cfg.id, 0.0) - 40)
            desc = (
                f"{actor_cfg.name} was caught {'successfully' if success else 'unsuccessfully'} "
                f"attempting to steal from {target_cfg.name}."
            )
        else:
            desc = (
                f"{actor_cfg.name} {'stole cash' if success else 'failed to steal'} "
                f"from {target_cfg.name} unnoticed."
            )
        return success, caught, desc

    @staticmethod
    def register_deceit_penalty(liar_state: AgentRuntimeState, discoverer_state: AgentRuntimeState, liar_id: str) -> None:
        """Apply reputation/opinion fallout once a lie is discovered by another agent."""
        liar_state.reputation = max(0.0, liar_state.reputation - REPUTATION_PENALTY_CAUGHT_LIE)
        discoverer_state.opinions[liar_id] = max(-100.0, discoverer_state.opinions.get(liar_id, 0.0) - 25)