"""Wraps one agent's turn with support for deterministic queues, interruptible plans, and LLM inference."""

import logging
import config
from state import WorldState, AgentState
from llm_client import LLMClient
import prompts

log = logging.getLogger("agent")

def try_queued_action(agent: AgentState, world: WorldState) -> dict:
    """Returns the next valid queued action, checking for expiry and event interrupts."""
    # 1. Event-Driven Replanning: Check for emergencies
    if agent.energy <= 20:
        agent.priority_level = "emergency"
        agent.action_queue.clear()
        agent.plan_status = "starving"
        return None

    # 2. Plan Expiry
    if world.tick > agent.plan_expiry_tick:
        agent.action_queue.clear()
        agent.plan_status = "expired"
        return None

    if agent.action_queue:
        next_action = agent.action_queue[0]
        
        # 3. Dynamic Validation (Contingency fallback)
        if _is_action_valid(next_action, agent):
            agent.action_queue.pop(0)
            next_action["thought"] = f"(Executing planned step: {next_action.get('action')})"
            return next_action
        elif "contingency_action" in next_action and next_action["contingency_action"] != "none":
            # Swap to contingency if primary fails syntactically 
            fallback = {"action": next_action["contingency_action"]}
            if _is_action_valid(fallback, agent):
                agent.action_queue.pop(0)
                fallback["thought"] = f"(Fallback triggered: {fallback['action']})"
                return fallback
                
        # Hard failure
        agent.action_queue.clear()
        agent.plan_status = "interrupted"
            
    return None

def _simulate_plan_feasibility(plan: list, agent: AgentState) -> list:
    """Deterministically dry-runs the plan to reserve resources and prune impossible futures."""
    simulated_cash = agent.cash
    simulated_wheat = agent.inventory.get("wheat", 0)
    valid_plan = []
    
    for step in plan:
        action = step.get("action")
        # Simulate cost reservations
        if action in ["buy_wheat", "buy_bread"] and simulated_cash < 10: # Assuming min price
            break # Drop the rest of the plan, financial failure imminent
        if action == "buy_land" and simulated_cash < config.LAND_PRICE:
            break
        if action in ["sell_wheat", "bake_bread"]:
            if simulated_wheat <= 0:
                break
            simulated_wheat -= 1
            
        valid_plan.append(step)
        
    return valid_plan

async def plan_new_action(agent: AgentState, world: WorldState, client: LLMClient, force_reply_to: list = None) -> dict:
    """Calls the LLM to generate a new hierarchical plan."""
    # Emergency Overrides
    if force_reply_to or world.global_event_msg:
        agent.action_queue.clear()
        agent.plan_status = "interrupted"

    people_here = ", ".join([a.name for a in world.agents_at(agent.current_location) if a.id != agent.id])
    recent_chat = " ".join(world.recent_dialogue(agent.current_location)[-2:])
    
    search_query = f"I am at {agent.current_location}. The people here are {people_here}. They are talking about: {recent_chat}"
    retrieved_memories = await agent.retrieve_relevant_memories(search_query, k=3)
    
    last_result = agent.short_term_memory[-1] if agent.short_term_memory else "None"
    
    user_prompt = prompts.build_user_prompt(agent, world, force_reply_to, retrieved_memories)
    
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            raw = await client.generate_json(prompts.SYSTEM_PROMPT, user_prompt)
            # ---> FIX: Pass 'world.tick' as the fourth argument <---
            return _validate_and_queue(raw, agent, force_reply_to is not None, world.tick)
        except Exception as e:
            if attempt == config.LLM_MAX_RETRIES:
                log.warning("%s: all %d decision attempts failed (%s: %s), defaulting to idle",
                            agent.name, config.LLM_MAX_RETRIES + 1, type(e).__name__, e)
                return {"thought": "(fallback)", "action": "idle", "target_name": None, "target": None, "dialogue": "..."}
            log.debug("%s decision attempt %d/%d failed: %s: %s",
                      agent.name, attempt + 1, config.LLM_MAX_RETRIES + 1, type(e).__name__, e)

def _is_action_valid(action_dict: dict, agent: AgentState) -> bool:
    """Sanity checks to ensure a queued action is still legally and geographically possible."""
    action = action_dict.get("action")
    
    # Financial checks
    if action in ["buy_wheat", "buy_bread", "buy_land"] and agent.cash <= 0:
        return False
    if action in ["sell_wheat", "bake_bread"] and agent.inventory.get("wheat", 0) <= 0:
        return False
        
    # Geographical checks
    if action in ["buy_wheat", "sell_wheat", "buy_bread", "sell_bread", "take_loan", "repay_loan", "buy_land", "post_notice"]:
        if agent.current_location != "Marketplace": 
            return False
        
    if action in ["bake_bread", "eat_bread"]:
        if agent.current_location != "Residential_Quarter": 
            return False
        
    if action == "gather_wheat":
        if agent.current_location != "Wilderness_Commons": 
            return False
        
    return True

def _validate_and_queue(raw: dict, agent: AgentState, is_reply: bool, current_tick: int) -> dict:
    # ... [keep existing is_reply logic] ...

    agent.long_term_goal = raw.get("long_term_goal", agent.long_term_goal)
    agent.short_term_goal = raw.get("short_term_goal", agent.short_term_goal)
    
    # Set explicit plan expiry to prevent infinite loops
    horizon = len(raw.get("plan", []))
    agent.plan_expiry_tick = current_tick + horizon + 1 
    
    raw_plan = raw.get("plan", [])
    if isinstance(raw_plan, dict):
        raw_plan = [raw_plan]
    if not isinstance(raw_plan, list) or len(raw_plan) == 0:
        log.warning("%s: malformed/empty plan array.", agent.name)
        raise ValueError("LLM returned an empty/missing plan array")

    cleaned_plan = [_clean_action(step) for step in raw_plan]
    
    # Prune plan based on deterministic resource constraints
    feasible_plan = _simulate_plan_feasibility(cleaned_plan, agent)
    if not feasible_plan:
        raise ValueError("Plan failed deterministic resource dry-run.")

    immediate_action = feasible_plan.pop(0)
    agent.action_queue = feasible_plan
    agent.plan_status = "active"
    agent.priority_level = "normal"

    immediate_action["thought"] = raw.get("reflection", "")[:200]
    return immediate_action

def _clean_action(raw_step) -> dict:
    """Standardizes a single action dictionary. Tolerates malformed steps from the LLM
    (e.g. the model returning "plan": ["do something", ...] - a bare string instead of
    an object) by falling back to a safe idle action instead of crashing the whole
    plan with AttributeError: 'str' object has no attribute 'get'."""
    if not isinstance(raw_step, dict):
        return {
            "action": "idle",
            "amount": None,
            "target_name": None,
            "target": None,
            "dialogue": "...",
            "notice_text": None,
        }

    action = raw_step.get("action")
    if action not in config.VALID_ACTIONS:
        action = "idle"

    raw_amount = raw_step.get("amount")
    try:
        amount = int(raw_amount) if raw_amount is not None else None
    except (TypeError, ValueError):
        amount = None

    target_name = raw_step.get("target_name")
    if isinstance(target_name, str):
        target_name = target_name.strip() or None

    return {
        "action": action,
        "amount": amount,
        "target_name": target_name,
        "target": raw_step.get("target"),
        "dialogue": raw_step.get("dialogue") or "...",
        "notice_text": raw_step.get("notice_text")
    }


async def generate_conversation(agent_a: AgentState, agent_b: AgentState, world: WorldState,
                                 opening_line: str, client: LLMClient) -> list:
    """One LLM call writes the rest of the exchange after agent_a's opening line.
    Returns a flat list of 'Speaker: text' strings (opening line included) - this is what
    simulation.py was already calling and expecting, it just didn't exist yet.
    Falls back to a single silent continuation line if the model call/schema fails, so a
    bad LLM response degrades the conversation instead of crashing the tick."""
    convo_history = [f"{agent_a.name}: {opening_line}"]
    prompt = prompts.build_conversation_prompt(agent_a, agent_b, world, opening_line, config.CONVERSATION_LINES)
    try:
        raw = await client.generate_json(prompts.CONVO_SYSTEM_PROMPT, prompt)
        lines = raw.get("dialogue", [])
        if not isinstance(lines, list) or not lines:
            raise ValueError("empty or malformed dialogue array")
        for line in lines:
            speaker = line.get("speaker") or agent_b.name
            text = (line.get("text") or "...").strip() or "..."
            convo_history.append(f"{speaker}: {text}")
    except Exception as e:
        log.warning("%s<->%s conversation generation failed: %s", agent_a.name, agent_b.name, e)
        convo_history.append(f"{agent_b.name}: ...")
    return convo_history