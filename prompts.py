import config
from state import WorldState, AgentState

def get_legal_actions(agent: AgentState) -> list:
    """Deterministically calculates what the agent is allowed to do right now."""
    actions = ["idle", "speak", "converse_with", "insult", "steal_cash"]
    
    # Location-specific rules
    if agent.current_location == "Marketplace":
        actions.extend(["buy_wheat", "sell_wheat", "buy_bread", "sell_bread", "take_loan", "repay_loan", "buy_land", "post_notice"])
        actions.append("move (Target MUST be: Residential_Quarter or Wilderness_Commons)")
    elif agent.current_location == "Residential_Quarter":
        actions.extend(["bake_bread", "eat_bread"])
        actions.append("move (Target MUST be: Marketplace or Wilderness_Commons)")
    elif agent.current_location == "Wilderness_Commons":
        actions.extend(["gather_wheat"])
        actions.append("move (Target MUST be: Marketplace or Residential_Quarter)")
        
    return actions

def get_dynamic_horizon(agent: AgentState) -> int:
    """Varies planning depth based on model capability, not provider. Any 'groq:...' string
    used to match here, including small/fast models like llama-3.1-8b-instant - giving them
    the same 6-step nested plan as an actual 70b+ model. That's a lot of structured JSON for
    a fast instant model to hold together at temperature 0.8, and a likely cause of both
    frequent unparseable output and outright 400 json_validate_failed errors from Groq's
    JSON-mode grammar decoder. Key on model size instead."""
    llm_lower = agent.llm.lower()
    if any(tag in llm_lower for tag in ("70b", "120b", "405b")):
        return 6
    return 3

SYSTEM_PROMPT = (
    "You are the cognitive driver for a character in a simulation. "
    "Respond with ONLY a valid JSON object, no markdown, no commentary."
)

CONVO_SYSTEM_PROMPT = (
    "You write short, in-character back-and-forth dialogue between two characters "
    "in a simulation. Respond with ONLY a valid JSON object, no markdown, no commentary."
)

RESPONSE_SCHEMA = """{
  "reflection": "<Analyze the [PREVIOUS ACTION OUTCOME]. Did it succeed or fail? Why?>",
  "long_term_goal": "<Your overarching ambition, grounded in your history>",
  "short_term_goal": "<What to accomplish next, adjusting for recent successes/failures>",
  "plan": [
    {
      "action": "<primary action choice>",
      "contingency_action": "<fallback action if primary is blocked (e.g. idle, move)>",
      "amount": <integer or null>,
      "target_name": "<exact name or null>",
      "target": "<location name or null>",
      "dialogue": "<what to say or null>"
    }
  ]
}"""

DIALOGUE_RULE = (
    "IMPORTANT: for \"speak\" and \"converse_with\", the \"dialogue\" field must be "
    "actual in-character words the agent says out loud - never null, never an empty "
    "string, and never just \"...\". If someone else is at your location, prefer "
    "\"converse_with\" (a real back-and-forth exchange) over \"speak\" (a one-line "
    "remark to no one in particular)."
)


def _char_block(agent: AgentState) -> str:
    """Static per-agent description (name/traits/background never change) - computed once
    and cached on the agent instance instead of re-formatted on every single prompt build."""
    cached = getattr(agent, "_char_block_cache", None)
    if cached is None:
        cached = f"{agent.name} | Traits: {', '.join(agent.personality_traits)} | {agent.background}"
        agent._char_block_cache = cached
    return cached


def build_user_prompt(agent: AgentState, world: WorldState, force_reply_to: list = None, retrieved_memories: list = None, last_result: str = None) -> str:
    legal_actions = get_legal_actions(agent)
    actions_str = "\n".join([f"- {a}" for a in legal_actions])
    others_here = [f"{a.name} (Opinion: {agent.social_graph.get(a.name, 0)})" for a in world.agents_at(agent.current_location) if a.id != agent.id]
    
    horizon = get_dynamic_horizon(agent)

    return f"""
[TIME] {world.clock_str()} (tick {world.tick})
[CHARACTER] {agent.name} | Traits: {", ".join(agent.personality_traits)}

[PREVIOUS ACTION OUTCOME]
{agent.last_action_result}
(You MUST score and reflect on this in your "reflection" field before planning.)

[CURRENT PRIORITY STATE]
{agent.priority_level.upper()}
(If EMERGENCY, abandon long-term goals and plan solely for survival/debt-relief).

[YOUR PHYSICAL STATE] 
Location: {agent.current_location} 
Energy: {agent.energy}/100 
Cash: ${agent.cash} | Debt: ${agent.debt}
Inventory: {agent.inventory.get('wheat', 0)} Wheat | {agent.inventory.get('bread', 0)} Bread

[LEGAL ACTIONS AT THIS LOCATION]
{actions_str}

[WHO IS HERE] {", ".join(others_here) if others_here else "No one."}
[RECENT CHAT] {chr(10).join(world.recent_dialogue(agent.current_location)) or "(none)"}

Generate your reflection, goal, and plan.
Your "plan" array must contain exactly {horizon} steps. 
Factor in Opportunity Cost: If market prices are vastly different than expected, change your strategy.
"""


def build_conversation_prompt(agent_a: AgentState, agent_b: AgentState, world: WorldState, opening_line: str, turns: int) -> str:
    """Single-shot prompt asking one model to write BOTH sides of a short exchange -
    replaces the old 4-round-trip forced-reply lock. Compromise: one model improvises
    both personas instead of each agent's own model replying in turn."""
    return f"""
[SCENE] {world.clock_str()} at {agent_a.current_location}
[CHARACTER A] {_char_block(agent_a)} (Opinion of B: {agent_a.social_graph.get(agent_b.name, 0)})
[CHARACTER B] {_char_block(agent_b)} (Opinion of A: {agent_b.social_graph.get(agent_a.name, 0)})
[OPENING LINE] {agent_a.name}: {opening_line}

Continue this conversation for {turns} more line(s) total, alternating speakers starting
with {agent_b.name}, staying true to each character's personality and current opinion of
the other. Output ONLY JSON:
{{"dialogue": [{{"speaker": "<exact name>", "text": "<line>"}}, ...]}}
"""