"""Deterministic world state with banking, energy cycles, RAG memory, and crime."""
from dataclasses import dataclass, field
from typing import Optional
import random
import math
import config
from llm_client import get_embedding

def _is_important_event(entry: str) -> bool:
    """Only conversation/theft/insult/trade/loan events are worth an embedding round-trip -
    routine idle/move/sleep/gather noise is skipped entirely."""
    if entry.startswith("Conversing with") or entry.startswith("Tried to talk to"):
        return True
    if entry.startswith("[") and "]" in entry:
        tag = entry[1:entry.index("]")]
        return tag in config.EMBED_ACTIONS
    return False

@dataclass
class AgentState:
    id: str
    name: str
    age: int
    gender: str
    personality_traits: list
    background: str
    cash: int
    land_owned: int
    current_location: str
    inventory: dict = field(default_factory=lambda: {"wheat": 0, "bread": 2})
    energy: int = 100
    sleep_timer: int = 0
    debt: int = 0
    long_term_goal: str = "Survive and accumulate wealth."
    short_term_goal: str = "Figure out what to do today."
    action_queue: list = field(default_factory=list)  # Fixed mutable default
    plan_status: str = "idle"
    reputation: int = 50  # 0 to 100 scale. Affects loan limits and trust.
    jail_timer: int = 0    # ticks remaining under arrest; forced idle while > 0
    last_action_result: str = "Simulation just started."
    priority_level: str = "normal" # Can be 'normal', 'emergency', 'opportunity'
    plan_expiry_tick: int = 0
    
    short_term_memory: list = field(default_factory=list)
    long_term_memory_vault: list = field(default_factory=list) 
    social_graph: dict = field(default_factory=dict)
    llm: str = "ollama:llama3.2:1b"
    last_retrieval_tick: int = -999  # for memory-retrieval cooldown
    def update_reputation(self, amount: int):
        self.reputation = max(0, min(100, self.reputation + amount))

    def consume_energy(self, amount: int) -> bool:
        """Returns False if the agent is starving (energy <= 0)."""
        self.energy -= amount
        if self.energy <= 0:
            self.energy = 0
            return False
        return True

    async def remember(self, entry: str):
        self.short_term_memory.append(entry)
        self.short_term_memory = self.short_term_memory[-3:]

        if not _is_important_event(entry):
            return  # routine/low-value action, not worth an embedding round-trip

        vector = await get_embedding(entry)
        if vector:
            self.long_term_memory_vault.append({"text": entry, "vector": vector})
            if len(self.long_term_memory_vault) > config.LONG_TERM_MEMORY_MAX:
                self.long_term_memory_vault.pop(0)
            
    async def retrieve_relevant_memories(self, query_text: str, k: int = 3) -> list:
        if not self.long_term_memory_vault:
            return []
            
        query_vector = await get_embedding(query_text)
        if not query_vector:
            return []
            
        scored_memories = []
        for mem in self.long_term_memory_vault:
            dot = sum(a * b for a, b in zip(query_vector, mem["vector"]))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in mem["vector"]))
            
            similarity = 0.0
            if norm_a and norm_b:
                similarity = dot / (norm_a * norm_b)
                
            scored_memories.append((similarity, mem["text"]))
            
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem[1] for mem in scored_memories[:k]]
        
    def update_opinion(self, other_name: str, delta: int):
        current = self.social_graph.get(other_name, 0)
        self.social_graph[other_name] = max(-100, min(100, current + delta))
        
    def expend_energy(self, amount: int):
        self.energy -= amount
        if self.energy <= 0:
            self.energy = 0
            self.sleep_timer = 3 

class WorldState:
    def __init__(self, agent_configs: list):
        self.tick = 0
        self.minutes_elapsed = 0
        self.unclaimed_land = config.TOTAL_LAND_PLOTS
        
        self.market = {
            "wheat": {"supply": 10, "price": 10},
            "bread": {"supply": 5, "price": 30}
        }
        
        self.bank_interest_rate = 0.05
        self.global_event_msg = None
        self.bulletin_board = []
        self.location_logs = {loc: [] for loc in config.LOCATIONS}
        self.history = []  # per-tick metrics snapshots, feeds the dashboard (see dashboard.py)
        
        self.agents = {}
        for cfg in agent_configs:
            inv = {"wheat": 0, "bread": 2}
            if "financials" in cfg:
                if "bread" in cfg["financials"]: inv["bread"] = cfg["financials"]["bread"]
                elif "food" in cfg["financials"]: inv["bread"] = cfg["financials"]["food"]
                
            self.agents[cfg["id"]] = AgentState(
                id=cfg["id"], name=cfg["name"], age=cfg["age"], gender=cfg["gender"],
                personality_traits=cfg["personality_traits"], background=cfg["background"],
                cash=cfg["financials"].get("cash", 0), land_owned=cfg["financials"].get("land_owned", 0),
                current_location=cfg.get("current_location", "Marketplace"),
                inventory=inv, llm=cfg.get("llm", "ollama:llama3.2:1b")
            )

    def clock_str(self) -> str:
        total_min = config.START_HOUR * 60 + self.minutes_elapsed
        h, m = divmod(total_min % (24 * 60), 60)
        return f"Day {total_min // (24 * 60) + 1}, {h:02d}:{m:02d}"

    def agents_at(self, location: str) -> list:
        return [a for a in self.agents.values() if a.current_location == location]

    def recent_dialogue(self, location: str) -> list:
        return self.location_logs[location][-config.DIALOGUE_HISTORY_LIMIT:]

    def apply_move(self, agent: AgentState, target: Optional[str]) -> str:
        if target not in config.LOCATIONS: return f"Failed: '{target}' invalid."
        agent.expend_energy(5)
        agent.current_location = target
        return f"Moved to {target}. Energy: {agent.energy}"

    def apply_speak(self, agent: AgentState, dialogue: Optional[str]) -> str:
        text = (dialogue or "").strip()
        if not text: return "Said nothing."
        self.location_logs[agent.current_location].append(f"{agent.name}: {text}")
        for other in self.agents_at(agent.current_location):
            if other.id != agent.id: other.update_opinion(agent.name, 1)
        return f'Said: "{text}"'
        
    def apply_post_notice(self, agent: AgentState, notice_text: Optional[str]) -> str:
        if agent.current_location != "Marketplace": return "Failed: Must be in Marketplace."
        if agent.cash < 5: return "Failed: Needs $5."
        text = (notice_text or "").strip()
        agent.cash -= 5
        agent.expend_energy(5)
        self.bulletin_board.insert(0, f"[{agent.name}]: {text}")
        self.bulletin_board = self.bulletin_board[:3]
        return f'Posted notice: "{text}"'

    def apply_buy_land(self, agent: AgentState) -> str:
        if agent.current_location != "Marketplace": return "Failed: Must be in Marketplace."
        if self.unclaimed_land <= 0: return "Failed: No land left."
        if agent.cash < config.LAND_PRICE: return "Failed: Insufficient funds."
        agent.cash -= config.LAND_PRICE
        agent.land_owned += 1
        self.unclaimed_land -= 1
        for other in self.agents_at(agent.current_location):
            if other.id != agent.id: other.update_opinion(agent.name, -5)
        return f"Bought land. Now owns {agent.land_owned} plot(s)."

    def apply_gather_wheat(self, agent: AgentState) -> str:
        if agent.current_location != "Wilderness_Commons": return "Failed: Must be in Wilderness."
        agent.expend_energy(15)
        yield_amt = random.randint(1, 3)
        agent.inventory["wheat"] += yield_amt
        return f"Gathered {yield_amt} wheat. Energy: {agent.energy}."

    def apply_bake_bread(self, agent: AgentState) -> str:
        if agent.current_location != "Residential_Quarter": return "Failed: Must bake at home (Residential)."
        if agent.inventory["wheat"] < 1: return "Failed: Need 1 wheat to bake."
        agent.expend_energy(15)
        agent.inventory["wheat"] -= 1
        agent.inventory["bread"] += 1
        return f"Baked 1 bread. Energy: {agent.energy}."

    def apply_eat_bread(self, agent: AgentState) -> str:
        if agent.inventory.get("bread", 0) > 0:
            agent.inventory["bread"] -= 1
            agent.energy = min(100, agent.energy + 40)
            return f"Ate 1 bread. Energy restored to {agent.energy}."
        return "Failed: No bread in inventory."

    def apply_trade(self, agent: AgentState, item: str, trade_type: str) -> str:
        if agent.current_location != "Marketplace": return "Failed: Must be in Marketplace."
        if item not in self.market: return "Failed: Invalid item."
        
        mkt = self.market[item]
        if trade_type == "buy":
            if mkt["supply"] <= 0: return f"Failed: Market out of {item}."
            if agent.cash < mkt["price"]: return f"Failed: Cannot afford {item}."
            agent.cash -= mkt["price"]
            agent.inventory[item] += 1
            mkt["supply"] -= 1
            mkt["price"] = min(config.MAX_PRICE, mkt["price"] + 2)
            return f"Bought {item} for ${mkt['price'] - 2}."
            
        elif trade_type == "sell":
            if agent.inventory[item] <= 0: return f"Failed: No {item} to sell."
            agent.inventory[item] -= 1
            agent.cash += mkt["price"]
            mkt["supply"] += 1
            mkt["price"] = max(5, mkt["price"] - 2)
            return f"Sold {item} for ${mkt['price'] + 2}."

    def apply_idle(self, agent: AgentState) -> str:
        if agent.current_location == "Residential_Quarter":
            agent.energy = min(100, agent.energy + 10)
            return f"Rested at home. Energy: {agent.energy}."
        return "Idled."

    def apply_sleep(self, agent: AgentState) -> str:
        agent.energy = min(100, agent.energy + 35)
        return f"Sleeping... Energy: {agent.energy}."
        
    def apply_steal_cash(self, agent: AgentState, target_name: Optional[str]) -> str:
        if agent.energy < 20: return "Failed: Not enough energy to steal."
        if not target_name: return "Failed: No target specified."
        
        target = next((a for a in self.agents_at(agent.current_location) if a.name == target_name and a.id != agent.id), None)
        if not target: return f"Failed: {target_name} is not here."
        
        agent.expend_energy(20)
        if random.random() > 0.5:
            stolen_amt = min(target.cash, random.randint(10, 50))
            if stolen_amt == 0: return f"Pickpocketed {target.name} but their pockets were empty!"
            target.cash -= stolen_amt
            agent.cash += stolen_amt
            target.action_queue.clear()
            target.plan_status = "interrupted"
            return f"Successfully stole ${stolen_amt} from {target.name} unnoticed."
        else:
            target.update_opinion(agent.name, -50)
            agent.update_reputation(-10)  # getting caught is public; hits creditworthiness too, not just this relationship
            self.location_logs[agent.current_location].append(f"*** ALERT: {agent.name} was caught trying to rob {target.name}! ***")
            return f"CAUGHT trying to steal from {target.name}! They now despise you and your reputation suffers."

    def apply_insult(self, agent: AgentState, target_name: Optional[str], insult_text: Optional[str]) -> str:
        if not target_name: return "Failed: No target specified."
        target = next((a for a in self.agents_at(agent.current_location) if a.name == target_name and a.id != agent.id), None)
        if not target: return f"Failed: {target_name} is not here."
        
        text = (insult_text or "You are worthless.").strip()
        self.location_logs[agent.current_location].append(f"{agent.name} (to {target.name}): {text}")
        target.update_opinion(agent.name, -30)
        return f'Insulted {target.name}: "{text}"'

    def apply_take_loan(self, agent: AgentState, amount: int) -> str:
        if not amount or amount <= 0:
            return "Failed: Invalid loan amount."
        if agent.current_location != "Marketplace":
            return "Failed: Must be at the Marketplace to visit the Bank."
        
        # Max loan is directly tied to reputation (e.g., 50 rep = $500 max)
        max_loan = agent.reputation * 10
        
        if agent.debt + amount > max_loan:
            return f"Failed: The bank refuses. With a reputation of {agent.reputation}, your credit limit is ${max_loan}."
        
        agent.cash += amount
        agent.debt += amount
        return f"Took a loan of ${amount}. Total debt is now ${agent.debt}."

    def apply_repay_loan(self, agent: AgentState, amount: int) -> str:
        if not amount or amount <= 0:
            return "Failed: Invalid repayment amount."
        if agent.current_location != "Marketplace":
            return "Failed: Must be at the Marketplace to repay loans."
        if agent.debt <= 0:
            return "Failed: You have no debt."
        
        actual_repayment = min(amount, agent.debt, agent.cash)
        if actual_repayment <= 0:
            return "Failed: Not enough cash to make a payment."
            
        agent.cash -= actual_repayment
        agent.debt -= actual_repayment
        
        # Consistent repayment builds institutional trust
        if actual_repayment >= 50:
            agent.update_reputation(2)
            
        return f"Repaid ${actual_repayment}. Remaining debt: ${agent.debt}. Reputation increased."

    def trigger_stochastic_event(self):
        r = random.random()
        self.global_event_msg = None
        if r < 0.05:
            self.market["wheat"]["supply"] += 10
            self.market["wheat"]["price"] = max(5, self.market["wheat"]["price"] - 5)
            self.global_event_msg = "EVENT: A bountiful harvest crashed wheat prices!"
        elif r < 0.10:
            self.market["bread"]["supply"] = max(0, self.market["bread"]["supply"] - 5)
            self.market["bread"]["price"] += 15
            self.global_event_msg = "EVENT: A fire in the market ruined bread supplies. Prices spiked!"
        elif r < 0.15:
            for a in self.agents.values(): a.cash += 50
            self.global_event_msg = "EVENT: The Mayor distributed a $50 stimulus to all citizens."

    async def advance_tick(self):
        self.tick += 1
        
        # Debt compounds at 5% per tick
        for agent in self.agents.values():
            if agent.debt > 0:
                agent.debt = int(agent.debt * 1.05)
                
                # Asset Seizure & Reputation Crash if debt exceeds dynamic limit
                max_loan = agent.reputation * 10
                if agent.debt > max_loan:
                    agent.cash = max(0, agent.cash - agent.debt)
                    agent.inventory["wheat"] = 0
                    agent.update_reputation(-15)
                    agent.debt = 0
                    await agent.remember("The bank seized my assets and ruined my reputation due to extreme debt.")
                    # NOTE: global_event_msg is wiped at the top of the *next* tick's
                    # trigger_stochastic_event() before it's ever printed - it never
                    # reached the console. Logging to location_logs instead so it
                    # actually surfaces (and persists in per-location history).
                    self.location_logs[agent.current_location].append(f"*** BANK SEIZURE: {agent.name}'s assets were seized! ***")

            # Constant Resource Pressure: Agents lose 5 energy per tick
            is_alive = agent.consume_energy(5)
            
            if not is_alive:
                # Starvation mechanics: auto-consume bread if available
                if agent.inventory["bread"] > 0:
                    agent.inventory["bread"] -= 1
                    agent.energy += 40
                    await agent.remember("I was starving, so I urgently ate bread from my inventory.")
                else:
                    # Penalties for starving: lose reputation and cannot move/work well
                    agent.update_reputation(-1)
                    await agent.remember("I am starving and have no food. I need to find bread immediately.")
                    
                    # Force their queue to clear so they have to re-plan for survival
                    agent.action_queue.clear()
                    agent.plan_status = "interrupted"

        self._run_sheriff_check()
        self.record_snapshot()

    def _run_sheriff_check(self):
        """Law enforcement: reputation crashing below the threshold (theft, repeated debt
        seizures, insults) gets an agent arrested. Jailed agents lose their turn for
        JAIL_DURATION ticks and get a small reputation credit on release so the penalty
        isn't a permanent lock-out of the loan/trade systems."""
        for agent in self.agents.values():
            if agent.jail_timer > 0:
                continue  # already serving time
            if agent.reputation < config.SHERIFF_REPUTATION_THRESHOLD:
                agent.jail_timer = config.JAIL_DURATION
                agent.current_location = config.JAIL_LOCATION
                agent.action_queue.clear()
                agent.plan_status = "jailed"
                self.location_logs[config.JAIL_LOCATION].append(
                    f"*** SHERIFF: {agent.name} was arrested for reputation {agent.reputation} ***"
                )

    def record_snapshot(self):
        """Appends one row of per-tick metrics for the dashboard. Cheap: just reads
        already-computed fields, no extra LLM/embedding calls."""
        snapshot = {
            "tick": self.tick,
            "clock": self.clock_str(),
            "market": {item: dict(data) for item, data in self.market.items()},
            "agents": {
                a.id: {
                    "name": a.name, "cash": a.cash, "reputation": a.reputation,
                    "debt": a.debt, "energy": a.energy, "location": a.current_location,
                    "jailed": a.jail_timer > 0,
                }
                for a in self.agents.values()
            },
        }
        self.history.append(snapshot)
        if len(self.history) > config.SNAPSHOT_HISTORY_MAX:
            self.history.pop(0)

    def to_dict(self) -> dict:
        """Serializes full world + agent state for save/resume. Excludes nothing -
        long_term_memory_vault embeddings are plain float lists and round-trip fine."""
        return {
            "tick": self.tick,
            "minutes_elapsed": self.minutes_elapsed,
            "unclaimed_land": self.unclaimed_land,
            "market": self.market,
            "bank_interest_rate": self.bank_interest_rate,
            "bulletin_board": self.bulletin_board,
            "location_logs": self.location_logs,
            "history": self.history,
            "agents": {aid: vars(a).copy() for aid, a in self.agents.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        """Rebuilds a WorldState from to_dict() output without re-running agent_configs
        (agents.json is only needed for a fresh run, not a resume)."""
        world = cls.__new__(cls)
        world.tick = data["tick"]
        world.minutes_elapsed = data["minutes_elapsed"]
        world.unclaimed_land = data["unclaimed_land"]
        world.market = data["market"]
        world.bank_interest_rate = data["bank_interest_rate"]
        world.global_event_msg = None
        world.bulletin_board = data["bulletin_board"]
        world.location_logs = data["location_logs"]
        world.history = data.get("history", [])
        world.agents = {aid: AgentState(**fields) for aid, fields in data["agents"].items()}
        return world

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "WorldState":
        import json
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))