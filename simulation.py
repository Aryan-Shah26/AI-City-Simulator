"""Discrete-tick simulation loop with batched inference, sleep optimization, and dynamic LLM routing."""
import time
import logging
import config
import asyncio
import collections
import agent as agent_module
from state import WorldState
from llm_client import get_client_for_agent

log = logging.getLogger("simulation")

ACTION_HANDLERS = {
    "move": lambda w, a, d: w.apply_move(a, d.get("target")),
    "speak": lambda w, a, d: w.apply_speak(a, d.get("dialogue")),
    "gather_wheat": lambda w, a, d: w.apply_gather_wheat(a),
    "bake_bread": lambda w, a, d: w.apply_bake_bread(a),
    "eat_bread": lambda w, a, d: w.apply_eat_bread(a),
    "buy_wheat": lambda w, a, d: w.apply_trade(a, "wheat", "buy"),
    "sell_wheat": lambda w, a, d: w.apply_trade(a, "wheat", "sell"),
    "buy_bread": lambda w, a, d: w.apply_trade(a, "bread", "buy"),
    "sell_bread": lambda w, a, d: w.apply_trade(a, "bread", "sell"),
    "post_notice": lambda w, a, d: w.apply_post_notice(a, d.get("notice_text")),
    "steal_cash": lambda w, a, d: w.apply_steal_cash(a, d.get("target_name")),
    "insult": lambda w, a, d: w.apply_insult(a, d.get("target_name"), d.get("dialogue")),
    "idle": lambda w, a, d: w.apply_idle(a),
    "take_loan": lambda w, a, d: w.apply_take_loan(a, d.get("amount")),
    "repay_loan": lambda w, a, d: w.apply_repay_loan(a, d.get("amount")),
    "buy_land": lambda w, a, d: w.apply_buy_land(a),
}


class Simulation:
    def __init__(self, world: WorldState):
        self.world = world
        # Cumulative per-stage profiling across the whole run.
        self.profile = {"decisions": 0.0, "conversations": 0.0, "actions": 0.0, "ticks": 0}

    async def run_tick(self, verbose: bool = True):
        w = self.world
        w.trigger_stochastic_event()

        if verbose:
            log.info("")
            log.info("===== %s (tick %d) =====", w.clock_str(), w.tick)
            if w.global_event_msg:
                log.info("*** %s ***", w.global_event_msg)

        acted_this_tick = set()

        # Step 1: Handle sleepers and jailed agents, and collect awake/free agents across
        # ALL locations up front. Both cost this tick's turn with no LLM call.
        active_by_location = {}
        for location in config.LOCATIONS:
            active_agents = []
            for a in w.agents_at(location):
                if a.jail_timer > 0:
                    a.jail_timer -= 1
                    acted_this_tick.add(a.id)
                    if a.jail_timer == 0:
                        a.update_reputation(config.JAIL_REPUTATION_RESTORE)
                        a.plan_status = "idle"
                        await a.remember("[jail] Served my time and was released.")
                        if verbose: log.info("- %s @ %s: Released from jail (reputation restored).", a.name, location)
                    else:
                        await a.remember(f"[jail] {a.jail_timer} tick(s) left.")
                        if verbose: log.info("- %s @ %s: In jail, %d tick(s) left.", a.name, location, a.jail_timer)
                elif a.sleep_timer > 0:
                    a.sleep_timer -= 1
                    result = w.apply_sleep(a)
                    await a.remember(f"[sleep] {result}")
                    acted_this_tick.add(a.id)
                    if verbose: log.info("- %s @ %s: Forced Sleep -> %s", a.name, location, result)
                else:
                    active_agents.append(a)
            active_by_location[location] = active_agents

        all_active = [a for agents in active_by_location.values() for a in agents]

        # Step 2: Skip the LLM entirely for agents that already have a valid queued action;
        # only the remainder need a real inference call.
        decision_start = time.perf_counter()
        resolved = []       
        needs_llm = []
        for a in all_active:
            # ---> FIX: Pass 'w' (the world state) as the second argument <---
            queued = agent_module.try_queued_action(a, w)
            if queued is not None:
                resolved.append((a, None, queued))
            else:
                needs_llm.append(a)

        async def fetch_decision(agent):
            client = get_client_for_agent(agent.llm)  # cached instance, reused across ticks
            dec = await agent_module.plan_new_action(agent, w, client)
            return agent, client, dec

        if config.GROUP_BY_MODEL:
            # Batch by model so Ollama isn't thrashed swapping between different local models
            # mid-tick; agents sharing a model still run fully concurrently within their group.
            groups = collections.OrderedDict()
            for a in needs_llm:
                groups.setdefault(a.llm, []).append(a)
            for group_agents in groups.values():
                BATCH_SIZE = 2
                for i in range(0, len(group_agents), BATCH_SIZE):
                    batch = group_agents[i:i+BATCH_SIZE]
                    resolved += await asyncio.gather(
                        *(fetch_decision(a) for a in batch)
                    )
                    
        else:
            resolved += await asyncio.gather(*[fetch_decision(a) for a in needs_llm])
        self.profile["decisions"] += time.perf_counter() - decision_start

        results_by_location = {loc: [] for loc in config.LOCATIONS}
        for a, client_a, decision in resolved:
            results_by_location[a.current_location].append((a, client_a, decision))

        # Step 3: Resolve actions location by location, sequentially, to prevent state conflicts
        conv_time = 0.0
        action_time = 0.0
        for location in config.LOCATIONS:
            agents_in_room = w.agents_at(location)
            for a, client_a, decision in results_by_location[location]:
                if a.id in acted_this_tick:
                    continue  # Was pulled into a conversation by an earlier agent

                action_str = decision.get("action", "idle")

                if action_str == "converse_with":
                    t0 = time.perf_counter()
                    target_name = decision.get("target_name")
                    target = next((tgt for tgt in agents_in_room if tgt.name == target_name and tgt.id != a.id), None)

                    if not target or target.id in acted_this_tick or target.sleep_timer > 0 or target.jail_timer > 0:
                        await a.remember(f"Tried to talk to {target_name}, but they were busy/gone.")
                        acted_this_tick.add(a.id)
                        conv_time += time.perf_counter() - t0
                        continue

                    if client_a is None:  # agent reached converse_with via a queued action
                        client_a = get_client_for_agent(a.llm)

                    initial_statement = decision.get("dialogue") or "Hello."
                    # ONE LLM call writes the whole exchange (was 2-4 sequential forced-reply calls).
                    convo_history = await agent_module.generate_conversation(a, target, w, initial_statement, client_a)

                    full_transcript = " | ".join(convo_history)
                    w.location_logs[location].append(f"[CONVO] {full_transcript}")
                    await a.remember(f"Conversing with {target.name}: {full_transcript}")
                    await target.remember(f"Conversing with {a.name}: {full_transcript}")

                    a.update_opinion(target.name, 2)
                    target.update_opinion(a.name, 2)

                    if verbose:
                        thought = decision.get("thought", "")
                        if thought:
                            log.info("- %s thought: %s", a.name, thought)
                        log.info("- [CONVO] %s <-> %s:\n    %s", a.name, target.name, "\n    ".join(convo_history))

                    acted_this_tick.add(a.id)
                    acted_this_tick.add(target.id)
                    conv_time += time.perf_counter() - t0

                else:
                    t0 = time.perf_counter()
                    handler = ACTION_HANDLERS.get(action_str, ACTION_HANDLERS["idle"])
                    
                    # Execute the action deterministically
                    result = handler(w, a, decision)
                    
                    # Store exact feedback for the next planning cycle
                    a.last_action_result = f"Attempted '{action_str}': {result}"

                    await a.remember(f"[{action_str}] {result}")

                    if verbose:
                        thought = decision.get("thought", "")
                        if thought:
                            log.info("- %s @ %s thought: %s", a.name, location, thought)
                        log.info("- %s @ %s: %s -> %s", a.name, location, action_str, result)

                    acted_this_tick.add(a.id)
                    action_time += time.perf_counter() - t0

        self.profile["conversations"] += conv_time
        self.profile["actions"] += action_time
        self.profile["ticks"] += 1

        log.debug("[profile] decisions=%.2fs total | this tick: conv=%.2fs action=%.2fs",
                  self.profile["decisions"], conv_time, action_time)

        await w.advance_tick()

    async def run(self, num_ticks: int, verbose: bool = True):
        for _ in range(num_ticks):
            await self.run_tick(verbose=verbose)

    def print_profile_summary(self):
        p = self.profile
        n = max(p["ticks"], 1)
        log.info("")
        log.info("----- PROFILE SUMMARY -----")
        log.info("Ticks run: %d", p["ticks"])
        log.info("Decision (LLM) time:   total=%.2fs  avg/tick=%.2fs", p["decisions"], p["decisions"]/n)
        log.info("Conversation time:     total=%.2fs  avg/tick=%.2fs", p["conversations"], p["conversations"]/n)
        log.info("Action execution time: total=%.2fs  avg/tick=%.2fs", p["actions"], p["actions"]/n)
        log.info("Avg tick time (profiled stages): %.2fs", (p["decisions"]+p["conversations"]+p["actions"])/n)