import json
import argparse
import os
import asyncio
import logging
import config
from state import WorldState
from llm_client import get_client_for_agent, get_stats, close_session # Assumed these are exported from llm_client in your local build
from simulation import Simulation
from historian import SimulationHistorian

log = logging.getLogger("main")


def setup_logging(log_file: str, debug: bool = False) -> None:
    """Console + file both show the tick narrative (agent thoughts/actions/conversations)
    at INFO. Retry-in-progress noise (429 backoff, transport retries) is logged at DEBUG,
    so it's invisible here by default - pass --debug to see it for troubleshooting. Real
    failures (retries exhausted, unparseable JSON, fatal API errors) are WARNING/ERROR and
    always show, regardless of --debug."""
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def load_agents(path="agents.json") -> list:
    with open(path) as f:
        return json.load(f)

async def run_analytics(log_file_path: str, client, facts_str: str) -> str:
    """Reads the simulation logs and prompts the LLM with exact state diffs to prevent math hallucinations."""
    if not os.path.exists(log_file_path):
        return "No log file found to analyze."
        
    with open(log_file_path, "r", encoding="utf-8") as f:
        logs = f.read()
        
    # --- PREVENT 400 BAD REQUEST (GROQ PAYLOAD LIMITS) ---
    # Caps the string to the last ~25,000 characters (roughly 6,000 tokens)
    if len(logs) > 25000:
        logs = "... [EARLIER LOGS TRUNCATED DUE TO API LIMITS] ...\n" + logs[-25000:]
        
    system_prompt = (
        "You are an expert sociologist, economist, and city historian. "
        "Your task is to analyze the simulation logs and compile a strictly factual 'Important Events Report'. "
        "CRITICAL INSTRUCTION: You are provided with COMPUTED MATHEMATICAL FACTS. "
        "Treat these facts as absolute truth. "
        "Use the raw simulation logs ONLY to explain the social narrative of HOW these mathematical facts occurred. "
        "Do not calculate numbers yourself."
    )
    
    user_prompt = f"""
=== COMPUTED MATHEMATICAL FACTS (DO NOT CONTRADICT THESE) ===
{facts_str}

=== RAW SIMULATION LOGS ===
{logs}
=== END OF LOGS ===

Analyze the data and extract the most significant developments. Focus strictly on:
1. Economic Milestones: Explain how the agents gained or lost their cash/inventory based on the facts provided.
2. Dynamic Market Shifts: How did agents react to changing prices?
3. Social Dynamics: Notable conversations, agreements, or emerging patterns.
4. Character Highlights: Did anyone act completely in line with or break their defined personality traits?

Provide a clean, bulleted historical summary. Group by categories. Do not hallucinate numbers.
"""
    log.info("[Engine] Sending logs and state diffs to the LLM for historical analysis... Please wait...")
    try:
        summary = await client.generate_text(system_prompt, user_prompt)
        return summary
    except Exception as e:
        return f"Analysis failed or model output format error: {str(e)}"


async def async_main():
    parser = argparse.ArgumentParser(description="Textual multi-agent city simulation (V1)")
    parser.add_argument("--ticks", type=int, default=10, help="Number of ticks to run")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-tick logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to a snapshot to resume from")
    parser.add_argument("--save", type=str, default=None, help="Path to write a snapshot to after the run")
    parser.add_argument("--dashboard", type=str, default=None, help="Path to write an HTML metrics dashboard after the run")
    parser.add_argument("--debug", action="store_true", help="Show HTTP retry/rate-limit noise (hidden by default)")
    args = parser.parse_args()

    log_file = "simulation_log.txt"
    setup_logging(log_file, debug=args.debug)
    log.info("=== SIMULATION RUN LOG ===")

    world = WorldState.load(args.resume) if args.resume else WorldState(load_agents())
    sim = Simulation(world)
    
    # --- TAKE STARTING SNAPSHOT ---
    initial_states = {
        a.id: {"cash": a.cash, "wheat": a.inventory.get('wheat',0), "bread": a.inventory.get('bread',0)} 
        for a in world.agents.values()
    }
    
    # Await the simulation loop
    await sim.run(num_ticks=args.ticks, verbose=not args.quiet)

    log.info("")
    log.info("===== FINAL STATE =====")

    # --- COMPUTE FACTS FOR LLM ---
    computed_facts = []
    for a in world.agents.values():
        start = initial_states[a.id]
        cash_delta = a.cash - start["cash"]
        wheat_delta = a.inventory.get('wheat', 0) - start["wheat"]
        bread_delta = a.inventory.get('bread', 0) - start["bread"]
        
        fact = f"{a.name}: "
        fact += f"Gained ${cash_delta} " if cash_delta >= 0 else f"Lost ${abs(cash_delta)} "
        fact += f"| Net Wheat: {wheat_delta} | Net Bread: {bread_delta}"
        computed_facts.append(fact)

        log.info("%s: cash=$%s, wheat=%s, bread=%s, debt=$%s",
                  a.name, a.cash, a.inventory.get('wheat', 0), a.inventory.get('bread', 0), a.debt)

    facts_str = "\n".join(computed_facts)

    log.info("Unclaimed land remaining: %s", world.unclaimed_land)
    log.info("Final Market Wheat Price: $%s (Supply: %s)", world.market['wheat']['price'], world.market['wheat']['supply'])
    log.info("Final Market Bread Price: $%s (Supply: %s)", world.market['bread']['price'], world.market['bread']['supply'])

    if hasattr(sim, 'print_profile_summary'):
        sim.print_profile_summary()

    if args.save:
        world.save(args.save)
        log.info("[Engine] Snapshot saved to %s", args.save)

    if args.dashboard:
        from dashboard import render_dashboard
        render_dashboard(world.history, args.dashboard)
        log.info("[Engine] Dashboard written to %s", args.dashboard)

    try:
        stats = get_stats()
        log.info("")
        log.info("----- LLM / EMBEDDING CALL STATS -----")
        log.info("LLM calls: %s  total_time=%.2fs  rate_limit_hits=%s  json_parse_failures=%s",
                  stats['llm_calls'], stats['llm_time'], stats.get('rate_limit_hits', 0), stats.get('json_parse_failures', 0))
        log.info("Embedding calls: %s  total_time=%.2fs  cache_hits=%s",
                  stats['embed_calls'], stats['embed_time'], stats['embed_cache_hits'])
    except Exception:
        pass  # Fails silently if get_stats() is undefined in your local environment

    try:
        if os.environ.get(config.GROQ_API_KEY_ENV):
            # Was "openai/gpt-oss-120b" - missing the "groq:" provider prefix, which is
            # exactly what threw "not enough values to unpack (expected 2, got 1)" out
            # of get_client_for_agent's llm_string.split(":", 1).
            historian_client = get_client_for_agent("groq:openai/gpt-oss-120b")
        else:
            historian_client = get_client_for_agent("ollama:llama3.2:1b")
        
        # Instantiate the map-reduce historian. chunk_chars bounds how many characters
        # of raw log go into a single "map" summarization call.
        historian = SimulationHistorian(client=historian_client, chunk_chars=12000)
        
        # Define what you want the historian to focus on
        focus_prompt = (
                "Write a comprehensive, multi-paragraph chronicle of the simulation. "
                "Focus on the main economic shifts, the alliances formed or broken, "
                "and the ultimate fate of the most active agents."
            )
            
        log_path = "simulation_log.txt" 
            
        final_chronicle = await historian.compile_final_history(
            log_file_path=log_path, 
            historical_focus=focus_prompt
            )
        log.info("")
        log.info("=== FINAL HISTORICAL CHRONICLE ===")
        log.info(final_chronicle)

        with open("final_history_report.txt", "w", encoding="utf-8") as f:
            f.write(final_chronicle)

    except Exception as e:
        log.error("Error compiling history: %s", e)
    finally:
        # Gracefully shut down the aiohttp session upon completion or crash
        await close_session()

if __name__ == "__main__":
    asyncio.run(async_main())