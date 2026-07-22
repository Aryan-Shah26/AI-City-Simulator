"""Central constants. Change values here, not in logic files."""

# --- Time ---
TICK_MINUTES = 15
START_HOUR = 8  # simulation starts at 08:00

# --- Locations ---
LOCATIONS = ["Marketplace", "Residential_Quarter", "Wilderness_Commons"]

# --- Economy (Marketplace) ---
TOTAL_LAND_PLOTS = 10
LAND_PRICE = 500
WORK_WAGE = 50  # earned only when action=work AND location=Marketplace
MAX_PRICE = 200  # ceiling on buy-side price drift, mirrors the existing floor of 5

# --- Law enforcement ---
SHERIFF_REPUTATION_THRESHOLD = 15  # reputation at/below this triggers arrest
JAIL_DURATION = 4                  # ticks an arrested agent loses their turn
JAIL_LOCATION = "Marketplace"      # where jailed agents are held (town jail)
JAIL_REPUTATION_RESTORE = 10       # partial rehab on release, avoids permanent lock-out

# --- Persistence ---
SNAPSHOT_HISTORY_MAX = 500  # cap on per-tick metrics kept in WorldState.history for the dashboard

# --- Memory / context windows ---
DIALOGUE_HISTORY_LIMIT = 4   # last N lines of location dialogue shown to agents (was 8, trimmed for prompt size)
AGENT_MEMORY_LIMIT = 6       # last N of an agent's own past actions
RETRIEVED_MEMORIES_LIMIT = 2 # cap on RAG memories injected into prompt (was 3)

# --- RAG / embedding cost controls ---
EMBED_ACTIONS = {  # only these action types are worth vectorizing into long-term memory
    "converse_with", "steal_cash", "insult", "buy_wheat", "sell_wheat",
    "buy_bread", "sell_bread", "take_loan", "repay_loan", "buy_land",
}
LONG_TERM_MEMORY_MAX = 40          # cap per-agent vault size (oldest dropped)
MEMORY_RETRIEVAL_COOLDOWN = 3      # ticks between RAG retrievals for a non-conversation agent
EMBEDDING_CACHE_SIZE = 500         # dedupe identical-text embedding calls

# --- Planning ---
PLAN_HORIZON = 6   # requested plan length from the LLM (was implicitly 1 step at a time)
CONVERSATION_LINES = 3  # additional lines generated after the opening line, in one LLM call

# --- Concurrency ---
MAX_CONCURRENT_LLM_REQUESTS = 2   # asyncio.Semaphore cap to avoid overwhelming local Ollama / rate limits
GROUP_BY_MODEL = True             # batch requests per llm model to reduce Ollama model-swap thrashing
GROQ_REQUESTS_PER_MINUTE = 28     # stay under Groq free-tier's ~30 RPM; Ollama (local) is not rate-limited at all
MAX_429_RETRIES = 6               # separate, larger budget than LLM_TRANSPORT_RETRIES - 429 is expected/recoverable,
                                   # unlike a real connection failure, so it deserves more patience

# --- LLM ---
LLM_PROVIDER = "ollama"          # "ollama" | "groq"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST = "http://localhost:11434"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 900               # headroom for a full plan (horizon 3-6 steps) to finish
                                    # without truncating mid-JSON, which independently
                                    # produces invalid JSON regardless of horizon size
LLM_MAX_RETRIES = 2
LLM_TRANSPORT_RETRIES = 2         # retries for connection-level failures (separate from schema-retry in agent.py)

VALID_ACTIONS = {"move", "speak", "gather_wheat", "bake_bread", "eat_bread", 
    "buy_wheat", "sell_wheat", "buy_bread", "sell_bread", 
    "post_notice", "converse_with", "idle",
    "insult", "steal_cash", "take_loan", "repay_loan", "buy_land"}