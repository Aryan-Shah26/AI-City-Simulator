# Multi-Agent Sim V1

## Setup
```
pip install pydantic httpx
export GROQ_API_KEY=...   # matches "groq:" prefix in agents.json -> llm field
```

## Run
```
python main.py
```

## Files
| File | Role |
|---|---|
| `models.py` | Pydantic schemas: `AgentConfig`, `AgentRuntimeState`, `AgentAction`, `MemoryEvent`, `BoardPost` |
| `llm_client.py` | Provider-agnostic async LLM wrapper, parses `"provider:model"` (e.g. `groq:openai/gpt-oss-20b`), temp=0 default |
| `memory.py` | Short-term rolling buffer + LLM-summarized long-term memory |
| `economy.py` | Trade + robbery resolution, reputation/opinion fallout |
| `agent.py` | LLM-driven planning (every `plan_interval` ticks), action selection, hardcapped 10-turn conversation |
| `world.py` | Tick orchestrator + async `CommunityBoard` |
| `agents.json` | Agent roster (your schema, loaded as-is) |
| `main.py` | Entry point |

## Config knobs (top of `main.py`)
- `NUM_TICKS` — sim length
- `PLAN_INTERVAL` — ticks between re-planning
- `MEMORY_BUFFER_SIZE` — short-term buffer size before LLM consolidation

## Notes
- `agents.json` currently has 1 agent. Append more objects (same schema) — trade/steal/converse
  all require a target agent to interact with.
- To add an LLM provider: add its endpoint + API-key env var to `PROVIDER_ENDPOINTS` /
  `PROVIDER_KEY_ENV` in `llm_client.py`.
- Deceit (`is_deceptive`) is recorded as private ground truth on the action; V1 doesn't
  auto-detect lies — wire `Economy.register_deceit_penalty` into your reveal logic when ready.