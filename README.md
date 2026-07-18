# 🧠 LangGraph + Streamlit + PostgreSQL Memory Chatbot

> **Production-ready project architecture** for a stateful AI chatbot that remembers conversations forever via PostgreSQL checkpoints, served with Streamlit, and fully containerized with Docker for easy VPS deployment.

---

## 🏗 Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Your VPS / Server                           │
│  ┌──────────────────────┐       ┌──────────────────────────────┐  │
│  │   Streamlit (app)    │       │      PostgreSQL 16           │  │
│  │                      │       │                              │  │
│  │  • Beautiful UI      │──────▶│  • LangGraph Checkpoints     │  │
│  │  • Streaming replies │  SQL  │  • Threaded conversation     │  │
│  │  • Session mgmt      │       │    state (thread_id)         │  │
│  │  • thread_id resume  │       │  • Persistent across restarts│  │
│  └──────────┬───────────┘       └──────────────────────────────┘  │
│             │                                                      │
│             │  LangGraph Agent (inside Streamlit process)          │
│             │  • StateGraph + Tools                                │
│             │  • PostgresSaver (via connection pool)               │
│             │  • OpenAI-compatible LLM                             │
└────────────────────────────────────────────────────────────────────┘
```

### Why this architecture?

| Decision | Rationale |
|----------|-----------|
| **Postgres as memory** | LangGraph's official production checkpointer. Survives container restarts, horizontal scaling possible later. |
| **Streamlit embedded agent** | Simplest possible deploy for VPS (one container + db). No need for separate FastAPI unless you want multi-client or heavy scale. |
| **Connection pool** | Streamlit can have many concurrent sessions; pool prevents connection exhaustion. |
| **thread_id as conversation key** | Users can resume any past chat by ID. Perfect for "save this conversation" UX. |
| **Docker Compose** | One command deploys everything on any VPS. Volumes keep data safe. Healthchecks make restarts reliable. |
| **Multi-stage Dockerfile** | Tiny final image, non-root user, healthcheck endpoint. |
| **Clean separation** | `agent/` is pure LangGraph (testable without Streamlit). UI only invokes the compiled graph. |

### Project Layout

```
langgraph-streamlit-memory/
├── docker-compose.yml          # Orchestrates app + postgres
├── Dockerfile                  # Multi-stage, production-ready
├── requirements.txt
├── .env.example                # Secrets template
├── .dockerignore
├── .gitignore
├── README.md
└── app/
    ├── main.py                 # Streamlit entrypoint + UI
    ├── agent/
    │   ├── graph.py            # LangGraph definition (pure)
    │   ├── state.py            # TypedDict state schema
    │   └── tools.py            # Example tools (time, calculator)
    ├── db/
    │   └── checkpointer.py     # PostgresSaver + connection pool
    └── utils/
        └── config.py           # pydantic-settings configuration
```

---

## 🚀 Quick Start (Local)

### 1. Prerequisites
- Docker & Docker Compose v2
- An OpenAI API key (or any OpenAI-compatible endpoint)

### 2. Configure
```bash
cp .env.example .env
# Edit .env → set OPENAI_API_KEY and a strong POSTGRES_PASSWORD
```

### 3. Launch
```bash
docker compose up -d --build
```

### 4. Open
http://localhost:8501

---

## 🖥 Deploy to Your VPS

```bash
# On your local machine
scp -r langgraph-streamlit-memory user@your-vps-ip:~/

# On the VPS
cd ~/langgraph-streamlit-memory
cp .env.example .env
nano .env   # put real secrets + change POSTGRES_PASSWORD

docker compose up -d --build

# Optional: follow logs
docker compose logs -f app
```

### Recommended VPS hardening (after basic deploy)
1. Put a reverse proxy in front (Caddy or Nginx) for HTTPS + domain.
2. Open only ports 80/443 (and SSH). Do **not** expose 5432.
3. Use a strong `POSTGRES_PASSWORD` and consider rotating it.
4. Set up automatic backups of the `postgres_data` volume.
5. (Optional) Add `restart: always` and a simple systemd unit if you want extra safety.

---

## 🧠 How Memory Works

1. Every graph step is automatically checkpointed by LangGraph into Postgres.
2. The `thread_id` (UUID) is the primary key for a conversation.
3. When you send a new message, LangGraph loads the full history for that `thread_id` and continues.
4. Closing the browser or restarting the container does **not** lose anything.
5. In the sidebar you can copy the current `thread_id` or paste an old one to resume.

---

## 🛠 Extending the Project

### Add more tools
Edit `app/agent/tools.py` and re-export in the `TOOLS` list. The graph will automatically bind them.

### Change the LLM
Just change `OPENAI_MODEL` or point `OPENAI_API_BASE` to Groq, Together, Ollama, Azure, etc.

### Switch to AsyncPostgresSaver
If you later move to FastAPI + async endpoints, the pattern is almost identical (use `AsyncPostgresSaver` + async pool).

### Add long-term memory (beyond checkpoints)
Use `langgraph.store.postgres.PostgresStore` for semantic / cross-thread memory.

### Production scaling path
When traffic grows:
1. Extract the agent into a FastAPI service (see `JoshuaC215/agent-service-toolkit` for inspiration).
2. Keep Streamlit as a thin client.
3. Put Redis + multiple app replicas behind a load balancer.
4. Postgres remains the single source of truth for checkpoints.

---

## 🧪 Local Development (without Docker)

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start a local Postgres (or use Docker only for db)
docker run -d --name pg -e POSTGRES_PASSWORD=pass -e POSTGRES_USER=langgraph \
  -e POSTGRES_DB=langgraph_memory -p 5432:5432 postgres:16-alpine

# .env
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=pass
OPENAI_API_KEY=sk-...

streamlit run app/main.py
```

---

## 📋 Quality Checklist Applied

- ✅ Type hints everywhere
- ✅ Pydantic settings (no hardcoded secrets)
- ✅ Connection pool with correct `autocommit` + `dict_row`
- ✅ Idempotent `.setup()` for checkpoint tables
- ✅ Streaming responses
- ✅ Clean separation of agent / UI / DB
- ✅ Multi-stage non-root Docker image
- ✅ Healthchecks on both services
- ✅ Graceful thread resume UX
- ✅ Logging

---

## License

MIT – do whatever you want with it.
```
