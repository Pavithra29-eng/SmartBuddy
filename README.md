# SmartBuddy — Build a Vector Database from Scratch in Python

A fully working Vector Database built from scratch in **Python** with a web UI.
Implements HNSW, KD-Tree, and Brute Force search algorithms side-by-side, plus a
RAG pipeline powered by a local LLM via Ollama.

> This is the Python rewrite of the original C++ project (formerly "VectorDB"),
> renamed **SmartBuddy**. Same algorithms, same REST API, same web UI — now
> running on Flask instead of a compiled C++ binary, so there's no compiler
> step at all.

Built as an educational project to show how production vector databases like
Pinecone, Weaviate, and Chroma actually work under the hood.

## What This Project Does

| Feature | Description |
|---|---|
| 3 Search Algorithms | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| 3 Distance Metrics | Cosine similarity, Euclidean distance, Manhattan distance |
| 16D Demo Vectors | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| 2D PCA Scatter Plot | Live visualization of semantic space — watch clusters form |
| Real Document Embedding | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| RAG Pipeline | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| Full REST API | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

HNSW (Hierarchical Navigable Small World) is the same algorithm used by
Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each
layer is progressively sparser — searches start at the top layer and zoom in,
achieving O(log N) complexity instead of O(N) for brute force.

## Prerequisites

You need 2 things installed:

1. **Python 3.9+** (no compiler needed — that's the whole point of the rewrite)
2. **Ollama** (runs the local AI models)

## Step-by-Step Setup

### Step 1 — Install Python

**Windows:** Go to https://python.org/downloads, download the installer, and
make sure you check **"Add Python to PATH"** during install.

**Mac:** Python 3 usually comes preinstalled. If not: `brew install python3`

**Linux:** `sudo apt install python3 python3-pip` (Debian/Ubuntu) or your
distro's equivalent.

Verify in a terminal / PowerShell:
```bash
python3 --version
```
(On Windows this may just be `python --version`.)

### Step 2 — Install Ollama (Local AI Models)

1. Go to https://ollama.com and download the installer for your OS
2. Run the installer — Ollama starts automatically in the background
3. Open a terminal and pull the two required models:
```bash
ollama pull nomic-embed-text
```
(~274 MB — this is the embedding model)
```bash
ollama pull llama3.2
```
(~2 GB — this is the language model)

4. Verify Ollama is running:
```bash
ollama list
```
You should see both models listed.

> Minimum specs for Ollama: 8GB RAM recommended. The models will use ~3GB total.

### Step 3 — Get the Project

Either clone it:
```bash
git clone https://github.com/YOUR_USERNAME/SmartBuddy.git
cd SmartBuddy
```
or just unzip the SmartBuddy folder you downloaded and `cd` into it.

### Step 4 — Install Python Dependencies

Inside the `SmartBuddy` folder:
```bash
pip install -r requirements.txt
```
(If `pip` isn't found, try `pip3` or `python3 -m pip install -r requirements.txt`.)

This installs `Flask` (the web server) and `requests` (used to talk to Ollama).
Everything else — HNSW, KD-Tree, Brute Force, the RAG pipeline — is plain
Python with no extra dependencies.

### Step 5 — Run Everything

**Terminal 1 — Start Ollama** (if not already running):
```bash
ollama serve
```
(If Ollama is already running in the background/tray, skip this.)

**Terminal 2 — Start the SmartBuddy server:**
```bash
python3 smartbuddy.py
```
You should see:
```
=== SmartBuddy Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open your browser and go to:
```
http://localhost:8080
```

## Using the Application

### Tab 1: Search (Demo Vectors)
- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: HNSW, KD-Tree, or Brute Force
- Choose distance metric: Cosine, Euclidean, or Manhattan
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

The scatter plot shows all 20 vectors projected to 2D using PCA. Notice how the
4 semantic categories (CS, Math, Food, Sports) form distinct clusters — this is
what "semantic similarity" looks like visually.

### Tab 2: Documents (Real Embeddings)
This uses Ollama to generate real 768-dimensional embeddings from any text.
1. Type a title (e.g., "Operating Systems Notes")
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)
1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question   → embedded with nomic-embed-text (768D vector)
2. HNSW search      → finds 3 most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2          → generates an answer based only on your documents
```

The answer streams in with a typewriter effect. Click the context chips to see
exactly which chunks the AI used.

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| POST | `/insert` | Insert a demo vector |
| DELETE | `/delete/:id` | Delete by ID |
| GET | `/items` | List all demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| GET | `/hnsw-info` | HNSW graph structure and layer stats |
| GET | `/stats` | Database statistics |

### Document & RAG Endpoints
| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| GET | `/doc/list` | — | List all stored documents |
| DELETE | `/doc/delete/:id` | — | Delete document chunk |
| POST | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| GET | `/status` | — | Ollama status and model info |

### Example: Search via curl
```bash
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl
```bash
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is dynamic programming?","k":3}'
```

## Project Structure

```
SmartBuddy/
├── smartbuddy.py     ← Flask server (REST API, Ollama client, routes)
├── engine.py          ← Core engine (HNSW, KD-Tree, BruteForce, VectorDB, DocumentDB)
├── requirements.txt   ← Python dependencies (Flask, requests)
├── index.html         ← Frontend (PCA scatter plot, chat UI, benchmark) — unchanged
└── README.md          ← This file
```

## Architecture

```
BruteForce          O(N·d)      Exact, baseline
KDTree               O(log N)    Exact, axis-aligned partitioning
HNSW                 O(log N)    Approximate, multilayer small-world graph

VectorDB             Unified interface over all 3 (16D demo vectors)
DocumentDB            HNSW-only index for real Ollama embeddings (768D)
OllamaClient          HTTP client → /api/embeddings + /api/generate
```

## Algorithm Deep Dive

### HNSW (Hierarchical Navigable Small World)
Nodes are inserted into a multilayer graph. Each node randomly gets assigned a
maximum layer. Layer 0 has all nodes with many connections; higher layers have
fewer nodes (exponentially fewer) with longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop a
layer, repeat. At each layer from your assigned max down to 0, run a beam
search (`ef_construction=200`) and connect to the M nearest neighbors
bidirectionally.

**Search:** Same greedy descent from top layer. At layer 0, expand to `ef`
nearest candidates using a priority queue.

**Why it's fast:** The upper layers act like a highway — you quickly get to
the right neighborhood, then zoom in at layer 0.

### KD-Tree (K-Dimensional Tree)
Binary space partitioning. Each node splits space along one dimension
(cycling through all dimensions). Search prunes entire subtrees when the
closest possible point in that subtree can't beat the current best.

**Weakness:** Degrades with high dimensions (curse of dimensionality). Works
well for ≤20D, becomes close to brute force at 768D.

### Why HNSW Wins at High Dimensions
KD-Tree pruning relies on axis-aligned distance bounds. In high dimensions,
almost all the space is near the boundary of the hypersphere — no subtrees get
pruned. HNSW's graph-based approach doesn't have this problem.

## Common Issues

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal |
| Embedding takes forever | Ollama is downloading the model on first use, wait 2 min |
| `python3: command not found` | Reinstall Python and check "Add to PATH" (Windows) |
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` again |
| Port 8080 already in use | **Windows:** `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F`. **Mac/Linux:** `lsof -i :8080` then `kill <pid>` |
| LLM answer is slow | Normal — llama3.2 takes 10–30s on a laptop CPU. Use `llama3.2:1b` for faster answers |

## Use a Smaller/Faster LLM

If `llama3.2` is too slow on your machine, switch to the 1B model:
```bash
ollama pull llama3.2:1b
```
Then edit `smartbuddy.py`, in the `OllamaClient.__init__` method:
```python
self.gen_model = "llama3.2:1b"   # change this
```
Restart the server (`python3 smartbuddy.py`) — no compile step needed.

## License

MIT — use this however you want.
