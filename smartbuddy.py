"""
SmartBuddy — Build a Vector Database from Scratch in Python
=============================================================
A fully working Vector Database with HNSW, KD-Tree, and Brute Force
search, plus a RAG pipeline powered by a local LLM via Ollama.

This is a Python re-implementation of the original C++ project
(previously called "VectorDB"), renamed **SmartBuddy**. Same REST API,
same web UI (index.html) — just running on Flask + Python instead of a
compiled C++ binary + cpp-httplib.

Run:
    pip install -r requirements.txt
    python smartbuddy.py

Then open http://localhost:8080
"""

import requests
from flask import Flask, request, jsonify, Response, send_from_directory

from engine import (
    DIMS, VectorDB, DocumentDB, load_demo, get_dist_fn, chunk_text,
)

# =====================================================================
#  OLLAMA CLIENT — wraps local Ollama REST API
#  Install:  https://ollama.com
#  Models:   ollama pull nomic-embed-text
#            ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2:1b"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str):
        """Returns [] if Ollama is not running or the model isn't pulled."""
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  APP SETUP
# =====================================================================

app = Flask(__name__, static_folder=None)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


@app.after_request
def add_cors(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


def parse_vec(s: str):
    out = []
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


# ── DEMO VECTOR ENDPOINTS ────────────────────────────────────────────

@app.get("/search")
def route_search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    results = [
        {
            "id": h["id"],
            "metadata": h["meta"],
            "category": h["cat"],
            "distance": round(h["dist"], 6),
            "embedding": [round(x, 4) for x in h["emb"]],
        }
        for h in out["hits"]
    ]
    return jsonify({
        "results": results,
        "latencyUs": out["us"],
        "algo": out["algo"],
        "metric": out["metric"],
    })


@app.post("/insert")
def route_insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})
    new_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": new_id})


@app.delete("/delete/<int:item_id>")
def route_delete(item_id: int):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.get("/items")
def route_items():
    items = db.all()
    return jsonify([
        {
            "id": v.id,
            "metadata": v.metadata,
            "category": v.category,
            "embedding": [round(x, 4) for x in v.emb],
        }
        for v in items
    ])


@app.get("/benchmark")
def route_benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify({
        "bruteforceUs": b["bfUs"],
        "kdtreeUs": b["kdUs"],
        "hnswUs": b["hnswUs"],
        "itemCount": b["n"],
    })


@app.get("/hnsw-info")
def route_hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────────

@app.post("/doc/insert")
def route_doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.delete("/doc/delete/<int:item_id>")
def route_doc_delete(item_id: int):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.get("/doc/list")
def route_doc_list():
    docs = doc_db.all()
    out = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": d.text.count(" ") + 1,
        })
    return jsonify(out)


@app.post("/doc/search")
def route_doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {"id": item.id, "title": item.title, "distance": round(d, 4)}
            for d, item in hits
        ]
    })


@app.post("/doc/ask")
def route_doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx = ""
    for i, (_, item) in enumerate(hits):
        ctx += f"[{i+1}] {item.title}:\n{item.text}\n\n"

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": item.id, "title": item.title, "text": item.text, "distance": round(d, 4)}
            for d, item in hits
        ],
        "docCount": doc_db.size(),
    })


@app.get("/status")
def route_status():
    up = ollama.is_available()
    return jsonify({
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.get("/stats")
def route_stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# ── FRONTEND ──────────────────────────────────────────────────────────

@app.get("/")
def route_index():
    return send_from_directory(".", "index.html")


if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== SmartBuddy Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)
