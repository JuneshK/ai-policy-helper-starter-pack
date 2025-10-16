import os
import time
import json
import hashlib
import uuid
from typing import List, Dict, Tuple

import numpy as np
from pydantic import BaseModel

from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm


# ---------------- Local Embedder ---------------- #
class LocalEmbedder:
    """Deterministic pseudo-embedding using SHA1 hash and RNG."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha1(text.encode("utf-8")).digest()
        rng_seed = int.from_bytes(h[:8], "big") % (2**32 - 1)
        rng = np.random.default_rng(rng_seed)
        vec = rng.standard_normal(self.dim).astype("float32")
        return vec / (np.linalg.norm(vec) + 1e-9)


# ---------------- Vector Stores ---------------- #
class InMemoryStore:
    """Simple in-memory vector store."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)
        q = query.reshape(1, -1)
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]


class QdrantStore:
    """Qdrant-based vector store with fallback."""

    def __init__(self, collection: str, dim: int = 384):
        self.collection = collection
        self.dim = dim
        self.client = None
        try:
            self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
            self._ensure_collection()
        except Exception as e:
            print(f"[QdrantStore] Could not connect: {e}")
            self.client = None

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        if not self.client:
            print("[QdrantStore] Client not available. Skipping upsert.")
            return

        points = [
            qm.PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=m)
            for v, m in zip(vectors, metadatas)
        ]

        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except Exception as e:
            print(f"[QdrantStore] Upsert failed: {e}")

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.client:
            print("[QdrantStore] Client not available. Returning empty results.")
            return []

        res = self.client.search(
            collection_name=self.collection, query_vector=query.tolist(), limit=k, with_payload=True
        )
        return [(float(r.score), dict(r.payload)) for r in res]


# ---------------- LLM Providers ---------------- #
class StubLLM:
    """Simple stub LLM for testing."""

    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = ["Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append("Summary:")
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)


class OpenAILLM:
    """OpenAI LLM wrapper."""

    def __init__(self, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.1
        )
        return resp.choices[0].message.content


# ---------------- Metrics ---------------- #
class Metrics:
    def __init__(self):
        self.t_retrieval: List[float] = []
        self.t_generation: List[float] = []

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)

    def summary(self) -> Dict[str, float]:
        avg_r = sum(self.t_retrieval) / len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation) / len(self.t_generation) if self.t_generation else 0.0
        return {
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }


# ---------------- RAG Engine ---------------- #
class RAGEngine:
    def __init__(self):
        # Embeddings
        self.embedder = LocalEmbedder(dim=384)
        # Vector store
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # LLM
        if settings.llm_provider == "openai" and settings.openai_api_key:
            try:
                self.llm = OpenAILLM(api_key=settings.openai_api_key)
                self.llm_name = "openai:gpt-4o-mini"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        # Metrics and state
        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    # ---------------- Ingest ---------------- #
    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors, metas = [], []
        before_titles = set(self._doc_titles)

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            vectors.append(self.embedder.embed(text))
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return len(self._doc_titles) - len(before_titles), len(metas)

    # ---------------- Retrieve ---------------- #
    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k=k)
        self.metrics.add_retrieval((time.time() - t0) * 1000.0)
        return [meta for score, meta in results]

    # ---------------- Generate ---------------- #
    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time() - t0) * 1000.0)
        return answer

    # ---------------- Stats ---------------- #
    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": settings.embedding_model,
            "llm_model": self.llm_name,
            **m,
        }


# ---------------- Helpers ---------------- #
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"title": d["title"], "section": d.get("section"), "text": ch})
    return out
