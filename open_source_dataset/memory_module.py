
from __future__ import annotations

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import numpy as np
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

azure_api_key = os.environ['AZURE_OPENAI_API_KEY']
azure_api_version=os.environ['AZURE_API_VERSION']
azure_endpoint = os.environ['AZURE_ENDPOINT']
# =========================
# Data models
# =========================
@dataclass
class PersonaClaim:
    module: str  # "Memory" | "Speak_Style"
    claim: str
    dialogue: List[Dict]
    confidence: Optional[float] = None
    diag_id: Optional[int] = None
    key: Optional[str] = None  # stable id

@dataclass
class SIB_triplet:
    situation: str
    internal_state: str
    behavior: str
    dialogue: List[Dict]
    diag_id: Optional[int] = None
    confidence: Optional[float] = None
    key: Optional[str] = None  # stable id

@dataclass
class RawDialogue:
    diag_id: int
    dialogue: List[Dict]
    key: Optional[str] = None  # stable id


@dataclass
class SearchResult_persona:
    key: str
    claim: str
    score: float
    dialogue: List[Dict]
    meta: Dict[str, Any]

@dataclass
class SearchResult_SIB:
    key: str
    situation: str
    internal_state: str
    behavior: str
    score: float
    dialogue: List[Dict]
    meta: Dict[str, Any]

@dataclass
class SearchResult_Dialogue:
    key: str
    dialogue: List[Dict]
    score: float
    meta: Dict[str, Any]



def _stable_key(*parts: Any) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class _JsonCache:
    """Simple JSON cache mapping text -> embedding (list[float])."""
    def __init__(self, path: str = ".emb_cache.json"):
        self.path = path
        self._data: Dict[str, List[float]] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def get(self, text: str) -> Optional[List[float]]:
        return self._data.get(text)

    def set(self, text: str, vec: Iterable[float]) -> None:
        self._data[text] = list(vec)

    def flush(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception:
            pass


# =========================
# Loaders from your JSON schemas
# =========================

def load_persona_claims_from_json(persona_json: List[Dict[str, Any]]) -> List[PersonaClaim | SIB_triplet]:
    items: List[PersonaClaim | SIB_triplet] = []
    for block in persona_json:
        module = block.get("module_type")
        raw = block.get("raw_text", {})
        rec = block.get("recognized_result", {})
        diag_id = raw.get("diag_id")
        diag = raw.get("dialogue", [])
        for it in rec.get("items", []) or []:
            if module == "SIB":
                situation = (it.get("situation") or "").strip()
                internal_state = (it.get("internal_state") or "").strip()
                behavior = (it.get("behavior") or "").strip()
                if not (situation or internal_state or behavior):
                    continue
                key = _stable_key(module, situation, internal_state, behavior)
                    

                items.append(
                    SIB_triplet(
                        situation=situation,
                        internal_state=internal_state,
                        behavior=behavior,
                        confidence=it.get("confidence"),
                        diag_id=diag_id,
                        dialogue=diag,
                        key=key,
                    )
                )
            else:
                claim = (it.get("claim") or "").strip()
                if not claim:
                    continue
                key = _stable_key(module, claim)
                items.append(
                    PersonaClaim(
                        module=module,
                        claim=claim,
                        confidence=it.get("confidence"),
                        diag_id=diag_id,
                        dialogue=diag,
                        key=key,
                    )
                )
    return items


def load_diags_from_json(raw_diag_json: List[Dict]) -> List[RawDialogue]:
    diags: List[RawDialogue] = []
    for art in raw_diag_json:
        diag_id = art.get("diag_id")
        dialogue = art.get("dialogue", [])
        if len(dialogue) == 0:
            continue
        key = _stable_key(diag_id)
        diags.append(
            RawDialogue(
                diag_id=diag_id,
                dialogue=dialogue,
                key=key,
            )
        )
    return diags


# =========================
# Core memory store (in-memory)
# =========================
class MemoryStore:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 1,
        cache_path: Optional[str] = ".emb_cache.json",
    ):
        self.client = OpenAI(api_key=azure_api_key, base_url=azure_endpoint)
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache = _JsonCache(cache_path) if cache_path else None

        # persona claims
        self._persona_texts: List[str] = []
        self._persona_meta: List[Dict[str, Any]] = []
        self._persona_vecs: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

        # raw diags
        self._diag_texts: List[str] = []
        self._diag_meta: List[Dict[str, Any]] = []
        self._diag_vecs: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

    # ---- embedding low-level ----
    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        cached = [self.cache.get(t) if self.cache else None for t in texts]
        to_compute_idx = [i for i, v in enumerate(cached) if v is None]
        if to_compute_idx:
            inputs = [texts[i] for i in to_compute_idx]
            for start in range(0, len(inputs), self.batch_size):
                chunk = inputs[start:start + self.batch_size]
                # print(chunk)
                resp = self.client.embeddings.create(model=self.model_name, input=chunk)
                vecs = [d.embedding for d in resp.data]
                if self.cache:
                    for i, v in zip(to_compute_idx[start:start + self.batch_size], vecs):
                        self.cache.set(texts[i], v)
                time.sleep(0.01)
        out: List[List[float]] = []
        for t, c in zip(texts, cached):
            if c is None and self.cache:
                c = self.cache.get(t)
            if c is None:
                resp = self.client.embeddings.create(model=self.model_name, input=[t])
                c = resp.data[0].embedding
            out.append(c)
        if self.cache:
            self.cache.flush()
        return np.array(out, dtype=np.float32)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return a_norm @ b_norm.T

    # ---- persona claims ----
    def upsert_persona_claims(self, claims: List[PersonaClaim | SIB_triplet]) -> None:
        if len(claims) == 0:
            return
        if claims[0].__class__ == SIB_triplet:
            texts = [c.situation for c in claims] # use situation as search text
            internal_states = [c.internal_state for c in claims]
            behaviors = [c.behavior for c in claims]
            metas = [
                {
                    "key": c.key or _stable_key("SIB", c.situation, c.internal_state, c.behavior),
                    "situation": c.situation,
                    "internal_state": c.internal_state,
                    "behavior": c.behavior,
                    "confidence": c.confidence,
                    "dialogue": c.dialogue,
                    "diag_id": c.diag_id,
                }
                for c in claims
            ]
            vecs = self._embed_batch(texts)
            internal_vecs = self._embed_batch(internal_states)
            behavior_vecs = self._embed_batch(behaviors)
            if self._persona_vecs.size == 0:
                self._persona_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
            self._persona_texts.extend(texts)
            self._persona_meta.extend(metas)
            self._persona_vecs = np.vstack([self._persona_vecs, vecs])
            if not hasattr(self, '_internal_texts'):
                self._internal_texts = []
                self._internal_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
            if not hasattr(self, '_behavior_texts'):
                self._behavior_texts = []
                self._behavior_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
            self._internal_texts.extend(internal_states)
            self._internal_vecs = np.vstack([self._internal_vecs, internal_vecs])
            self._behavior_texts.extend(behaviors)
            self._behavior_vecs = np.vstack([self._behavior_vecs, behavior_vecs])
        else:
            texts = [c.claim for c in claims]
            metas = [
                {
                    "key": c.key or _stable_key(c.module, c.claim),
                    "module": c.module,
                    "confidence": c.confidence,
                    "dialogue": c.dialogue,
                    "diag_id": c.diag_id,
                    "claim": c.claim,
                }
                for c in claims
            ]
            vecs = self._embed_batch(texts)
            if self._persona_vecs.size == 0:
                self._persona_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
            self._persona_texts.extend(texts)
            self._persona_meta.extend(metas)
            self._persona_vecs = np.vstack([self._persona_vecs, vecs])

    # ---- raw reply-pairs ----
    @staticmethod
    def _pair_search_text(p: RawDialogue) -> str:
        long_text = ""
        for turn in p.dialogue:
            long_text += f"{turn.get('role','')}: {turn.get('content','')}\n"
        # print(long_text)
        if len(long_text) > 5000:
            long_text = long_text[:5000]
        # print(f"Dialogue text length: {len(long_text)}")
        return long_text

    def upsert_pairs(self, diags: List[RawDialogue]) -> None:
        texts = [self._pair_search_text(p) for p in diags]
        metas = [
            {
                "key": p.key,
                "diag_id": p.diag_id,
                "dialogue": p.dialogue,
                "raw": asdict(p),
            }
            for p in diags
        ]
        vecs = self._embed_batch(texts)
        if self._diag_vecs.size == 0:
            self._diag_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
        self._diag_texts.extend(texts)
        self._diag_meta.extend(metas)
        self._diag_vecs = np.vstack([self._diag_vecs, vecs])


# =========================
# On-disk index format (per-file)
# =========================
@dataclass
class EmbeddingIndex:
    texts: List[str] = field(default_factory=list)
    metas: List[Dict[str, Any]] = field(default_factory=list)
    vecs: List[List[float]] = field(default_factory=list)  # JSON-friendly

    # SIB only
    internal_texts: List[str] = field(default_factory=list)
    internal_vecs: List[List[float]] = field(default_factory=list)
    behavior_texts: List[str] = field(default_factory=list)
    behavior_vecs: List[List[float]] = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "texts": self.texts, 
            "metas": self.metas, 
            "vecs": self.vecs,
            "internal_texts": self.internal_texts,
            "internal_vecs": self.internal_vecs,
            "behavior_texts": self.behavior_texts,
            "behavior_vecs": self.behavior_vecs
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "EmbeddingIndex":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return EmbeddingIndex(
            texts=obj.get("texts", []),
            metas=obj.get("metas", []),
            vecs=obj.get("vecs", []),
            internal_texts=obj.get("internal_texts", []),
            internal_vecs=obj.get("internal_vecs", []),
            behavior_texts=obj.get("behavior_texts", []),
            behavior_vecs=obj.get("behavior_vecs", []),
        )

    def to_matrix(self) -> np.ndarray:
        if not self.vecs:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.array(self.vecs, dtype=np.float32)

    def to_internal_matrix(self) -> np.ndarray:
        if not self.internal_vecs:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.array(self.internal_vecs, dtype=np.float32)
    
    def to_behavior_matrix(self) -> np.ndarray:
        if not self.behavior_vecs:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.array(self.behavior_vecs, dtype=np.float32)

# =========================
# Builders (per-file indices)
# =========================

def build_persona_index_file(persona_json: List[Dict[str, Any]], out_index_path: str, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> None:
    store = MemoryStore(model_name=model_name, cache_path=cache_path)
    claims = load_persona_claims_from_json(persona_json)
    store.upsert_persona_claims(claims)
    # print(claims)
    if len(claims) == 0:
        print("no claims found!")
        idx = EmbeddingIndex()
        idx.save(out_index_path)
        return
    if claims and claims[0].__class__ == SIB_triplet:
        idx = EmbeddingIndex(
            texts=store._persona_texts,
            metas=store._persona_meta,
            vecs=store._persona_vecs.astype(float).tolist(),
            internal_texts=store._internal_texts,
            internal_vecs=store._internal_vecs.astype(float).tolist(),
            behavior_texts=store._behavior_texts,
            behavior_vecs=store._behavior_vecs.astype(float).tolist(),
        )
    else:
        idx = EmbeddingIndex(
            texts=store._persona_texts,
            metas=store._persona_meta,
            vecs=store._persona_vecs.astype(float).tolist(),
        )
    
    idx.save(out_index_path)


def build_dialogue_index_file(raw_diags_json: List[Dict[str, Any]], out_index_path: str, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> None:
    store = MemoryStore(model_name=model_name, cache_path=cache_path)
    pairs = load_diags_from_json(raw_diags_json)
    store.upsert_pairs(pairs)
    idx = EmbeddingIndex(
        texts=store._diag_texts,
        metas=store._diag_meta,
        vecs=store._diag_vecs.astype(float).tolist(),
    )
    idx.save(out_index_path)


# =========================
# Retrieval over on-disk indices
# =========================

def retrieve_topk_per_index(query: str, index_path: str, k: int = 3, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> Union[List[SearchResult_persona], List[SearchResult_Dialogue], List[SearchResult_SIB]]:
    idx = EmbeddingIndex.load(index_path)
    texts, metas, mat = idx.texts, idx.metas, idx.to_matrix()
    if not texts:
        return []
    tmp = MemoryStore(model_name=model_name, cache_path=cache_path)
    qv = tmp._embed_batch([query])
    sims = tmp._cosine_sim(qv, mat)[0]
    order = np.argsort(-sims)[:k]
    if "pair" in os.path.basename(index_path):
        result = [
            SearchResult_Dialogue(
                key=metas[i].get("key", str(i)),
                dialogue=metas[i].get("dialogue", ""),
                score=float(sims[i]),
                meta=metas[i],
            ) for i in order
        ]
        # print(result)
        return result
    elif "SIB" in os.path.basename(index_path):
        result = [
            SearchResult_SIB(
                key=metas[i].get("key", str(i)),
                situation=metas[i].get("situation", ""),
                internal_state=metas[i].get("internal_state", ""),
                behavior=metas[i].get("behavior", ""),
                dialogue=metas[i].get("dialogue", ""),
                score=float(sims[i]),
                meta=metas[i],
            ) for i in order
        ]
        # print("sib:", result)
        return result
    else:
        result = [
            SearchResult_persona(
                key=metas[i].get("key", str(i)),
                claim=texts[i],
                dialogue= metas[i].get("dialogue", ""),
                score=float(sims[i]),
                meta=metas[i],
            ) for i in order
        ]
        # print(result)
        return result


def retrieve_topk_across_indices(query: str, index_map: Dict[str, str], k_per_index: int = 3, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> Union[List[SearchResult_persona], List[SearchResult_Dialogue]]:
    tmp = MemoryStore(model_name=model_name, cache_path=cache_path)
    qv = tmp._embed_batch([query])
    out = {}
    for name, path in index_map.items():
        idx = EmbeddingIndex.load(path)
        texts, metas, mat = idx.texts, idx.metas, idx.to_matrix()
        if not texts:
            out[name] = []
            continue
        sims = tmp._cosine_sim(qv, mat)[0]
        order = np.argsort(-sims)[:k_per_index]
        if name == "Dialogue":
            out[name] = [
                SearchResult_Dialogue(
                    key=metas[i].get("key", str(i)),
                    dialogue=metas[i].get("dialogue", ""),
                    score=float(sims[i]),
                    meta=metas[i],
                ) for i in order
            ]
        elif name == "SIB":
            out[name] = [
                SearchResult_SIB(
                    key=metas[i].get("key", str(i)),
                    situation=metas[i].get("situation", ""),
                    internal_state=metas[i].get("internal_state", ""),
                    behavior=metas[i].get("behavior", ""),
                    dialogue= metas[i].get("dialogue", ""),
                    score=float(sims[i]),
                    meta=metas[i],
                ) for i in order
            ]
        else:
            out[name] = [
                SearchResult_persona(
                    key=metas[i].get("key", str(i)),
                    claim=texts[i],
                    dialogue= metas[i].get("dialogue", ""),
                    score=float(sims[i]),
                    meta=metas[i],
                ) for i in order
            ]
    return out


# =========================
# CLI entry
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build-index", help="Build an on-disk index JSON from a persona or pairs JSON file")
    p_build.add_argument("--input", required=True)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--person_name", required=True)
    p_build.add_argument("--dataset", required=True, choices=["RoleBench", "RoleAgentBench", "CharacterEval"])  # for future use
    p_build.add_argument("--type", default="persona", choices=["persona", "dialogue"])  # persona claims or raw pairs
    p_build.add_argument("--embedding_model", default=os.getenv("EMB_MODEL", "text-embedding-3-small"))
    p_build.add_argument("--extraction_model", required=True)

    p_query = sub.add_parser("query-many", help="Query multiple on-disk indices; each returns its own top-k")
    p_query.add_argument("--query", required=True)
    p_query.add_argument("--indexes", nargs="+", required=True, help="NAME:PATH entries, e.g. Memory:./memory.index.json")
    p_query.add_argument("--k", type=int, default=3)
    p_query.add_argument("--person_name", required=True)
    p_query.add_argument("--dataset", required=True, choices=["RoleBench", "RoleAgentBench"])
    p_query.add_argument("--embedding_model", default=os.getenv("EMB_MODEL", "text-embedding-3-small"))
    p_query.add_argument("--extraction_model", required=True)

    args = parser.parse_args()

    if args.cmd == "build-index":
        print(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.type == "persona":
            build_persona_index_file(data, args.out, model_name=args.embedding_model, cache_path=f"./data/{args.dataset}/character_data/{args.person_name}/persona_module/{args.extraction_model}/emb_cache.json")
        else:
            build_dialogue_index_file(data, args.out, model_name=args.embedding_model, cache_path=f"./data/{args.dataset}/character_data/{args.person_name}/persona_module/{args.extraction_model}/emb_cache.json")
        print(f"Saved index to {args.out}")

    elif args.cmd == "query-many":
        idx_map: Dict[str, str] = {}
        for ent in args.indexes:
            if ":" not in ent:
                raise SystemExit(f"Invalid index spec: {ent}. Use NAME:PATH")
            name, path = ent.split(":", 1)
            idx_map[name] = path
        res = retrieve_topk_across_indices(
            query=args.query,
            index_map=idx_map,
            k_per_index=args.k,
            model_name=args.embedding_model,
            cache_path=f"./data/{args.dataset}/character_data/{args.person_name}/persona_module/{args.extraction_model}/emb_cache.json",
        )
        for name, lst in res.items():
            print(f"[{name}] top-{args.k}")
            print(lst)
    else:
        # No subcommand: library mode only
        pass

