#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    article_text: str
    evidence_pair: Dict[Any, Dict]
    confidence: Optional[float] = None
    article_id: Optional[int] = None
    publish_date: Optional[str] = None
    evidence_pair_id: Optional[List[int]] = None
    key: Optional[str] = None  # stable id

@dataclass
class SIB_triplet:
    situation: str
    internal_state: str
    behavior: str
    article_text: str
    evidence_pair: Dict[Any, Dict]
    confidence: Optional[float] = None
    article_id: Optional[int] = None
    publish_date: Optional[str] = None
    evidence_pair_id: Optional[List[int]] = None
    key: Optional[str] = None  # stable id

@dataclass
class ReplyPair:
    article_id: int
    pair_id: int
    fan_text: str
    author_text: str
    article_snippet: Optional[str] = None
    publish_date: Optional[str] = None
    key: Optional[str] = None  # stable id


@dataclass
class SearchResult_persona:
    key: str
    claim: str
    score: float
    ref_article: str
    evidence_pair: Dict[int, Dict]
    meta: Dict[str, Any]

@dataclass
class SearchResult_SIB:
    key: str
    situation: str
    internal_state: str
    behavior: str
    score: float
    ref_article: str
    evidence_pair: Dict[int, Dict]
    meta: Dict[str, Any]

@dataclass
class SearchResult_pair:
    key: str
    fan_text: str
    author_text: str
    article: str
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
        article_id = raw.get("article_id")
        publish_date = raw.get("publish_date")
        for it in rec.get("items", []) or []:
            if module == "SIB":
                situation = (it.get("situation") or "").strip()
                internal_state = (it.get("internal_state") or "").strip()
                behavior = (it.get("behavior") or "").strip()
                if not (situation or internal_state or behavior):
                    continue
                key = _stable_key(module, situation, internal_state, behavior)
                pair_map = {}
                for i in range(len(it.get("evidence_pair_id"))):
                    pair_id = it.get("evidence_pair_id")[i]
                    for j, pair in enumerate(raw.get("pairs", [])):
                        if pair.get("pair_id") == pair_id:
                            pair_map[it.get("evidence_pair_id")[i]] = pair
                            break
                    if it.get("evidence_pair_id")[i] not in pair_map:
                        print(f"Warning: pair_id {it.get('evidence_pair_id')[i]} not found in article_id {article_id}")
                    

                items.append(
                    SIB_triplet(
                        situation=situation,
                        internal_state=internal_state,
                        behavior=behavior,
                        confidence=it.get("confidence"),
                        article_id=article_id,
                        publish_date=publish_date,
                        evidence_pair_id=it.get("evidence_pair_id"),
                        article_text=raw.get("article", ""),
                        evidence_pair=pair_map,
                        key=key,
                    )
                )
            else:
                claim = (it.get("claim") or "").strip()
                if not claim:
                    continue
                key = _stable_key(module, claim)
                pair_map = {}
                for i in range(len(it.get("evidence_pair_id"))):
                    pair_id = it.get("evidence_pair_id")[i]
                    for j, pair in enumerate(raw.get("pairs", [])):
                        if pair.get("pair_id") == pair_id:
                            pair_map[it.get("evidence_pair_id")[i]] = pair
                            break
                    if it.get("evidence_pair_id")[i] not in pair_map:
                        print(f"Warning: pair_id {pair_id} not found in article_id {article_id}")
                items.append(
                    PersonaClaim(
                        module=module,
                        claim=claim,
                        confidence=it.get("confidence"),
                        article_id=article_id,
                        publish_date=publish_date,
                        evidence_pair_id=it.get("evidence_pair_id"),
                        article_text=raw.get("article", ""),
                        evidence_pair=pair_map,
                        key=key,
                    )
                )
    return items


def load_pairs_from_json(target_name: str, raw_pairs_json: List[Dict[str, Any]], article_snippet_chars: int = -1) -> List[ReplyPair]:
    pairs: List[ReplyPair] = []
    for art in raw_pairs_json:
        article_id = art.get("article_id")
        publish_date = art.get("publish_date")
        article_text = (art.get("article") or "").strip()
        if article_snippet_chars <= 0:
            snippet = article_text # use full article
        else:
            snippet = (article_text[:article_snippet_chars] + "…")
        snippet = (article_text[:article_snippet_chars] + "…") if article_text else None
        for pr in art.get("pairs", []) or []:
            pair_id = pr.get("pair_id")
            fan = (pr.get("Fans") or "").strip()
            author = (pr.get(target_name) or "").strip()
            if not (fan or author):
                continue
            key = _stable_key("PAIR", article_id, pair_id)
            pairs.append(
                ReplyPair(
                    article_id=article_id,
                    pair_id=pair_id,
                    fan_text=fan,
                    author_text=author,
                    article_snippet=snippet,
                    publish_date=publish_date,
                    key=key,
                )
            )
    return pairs


# =========================
# Core memory store (in-memory)
# =========================
class MemoryStore:
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        batch_size: int = 256,
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

        # raw reply-pairs
        self._pair_texts: List[str] = []
        self._pair_meta: List[Dict[str, Any]] = []
        self._pair_vecs: np.ndarray = np.zeros((0, 1536), dtype=np.float32)

    # ---- embedding low-level ----
    def _embed_batch(self, texts: List[str], use_tqdm = True) -> np.ndarray:
        cached = [self.cache.get(t) if self.cache else None for t in texts]
        to_compute_idx = [i for i, v in enumerate(cached) if v is None]
        if to_compute_idx:
            inputs = [texts[i] for i in to_compute_idx]
            if use_tqdm:
                for start in tqdm(range(0, len(inputs), self.batch_size), desc="Embedding..."):
                    chunk = inputs[start:start + self.batch_size]
                    resp = self.client.embeddings.create(model=self.model_name, input=chunk)
                    vecs = [d.embedding for d in resp.data]
                    if self.cache:
                        for i, v in zip(to_compute_idx[start:start + self.batch_size], vecs):
                            self.cache.set(texts[i], v)
                    time.sleep(0.01)
            else:
                for start in range(0, len(inputs), self.batch_size):
                    chunk = inputs[start:start + self.batch_size]
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
        if claims[0].__class__ == SIB_triplet:
            texts = [c.situation for c in claims] # use situation as search text
            internal_states = [c.internal_state for c in claims]
            behaviors = [c.behavior for c in claims]
            metas = []
            for c in claims:
                meta = {
                    "key": c.key or _stable_key("SIB", c.situation, c.internal_state, c.behavior),
                    "situation": c.situation,
                    "internal_state": c.internal_state,
                    "behavior": c.behavior,
                    "confidence": c.confidence,
                    "article_id": c.article_id,
                    "publish_date": c.publish_date,
                    "evidence_pair_id": c.evidence_pair_id,
                    "article_text": c.article_text,
                    "evidence_pair": c.evidence_pair,
                }
                meta["data_source"] = "community"
                metas.append(meta)
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
            metas = []
            for c in claims:
                meta = {
                    "key": c.key or _stable_key(c.module, c.claim),
                    "module": c.module,
                    "confidence": c.confidence,
                    "article_id": c.article_id,
                    "publish_date": c.publish_date,
                    "evidence_pair_id": c.evidence_pair_id,
                    "article_text": c.article_text,
                    "evidence_pair": c.evidence_pair,
                    "claim": c.claim,
                }
                meta["data_source"] = "community"
                metas.append(meta)
            vecs = self._embed_batch(texts)
            if self._persona_vecs.size == 0:
                self._persona_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
            self._persona_texts.extend(texts)
            self._persona_meta.extend(metas)
            self._persona_vecs = np.vstack([self._persona_vecs, vecs])

    def retrieve_topk_persona_by_module(self, query: str, k_per_module: int = 3) -> Union[Dict[str, List[SearchResult_persona]], Dict[str, List[SearchResult_SIB]], Dict[str, List[SearchResult_pair]]]:
        if not self._persona_texts:
            return {}
        qv = self._embed_batch([query])
        sims = self._cosine_sim(qv, self._persona_vecs)[0]
        grouped: Dict[str, List[Tuple[int, float]]] = {}
        for idx, meta in enumerate(self._persona_meta):
            mod = meta.get("module", "Unknown")
            if mod == "Unknown":
                mod = "Pair"
            grouped.setdefault(mod, []).append((idx, float(sims[idx])))
        out = {}
        for mod, lst in grouped.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            top = lst[:k_per_module]
            if mod == "Pair":
                out[mod] = [
                    SearchResult_pair(
                        key=self._persona_meta[i]["key"],
                        fan_text=self._persona_meta[i]["fan_text"],
                        author_text=self._persona_meta[i]["author_text"],
                        article=self._persona_meta[i]["article_snippet"],
                        score=sc,
                        meta=self._persona_meta[i],
                    )
                    for i, sc in top
                ]
            else:
                out[mod] = [
                    SearchResult_persona(
                        key=self._persona_meta[i]["key"],
                        claim=self._persona_texts[i],
                        ref_article= self._persona_meta[i]["article_text"],
                        evidence_pair= self._persona_meta[i]["evidence_pair"],
                        score=sc,
                        meta=self._persona_meta[i],
                    )
                    for i, sc in top
                ]
        return out

    # ---- raw reply-pairs ----
    @staticmethod
    def _pair_search_text(p: ReplyPair) -> str:
        return p.fan_text

    def upsert_pairs(self, pairs: List[ReplyPair]) -> None:
        texts = [self._pair_search_text(p) for p in pairs]
        metas = [
            {
                "key": p.key or _stable_key("PAIR", p.article_id, p.pair_id),
                "article_id": p.article_id,
                "pair_id": p.pair_id,
                "publish_date": p.publish_date,
                "fan_text": p.fan_text,
                "author_text": p.author_text,
                "article_snippet": p.article_snippet,
                "raw": asdict(p),
            }
            for p in pairs
        ]
        vecs = self._embed_batch(texts)
        if self._pair_vecs.size == 0:
            self._pair_vecs = np.zeros((0, vecs.shape[1]), dtype=np.float32)
        self._pair_texts.extend(texts)
        self._pair_meta.extend(metas)
        self._pair_vecs = np.vstack([self._pair_vecs, vecs])


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

def build_persona_index_file(
    persona_json: List[Dict[str, Any]], 
    out_index_path: str, 
    model_name: str = "text-embedding-3-small", 
    cache_path: Optional[str] = ".emb_cache.json",
) -> None:
    store = MemoryStore(model_name=model_name, cache_path=cache_path)
    
    # Load community data
    claims = load_persona_claims_from_json(persona_json)
    
    
    store.upsert_persona_claims(claims)
    
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
    print(f"Saved merged index to: {out_index_path}")


def build_pairs_index_file(target_name: str, raw_pairs_json: List[Dict[str, Any]], out_index_path: str, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> None:
    store = MemoryStore(model_name=model_name, cache_path=cache_path)
    pairs = load_pairs_from_json(target_name, raw_pairs_json)
    store.upsert_pairs(pairs)
    idx = EmbeddingIndex(
        texts=store._pair_texts,
        metas=store._pair_meta,
        vecs=store._pair_vecs.astype(float).tolist(),
    )
    idx.save(out_index_path)


# =========================
# Retrieval over on-disk indices
# =========================

def retrieve_topk_per_index(query: str, index_path: str, k: int = 3, model_name: str = "text-embedding-3-small", cache_path: Optional[str] = ".emb_cache.json") -> Union[List[SearchResult_persona], List[SearchResult_pair], List[SearchResult_SIB]]:
    idx = EmbeddingIndex.load(index_path)
    texts, metas, mat = idx.texts, idx.metas, idx.to_matrix()
    if not texts:
        return []
    tmp = MemoryStore(model_name=model_name, cache_path=cache_path)
    qv = tmp._embed_batch([query], use_tqdm=False)
    sims = tmp._cosine_sim(qv, mat)[0]
    order = np.argsort(-sims)[:k]
    # print(order)
    if "pair" in os.path.basename(index_path):
        # print("有pair")
        result = [
            SearchResult_pair(
                key=metas[i].get("key", str(i)),
                fan_text=metas[i].get("fan_text", ""),
                author_text=metas[i].get("author_text", ""),
                article=metas[i].get("article_snippet", ""),
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
                ref_article= metas[i].get("article_text", ""),
                evidence_pair= metas[i].get("evidence_pair", ""),
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
                ref_article= metas[i].get("article_text", ""),
                evidence_pair= metas[i].get("evidence_pair", ""),
                score=float(sims[i]),
                meta=metas[i],
            ) for i in order
        ]
        # print(result)
        return result


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
    p_build.add_argument("--target_name", required=True)
    p_build.add_argument("--type", default="persona", choices=["persona", "pairs"])  # persona claims or raw pairs
    p_build.add_argument("--embedding_model", default=os.getenv("EMB_MODEL", "text-embedding-3-small"))
    p_build.add_argument("--extraction_model", required=True)

    args = parser.parse_args()

    if args.cmd == "build-index":
        print(args.input)
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        if args.type == "persona":
            build_persona_index_file(
                data, 
                args.out, 
                model_name=args.embedding_model, 
                cache_path=f"./data/{args.target_name}/persona_module/{args.extraction_model}/emb_cache.json",
            )
        else:
            build_pairs_index_file(args.target_name, data, args.out, model_name=args.embedding_model, cache_path=f"./data/{args.target_name}/persona_module/{args.extraction_model}/emb_cache.json")
        print(f"Saved index to {args.out}")