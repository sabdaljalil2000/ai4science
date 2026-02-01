# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Iterable, Optional
# import os
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# @dataclass
# class RetrievedSentence:
#     doc_id: str
#     sent_id: int
#     text: str
#     score: float

# class FaissSentenceRetriever:
#     """A simple FAISS index over evidence sentences (for SciFact-Open style retrieval).

#     You provide:
#       - sentences: list of (doc_id, sent_id, text)
#     We build:
#       - embeddings via SentenceTransformer
#       - FAISS index for cosine similarity (inner product with normalized vecs)
#     """
#     def __init__(self, embed_model: str, index_path: str, device: str = "cpu"):
#         self.embed_model_name = embed_model
#         self.index_path = index_path
#         self.device = device
#         self.model = SentenceTransformer(embed_model, device=(device if device != "cuda" else "cuda"))
#         self.index: Optional[faiss.Index] = None
#         self.meta: Optional[List[Tuple[str,int,str]]] = None

#     # def build(self, sentences: List[Tuple[str,int,str]], batch_size: int = 256) -> None:
#     #     self.meta = sentences
#     #     texts = [t for (_,_,t) in sentences]
#     #     embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
#     #     embs = np.asarray(embs, dtype="float32")
#     #     dim = embs.shape[1]
#     #     index = faiss.IndexFlatIP(dim)
#     #     index.add(embs)
#     #     self.index = index
#     #     os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
#     #     faiss.write_index(index, self.index_path)
#     #     np.save(self.index_path + ".meta.npy", np.array(self.meta, dtype=object), allow_pickle=True)

#     def build(self, sentences):
#         """
#         Build FAISS over sentence embeddings.

#         Accepts either:
#         A) iterable of (doc_id, sent_id, text) tuples
#         B) iterable of EvidenceSentence-like objects with attributes: doc_id, sent_id, text
#         """
#         import numpy as np
#         import faiss

#         # Normalize inputs to tuples
#         norm = []
#         for x in sentences:
#             # Case A: tuple/list
#             if isinstance(x, (tuple, list)) and len(x) >= 3:
#                 doc_id, sent_id, text = x[0], x[1], x[2]
#                 norm.append((str(doc_id), int(sent_id), str(text)))
#                 continue

#             # Case B: EvidenceSentence-like object
#             if hasattr(x, "doc_id") and hasattr(x, "sent_id") and hasattr(x, "text"):
#                 norm.append((str(x.doc_id), int(x.sent_id), str(x.text)))
#                 continue

#             raise TypeError(f"Unsupported sentence record type: {type(x)}")

#         self.meta = [{"doc_id": d, "sent_id": i} for (d, i, _) in norm]
#         texts = [t for (_, _, t) in norm]

#         # Embed
#         embs = self._embed(texts)  # should return np.ndarray [N, D], float32

#         if not isinstance(embs, np.ndarray):
#             embs = np.array(embs)
#         embs = embs.astype("float32")

#         # Build FAISS (inner product if normalized, else L2 — keep whatever your _embed implies)
#         d = embs.shape[1]
#         index = faiss.IndexFlatIP(d)
#         index.add(embs)

#         self.index = index
#         self.save()

#     def load(self) -> None:
#         self.index = faiss.read_index(self.index_path)
#         self.meta = list(np.load(self.index_path + ".meta.npy", allow_pickle=True))

#     def search(self, query: str, top_k: int = 20) -> List[RetrievedSentence]:
#         assert self.index is not None and self.meta is not None, "Index not built/loaded"
#         q = self.model.encode([query], normalize_embeddings=True)
#         q = np.asarray(q, dtype="float32")
#         scores, idxs = self.index.search(q, top_k)
#         out: List[RetrievedSentence] = []
#         for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
#             if idx < 0: 
#                 continue
#             doc_id, sent_id, text = self.meta[idx]
#             out.append(RetrievedSentence(doc_id=str(doc_id), sent_id=int(sent_id), text=str(text), score=float(score)))
#         return out
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union, Any, Dict
import os
import json

import numpy as np


@dataclass
class RetrievedSentence:
    doc_id: str
    sent_id: int
    text: str
    score: float


class FaissSentenceRetriever:
    """
    FAISS sentence retriever for SciFact-Open.

    It builds an index over sentence embeddings.
    Sentences can be provided as either:
      - tuples: (doc_id, sent_id, text)
      - objects with attributes: doc_id, sent_id, text (e.g., EvidenceSentence)
    """

    def __init__(self, embed_model: str, index_path: str, device: str = "cpu"):
        self.embed_model = embed_model
        self.index_path = index_path
        self.device = device

        self.index = None  # faiss.Index
        self.meta: List[Dict[str, Any]] = []     # aligned with FAISS vectors: {"doc_id":..., "sent_id":...}
        self.texts: List[str] = []               # aligned with meta

        self._encoder = None  # lazy-loaded SentenceTransformer

    # --------------------------
    # Embeddings
    # --------------------------
    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model, device=self.device)
        return self._encoder

    def _embed(self, texts: List[str]) -> np.ndarray:
        enc = self._get_encoder()
        # normalize_embeddings=True makes inner product = cosine similarity
        embs = enc.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        return embs.astype("float32")

    # --------------------------
    # Build / Save / Load
    # --------------------------
    def build(self, sentences: Iterable[Any]):
        """
        Build FAISS index from sentence records.
        """
        import faiss

        norm: List[Tuple[str, int, str]] = []
        for x in sentences:
            # Case A: tuple/list
            if isinstance(x, (tuple, list)) and len(x) >= 3:
                doc_id, sent_id, text = x[0], x[1], x[2]
                norm.append((str(doc_id), int(sent_id), str(text)))
                continue

            # Case B: EvidenceSentence-like object
            if hasattr(x, "doc_id") and hasattr(x, "sent_id") and hasattr(x, "text"):
                norm.append((str(x.doc_id), int(x.sent_id), str(x.text)))
                continue

            raise TypeError(f"Unsupported sentence record type: {type(x)}")

        self.meta = [{"doc_id": d, "sent_id": i} for (d, i, _) in norm]
        self.texts = [t for (_, _, t) in norm]

        embs = self._embed(self.texts)  # [N, D]
        d = int(embs.shape[1])

        # Cosine similarity with normalized embeddings => use inner product
        index = faiss.IndexFlatIP(d)
        index.add(embs)

        self.index = index
        self.save()

    def save(self):
        """
        Saves:
          - FAISS index to <index_path>
          - metadata to <index_path>.meta.jsonl
          - texts to <index_path>.texts.jsonl
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        import faiss

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        meta_path = self.index_path + ".meta.jsonl"
        txt_path = self.index_path + ".texts.jsonl"

        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m) + "\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            for t in self.texts:
                f.write(json.dumps({"text": t}) + "\n")

    def load(self):
        """
        Loads index + metadata + texts.
        """
        import faiss

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(self.index_path)

        meta_path = self.index_path + ".meta.jsonl"
        txt_path = self.index_path + ".texts.jsonl"

        if not os.path.exists(meta_path):
            raise FileNotFoundError(meta_path)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(txt_path)

        self.index = faiss.read_index(self.index_path)

        self.meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))

        self.texts = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.texts.append(json.loads(line)["text"])

        if len(self.meta) != len(self.texts):
            raise RuntimeError(f"meta/text length mismatch: {len(self.meta)} vs {len(self.texts)}")

    # --------------------------
    # Search
    # --------------------------
    def search(self, query: str, top_k: int = 20) -> List[RetrievedSentence]:
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        q_emb = self._embed([query])  # [1, D]
        scores, idxs = self.index.search(q_emb, top_k)

        out: List[RetrievedSentence] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            m = self.meta[i]
            out.append(
                RetrievedSentence(
                    doc_id=str(m["doc_id"]),
                    sent_id=int(m["sent_id"]),
                    text=self.texts[i],
                    score=float(score),
                )
            )
        return out
