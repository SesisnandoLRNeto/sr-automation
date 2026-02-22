"""Indexação semântica em ChromaDB com re-ranking via Cross-Encoder."""

import logging

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Embeddings + ChromaDB + Cross-Encoder re-ranking."""

    def __init__(self, config: dict):
        emb_cfg = config["embeddings"]
        vs_cfg = config["vector_store"]
        rr_cfg = config["reranking"]

        logger.info(f"Carregando embedder: {emb_cfg['model']}...")
        self.embedder = SentenceTransformer(
            emb_cfg["model"], device=emb_cfg.get("device", "cpu")
        )

        logger.info(f"Carregando re-ranker: {rr_cfg['model']}...")
        self.reranker = CrossEncoder(rr_cfg["model"])
        self.rerank_top_k = rr_cfg.get("top_k", 20)

        logger.info(f"Inicializando ChromaDB em {vs_cfg['persist_directory']}...")
        self.client = chromadb.PersistentClient(path=vs_cfg["persist_directory"])
        self.collection = self.client.get_or_create_collection(
            name=vs_cfg["collection_name"],
            metadata={"hnsw:space": "cosine"},
        )

    def index_corpus(self, corpus_df: pd.DataFrame) -> None:
        """Indexa title+abstract de cada artigo no ChromaDB."""
        existing_count = self.collection.count()
        if existing_count >= len(corpus_df):
            logger.info(
                f"Collection já contém {existing_count} docs (corpus: {len(corpus_df)}). "
                "Pulando indexação."
            )
            return

        logger.info(f"Indexando {len(corpus_df)} artigos...")
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Embeddings"):
            text = f"{row.get('title', '')} {row.get('abstract', '')}"
            emb = self.embedder.encode(text).tolist()

            ids.append(str(row["id"]))
            embeddings.append(emb)
            metadatas.append({
                "title": str(row.get("title", "")),
                "doi": str(row.get("doi", "")),
                "year": int(row.get("year", 0)) if row.get("year") else 0,
                "source": str(row.get("source", "")),
            })
            documents.append(text)

        # ChromaDB aceita batch upsert
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=documents[i:end],
            )

        logger.info(f"Indexação concluída. Total: {self.collection.count()} docs.")

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        """
        Busca semântica + re-ranking com Cross-Encoder.
        Retorna top rerank_top_k resultados.
        """
        query_emb = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return []

        candidates = []
        for i, doc_id in enumerate(results["ids"][0]):
            candidates.append({
                "id": doc_id,
                "title": results["metadatas"][0][i].get("title", ""),
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
            })

        # Re-ranking com Cross-Encoder
        pairs = [(query, c["document"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["score"] = float(score)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_results = candidates[: self.rerank_top_k]

        for rank, item in enumerate(top_results, 1):
            item["rank"] = rank

        return top_results
