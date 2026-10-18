"""ChromaDB persistence for embedded chunks."""
import logging

import chromadb
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ChromaStorageError(Exception):
    """Raised when chunks cannot be written to ChromaDB."""


class ChromaQueryError(Exception):
    """Raised when ChromaDB cannot be searched (e.g. storage unavailable)."""


class ChromaService:
    def __init__(self, persist_directory: str, collection_name: str, client=None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = client or chromadb.PersistentClient(path=persist_directory)
        # embedding_function=None: embeddings always come from EmbeddingService,
        # so Chroma never silently embeds with its own default model.
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def make_chunk_id(document_id: str, chunk_index: int) -> str:
        return f"{document_id}_{chunk_index}"

    def add_documents(
        self,
        chunks: list[Document],
        embeddings: list[list[float]],
        extra_metadata: dict | None = None,
    ) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ChromaStorageError("Chunk and embedding counts do not match")
        if not chunks:
            return []

        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            metadata = {**chunk.metadata, **(extra_metadata or {})}
            # Chroma metadata values must be str/int/float/bool (no None).
            metadata = {k: v for k, v in metadata.items() if v is not None}
            ids.append(self.make_chunk_id(metadata["document_id"], metadata["chunk_index"]))
            documents.append(chunk.page_content)
            metadatas.append(metadata)

        try:
            # add() (not upsert) so an ID collision fails instead of overwriting.
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        except Exception as exc:
            raise ChromaStorageError("Failed to store chunks in ChromaDB") from exc

        logger.info(
            "ChromaDB insert complete collection=%s records=%d total=%d",
            self.collection_name, len(ids), self.collection.count(),
        )
        return ids

    def query(
        self,
        query_embedding: list[float],
        k: int,
        document_id: str | None = None,
        contains: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return the k chunks nearest to query_embedding, as (Document, distance).

        distance is cosine distance (collection uses hnsw:space=cosine), so
        smaller = more similar. Results are ordered nearest first.
        `contains` limits the search to chunks whose text includes that string.
        """
        where = {"document_id": document_id} if document_id else None
        where_document = {"$contains": contains} if contains else None
        try:
            if self.collection.count() == 0:
                return []
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaQueryError("Failed to query ChromaDB") from exc

        # query() is batched: one result list per query embedding. We sent one.
        return [
            (Document(page_content=text, metadata=dict(metadata or {})), distance)
            for text, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return self.collection.count()
        return len(self.collection.get(where={"document_id": document_id}, include=[])["ids"])

    def get_document_chunks(self, document_id: str) -> dict:
        return self.collection.get(
            where={"document_id": document_id}, include=["metadatas", "documents"]
        )
