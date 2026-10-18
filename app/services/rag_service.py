"""Retrieval-augmented generation: question -> relevant chunks -> Qwen -> answer.

The steps are kept separate and explicit on purpose:

    1. retrieve_documents()  question -> query embedding -> ChromaDB -> RETRIEVAL_TOP_K candidates
    2. rerank_documents()    reranker scores (question, chunk) pairs -> best RERANK_TOP_K
    3. build_context()       chunks -> one numbered text block for the prompt
    4. generate_answer()     LLMService: prompt(context, question) -> Qwen -> ONE answer
    5. build_sources()       chunk metadata -> source list for the frontend

ChromaDB only retrieves text. The reranker only reorders it. Qwen only writes
the answer from the chunks that survive.
"""
import logging
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.services.llm_service import NOT_AVAILABLE
from app.services.reranker_service import RerankerError

logger = logging.getLogger(__name__)

# "what is my mailid?", "my mail", "email address", "contact details" ... An
# address like name@gmail.com shares almost no meaning with such a question,
# so vector search alone ranks the chunk that holds it too low (or cuts it off).
_EMAIL_QUESTION = re.compile(r"\b(e-?mail|mail|gmail|contact)(\s*ids?)?\b", re.IGNORECASE)
_EMAIL_ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Extra chunks fetched for an email question, on top of the vector-search ones.
EMAIL_CHUNKS_K = 2


class NoRelevantDocumentsError(Exception):
    """Raised when retrieval finds no chunk relevant enough to answer from."""


class ChromaRetriever(BaseRetriever):
    """LangChain retriever over the Stage 1 ChromaDB collection.

    A retriever is LangChain's standard "text in, Documents out" interface.
    It embeds the query with the same EmbeddingService that embedded the chunks
    during /upload-pdf, so the query vector and the stored vectors are in the
    same embedding space and their distances are meaningful.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_service: Any
    chroma_service: Any
    k: int = 4
    max_distance: float | None = None
    document_id: str | None = None

    def generate_query_embedding(self, question: str) -> list[float]:
        return self.embedding_service.embed_query(question)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        query_vector = self.generate_query_embedding(query)
        results = self.chroma_service.query(query_vector, self.k, self.document_id)

        documents, seen = [], set()
        for document, distance in results:
            if self.max_distance is not None and distance > self.max_distance:
                continue
            document.metadata["distance"] = distance
            documents.append(document)
            seen.add(_chunk_key(document))

        if _EMAIL_QUESTION.search(query):
            # Chunks that actually contain an address, exempt from the distance
            # cutoff: the text match already proves they are relevant.
            matches = self.chroma_service.query(
                query_vector, EMAIL_CHUNKS_K, self.document_id, contains="@"
            )
            for document, distance in matches:
                if _chunk_key(document) in seen or not _EMAIL_ADDRESS.search(document.page_content):
                    continue
                document.metadata["distance"] = distance
                document.metadata["keyword_match"] = True
                documents.append(document)
                seen.add(_chunk_key(document))
        return documents


def _chunk_key(document: Document) -> tuple:
    return document.metadata.get("document_id"), document.metadata.get("chunk_index")


class RAGService:
    def __init__(
        self,
        embedding_service,
        chroma_service,
        llm_service,
        reranker=None,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 4,
        max_distance: float | None = None,
    ):
        if retrieval_top_k <= 0 or rerank_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K and RERANK_TOP_K must be positive")
        if rerank_top_k > retrieval_top_k:
            raise ValueError("RERANK_TOP_K cannot be larger than RETRIEVAL_TOP_K")
        self.embedding_service = embedding_service
        self.chroma_service = chroma_service
        self.llm_service = llm_service
        # None = reranking switched off: keep ChromaDB's order.
        self.reranker = reranker
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.max_distance = max_distance

    # --- Step 1: retrieval ------------------------------------------------------
    def retrieve_documents(self, question: str, document_id: str | None = None) -> list[Document]:
        """Fetch RETRIEVAL_TOP_K candidates, nearest first.

        More than we will finally use, so the reranker has something to choose from.
        """
        retriever = ChromaRetriever(
            embedding_service=self.embedding_service,
            chroma_service=self.chroma_service,
            k=self.retrieval_top_k,
            max_distance=self.max_distance,
            document_id=document_id,
        )
        documents = retriever.invoke(question)
        logger.info(
            "Retrieval complete k=%d document_id=%s results=%d distances=%s",
            self.retrieval_top_k, document_id, len(documents),
            [round(d.metadata["distance"], 3) for d in documents],
        )
        return documents

    # --- Step 2: reranking ------------------------------------------------------
    def rerank_documents(self, question: str, documents: list[Document]) -> list[Document]:
        """Keep the RERANK_TOP_K candidates the reranker scores as most relevant.

        Keyword-matched chunks (see ChromaRetriever) always stay: the reranker
        scores "mail id" against an address as poorly as vector search does.
        """
        pinned = [d for d in documents if d.metadata.get("keyword_match")][: self.rerank_top_k]
        documents = [d for d in documents if not d.metadata.get("keyword_match")]
        slots = self.rerank_top_k - len(pinned)
        if self.reranker is None or not documents or slots == 0:
            return pinned + documents[:slots]
        try:
            ranked = self.reranker.rerank(question, documents, slots)
        except RerankerError:
            # Still answer, just with ChromaDB's order, rather than fail /chat.
            logger.exception("Reranking failed; falling back to vector-search order")
            return pinned + documents[:slots]
        logger.info(
            "Reranking complete candidates=%d kept=%d scores=%s chunk_indexes=%s",
            len(documents), len(ranked),
            [round(d.metadata["rerank_score"], 3) for d in ranked],
            [d.metadata.get("chunk_index") for d in ranked],
        )
        return pinned + ranked

    # --- Step 3: context --------------------------------------------------------
    @staticmethod
    def build_context(documents: list[Document]) -> str:
        """Number each chunk so the model can cite it as [Source N]."""
        blocks = []
        for number, document in enumerate(documents, start=1):
            meta = document.metadata
            header = (
                f"[Source {number}] file: {meta.get('filename', 'unknown')}, "
                f"page: {meta.get('page_number', '?')}"
            )
            blocks.append(f"{header}\n{document.page_content.strip()}")
        return "\n\n---\n\n".join(blocks)

    # --- Step 5: sources --------------------------------------------------------
    @staticmethod
    def build_sources(documents: list[Document]) -> list[dict]:
        """Source list in the same order as [Source N] in the context.

        The metadata was attached to each chunk during /upload-pdf (PDFService
        adds document_id/filename/page_number, ChunkingService adds
        chunk_index) and stored in ChromaDB alongside the chunk text.
        """
        return [
            {
                "source_number": number,
                "document_id": document.metadata.get("document_id"),
                "filename": document.metadata.get("filename"),
                "page": document.metadata.get("page_number"),
                "chunk_index": document.metadata.get("chunk_index"),
                # 0..1 from the reranker; None if reranking was off or failed.
                "relevance_score": document.metadata.get("rerank_score"),
            }
            for number, document in enumerate(documents, start=1)
        ]

    # --- Full pipeline ----------------------------------------------------------
    def answer(self, question: str, document_id: str | None = None) -> dict:
        candidates = self.retrieve_documents(question, document_id)
        if not candidates:
            raise NoRelevantDocumentsError("No relevant information found in the uploaded documents")

        documents = self.rerank_documents(question, candidates)
        context = self.build_context(documents)
        # Step 4: generation. Qwen only sees the top-ranked text; it has no
        # access to ChromaDB itself.
        answer = self.llm_service.generate_answer(question, context)
        answer = self.check_grounding(question, answer, documents)

        return {"answer": answer, "sources": self.build_sources(documents)}

    @staticmethod
    def check_grounding(question: str, answer: str, documents: list[Document]) -> str:
        """Make sure every email address in the answer is copied exactly from the context.

        Small models ignore "do not invent information": they make up plausible
        addresses (e.g. test@gmail.com), or garble the real one while copying
        it (saranya.m.s@gmail.com). For an email question whose context holds
        an address, the answer is then rebuilt from the chunk text itself;
        otherwise it becomes NOT_AVAILABLE.
        """
        # address (lowercase) -> (address as written, [Source N] it came from)
        found = {}
        for number, document in enumerate(documents, start=1):
            for address in _EMAIL_ADDRESS.findall(document.page_content):
                found.setdefault(address.lower(), (address, number))

        in_answer = {a.lower() for a in _EMAIL_ADDRESS.findall(answer)}
        invented = in_answer - found.keys()
        if not invented and (in_answer or not _EMAIL_QUESTION.search(question)):
            return answer

        logger.warning(
            "Answer replaced: %d email address(es) not in the context, %d in the context",
            len(invented), len(found),
        )
        if not found or not _EMAIL_QUESTION.search(question):
            return NOT_AVAILABLE
        listed = ", ".join(f"{address} [Source {number}]" for address, number in found.values())
        if len(found) == 1:
            return f"The email address in the uploaded documents is {listed}."
        return f"The email addresses in the uploaded documents are: {listed}."
