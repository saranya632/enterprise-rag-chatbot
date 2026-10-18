import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from ollama import ResponseError

from app.services import ChromaQueryError, LLMService, RerankerError
from tests.conftest import authenticated_client, fake_reranker, make_test_app, upload
from tests.pdf_factory import make_pdf


class RecordingFakeLLM(FakeListChatModel):
    """Fake chat model (stands in for Qwen) that remembers the prompt it got."""

    last_prompt: list = []

    def _call(self, messages, *args, **kwargs):
        self.last_prompt = messages
        return super()._call(messages, *args, **kwargs)


class FailingLLM(FakeListChatModel):
    error: Exception

    def _call(self, *args, **kwargs):
        raise self.error


def make_app(tmp_path, chroma_service, embedding_service, llm, **config):
    llm_service = LLMService(model="fake", base_url="http://localhost:11434", llm=llm)
    config.setdefault(
        # Fake embeddings give arbitrary distances; the real cutoff from
        # .env must not apply here.
        "RETRIEVAL_MAX_DISTANCE", None,
    )
    return make_test_app(
        tmp_path,
        services={"chroma": chroma_service, "embedding": embedding_service, "llm": llm_service},
        **config,
    )


@pytest.fixture
def fake_llm():
    return RecordingFakeLLM(responses=["The procedure has three steps [Source 1]."])


@pytest.fixture
def chat_app(tmp_path, chroma_service, embedding_service, fake_llm):
    return make_app(tmp_path, chroma_service, embedding_service, fake_llm)


@pytest.fixture
def chat_client(chat_app):
    return authenticated_client(chat_app)


def ask(client, **body):
    return client.post("/chat", json=body)


# --- Success ------------------------------------------------------------------

def test_chat_returns_answer_and_sources(chat_client, sample_pdf_bytes):
    document_id = upload(chat_client, sample_pdf_bytes).get_json()["document_id"]

    response = ask(chat_client, question="What is the procedure?")

    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["question"] == "What is the procedure?"
    assert body["answer"] == "The procedure has three steps [Source 1]."
    assert len(body["sources"]) == 4  # RERANK_TOP_K default
    for number, source in enumerate(body["sources"], start=1):
        assert set(source) == {
            "source_number", "document_id", "filename", "page", "chunk_index", "relevance_score",
        }
        assert 0 <= source["relevance_score"] <= 1
        assert source["source_number"] == number
        assert source["document_id"] == document_id
        assert source["filename"] == "sample.pdf"
        assert 1 <= source["page"] <= 3


def test_prompt_contains_grounding_rules_context_and_question(chat_client, fake_llm, sample_pdf_bytes):
    upload(chat_client, sample_pdf_bytes)
    ask(chat_client, question="What is the procedure?")

    system, human = fake_llm.last_prompt
    assert "never from your own knowledge" in system.content
    assert "using only the context above" in human.content
    assert "not available in the uploaded documents" in human.content
    assert "[Source 1] file: sample.pdf, page:" in human.content
    # Rules come after the context, and the question is last, where small
    # models pay the most attention.
    assert human.content.index("Instructions:") > human.content.index("[Source 1]")
    assert human.content.rstrip().endswith("Question:\nWhat is the procedure?\n\nAnswer:")


def test_think_block_is_removed_from_answer(tmp_path, chroma_service, embedding_service, sample_pdf_bytes):
    llm = RecordingFakeLLM(responses=["<think>internal reasoning</think>\nThe answer [Source 1]."])
    client = authenticated_client(make_app(tmp_path, chroma_service, embedding_service, llm))
    upload(client, sample_pdf_bytes)

    assert ask(client, question="procedure?").get_json()["answer"] == "The answer [Source 1]."


def test_rerank_top_k_is_configurable(tmp_path, chroma_service, embedding_service, fake_llm, sample_pdf_bytes):
    client = authenticated_client(make_app(tmp_path, chroma_service, embedding_service, fake_llm, RERANK_TOP_K=2))
    upload(client, sample_pdf_bytes)

    assert len(ask(client, question="procedure?").get_json()["sources"]) == 2


def test_rerank_top_k_larger_than_retrieval_top_k_is_rejected(tmp_path, chroma_service, embedding_service, fake_llm):
    with pytest.raises(ValueError, match="RERANK_TOP_K"):
        make_app(tmp_path, chroma_service, embedding_service, fake_llm, RETRIEVAL_TOP_K=3, RERANK_TOP_K=4)


# --- Reranking ----------------------------------------------------------------

def test_only_top_reranked_chunks_reach_qwen(tmp_path, chroma_service, embedding_service, fake_llm, sample_pdf_bytes):
    # The fake reranker scores chunks by how often "Page 3" occurs, so every
    # page-3 chunk must beat every other chunk, whatever order ChromaDB used.
    client = authenticated_client(make_test_app(
        tmp_path,
        services={
            "chroma": chroma_service,
            "embedding": embedding_service,
            "llm": LLMService(model="fake", base_url="http://localhost:11434", llm=fake_llm),
            "reranker": fake_reranker("Page 3"),
        },
        RETRIEVAL_MAX_DISTANCE=None, RETRIEVAL_TOP_K=20, RERANK_TOP_K=2,
    ))
    upload(client, sample_pdf_bytes)

    sources = ask(client, question="What is on page 3?").get_json()["sources"]

    assert [s["page"] for s in sources] == [3, 3]
    scores = [s["relevance_score"] for s in sources]
    assert scores == sorted(scores, reverse=True)
    # Qwen got exactly the two kept chunks, nothing from pages 1-2.
    _, human = fake_llm.last_prompt
    assert "[Source 2]" in human.content and "[Source 3]" not in human.content
    assert "Page 1 line" not in human.content and "Page 2 line" not in human.content


def test_reranker_failure_falls_back_to_vector_order(chat_client, chat_app, monkeypatch, sample_pdf_bytes):
    def broken(*args, **kwargs):
        raise RerankerError("model file missing")

    upload(chat_client, sample_pdf_bytes)
    monkeypatch.setattr(chat_app.extensions["rag"].reranker, "score", broken)

    response = ask(chat_client, question="procedure?")

    assert response.status_code == 200
    sources = response.get_json()["sources"]
    assert len(sources) == 4 and all(s["relevance_score"] is None for s in sources)


def test_reranking_can_be_disabled(tmp_path, chroma_service, embedding_service, fake_llm, sample_pdf_bytes):
    app = make_test_app(
        tmp_path,
        services={
            "chroma": chroma_service,
            "embedding": embedding_service,
            "llm": LLMService(model="fake", base_url="http://localhost:11434", llm=fake_llm),
            "reranker": None,
        },
        RETRIEVAL_MAX_DISTANCE=None, RERANKER_ENABLED=False,
    )
    assert app.extensions["rag"].reranker is None
    client = authenticated_client(app)
    upload(client, sample_pdf_bytes)

    assert len(ask(client, question="procedure?").get_json()["sources"]) == 4


def test_document_id_filters_sources(chat_client, sample_pdf_bytes):
    first = upload(chat_client, sample_pdf_bytes).get_json()["document_id"]
    upload(chat_client, sample_pdf_bytes, filename="other.pdf")

    sources = ask(chat_client, question="procedure?", document_id=first).get_json()["sources"]

    assert sources and all(s["document_id"] == first for s in sources)


# --- Validation (400) ---------------------------------------------------------

@pytest.mark.parametrize(
    "body, error",
    [
        ({}, "Question is required"),
        ({"question": ""}, "Question is required"),
        ({"question": "   "}, "Question is required"),
        ({"question": 42}, "must be a string"),
        ({"question": "x" * 2001}, "longer than 2000"),
        ({"question": "ok?", "document_id": ""}, "Invalid document_id"),
    ],
)
def test_invalid_question_returns_400(chat_client, body, error):
    response = chat_client.post("/chat", json=body)
    assert response.status_code == 400
    assert error in response.get_json()["error"]


def test_invalid_json_returns_400(chat_client):
    response = chat_client.post("/chat", data="{not json", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must be a JSON object"


def test_non_json_body_returns_400(chat_client):
    response = chat_client.post("/chat", data="question=hi", content_type="text/plain")
    assert response.status_code == 400


# --- No relevant chunks (404) -------------------------------------------------

def test_empty_collection_returns_404(chat_client):
    response = ask(chat_client, question="What is the procedure?")
    assert response.status_code == 404
    assert response.get_json()["error"] == "No relevant information found in the uploaded documents"


def test_unknown_document_id_returns_404(chat_client, sample_pdf_bytes):
    upload(chat_client, sample_pdf_bytes)
    assert ask(chat_client, question="procedure?", document_id="missing").status_code == 404


def test_distance_cutoff_can_exclude_everything(tmp_path, chroma_service, embedding_service, fake_llm, sample_pdf_bytes):
    client = authenticated_client(make_app(
        tmp_path, chroma_service, embedding_service, fake_llm, RETRIEVAL_MAX_DISTANCE=-1.0
    ))
    upload(client, sample_pdf_bytes)

    assert ask(client, question="procedure?").status_code == 404


def _cv_pdf():
    """A long PDF whose one contact line is easy for vector search to miss."""
    filler = [f"Project {i}: tested solar inverter firmware and battery storage." for i in range(40)]
    return make_pdf([filler[:20] + ["CONTACT", "Email: jane.doe@example.org"] + filler[20:]])


def test_email_question_retrieves_chunk_with_address_despite_cutoff(
    tmp_path, chroma_service, embedding_service, fake_llm
):
    # A cutoff that excludes every chunk by distance: only the text match can find it.
    client = authenticated_client(make_app(
        tmp_path, chroma_service, embedding_service, fake_llm, RETRIEVAL_MAX_DISTANCE=-1.0
    ))
    upload(client, _cv_pdf(), filename="cv.pdf")

    assert ask(client, question="what is my mail id").status_code == 200
    _, human = fake_llm.last_prompt
    assert "jane.doe@example.org" in human.content


@pytest.mark.parametrize("llm_answer", [
    "Your email ID is test@gmail.com [Source 1].",       # invented
    "Your email ID is jane.d@example.org [Source 1].",   # garbled while copying
    "Your email ID is jane.doe @example.org",            # split by a space
    "The information is not available in the uploaded documents.",
])
def test_email_answer_is_rebuilt_from_context(tmp_path, chroma_service, embedding_service, llm_answer):
    llm = RecordingFakeLLM(responses=[llm_answer])
    client = authenticated_client(make_app(tmp_path, chroma_service, embedding_service, llm))
    upload(client, _cv_pdf(), filename="cv.pdf")

    answer = ask(client, question="what is my mailid").get_json()["answer"]
    assert answer.startswith("The email address in the uploaded documents is jane.doe@example.org [Source ")


def test_invented_email_without_one_in_context_is_rejected(chat_client, sample_pdf_bytes, fake_llm):
    fake_llm.responses = ["Write to test@gmail.com for the procedure."]
    upload(chat_client, sample_pdf_bytes)

    answer = ask(chat_client, question="What is the procedure?").get_json()["answer"]
    assert answer == "The information is not available in the uploaded documents."


def test_answer_with_email_from_context_is_kept(tmp_path, chroma_service, embedding_service):
    llm = RecordingFakeLLM(responses=["Your email ID is Jane.Doe@example.org [Source 1]."])
    client = authenticated_client(make_app(tmp_path, chroma_service, embedding_service, llm))
    upload(client, _cv_pdf(), filename="cv.pdf")

    body = ask(client, question="what is my email id?").get_json()
    assert body["answer"] == "Your email ID is Jane.Doe@example.org [Source 1]."


# --- Infrastructure errors ----------------------------------------------------

def test_chroma_unavailable_returns_503(chat_client, chat_app, monkeypatch):
    def broken_query(*args, **kwargs):
        raise ChromaQueryError("disk gone")

    monkeypatch.setattr(chat_app.extensions["rag"].chroma_service, "query", broken_query)

    response = ask(chat_client, question="procedure?")
    assert response.status_code == 503
    assert "disk gone" not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    "error, status",
    [
        (ConnectionError("Failed to connect to Ollama"), 503),           # Ollama not running
        (ResponseError("model 'qwen3:8b' not found", 404), 503),         # model not pulled
        (httpx.ReadTimeout("timed out"), 504),                           # generation too slow
        (ResponseError("model runner crashed", 500), 500),              # generation error
    ],
)
def test_ollama_errors_map_to_status_without_leaking(tmp_path, chroma_service, embedding_service, sample_pdf_bytes, error, status):
    client = authenticated_client(make_app(
        tmp_path, chroma_service, embedding_service, FailingLLM(responses=[""], error=error)
    ))
    upload(client, sample_pdf_bytes)

    response = ask(client, question="procedure?")

    assert response.status_code == status
    text = response.get_data(as_text=True)
    assert "qwen3" not in text and "crashed" not in text and "Traceback" not in text


def test_unexpected_error_returns_500(chat_client, chat_app, monkeypatch, sample_pdf_bytes):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    upload(chat_client, sample_pdf_bytes)
    monkeypatch.setattr(chat_app.extensions["rag"], "build_context", boom)

    response = ask(chat_client, question="procedure?")
    assert response.status_code == 500
    assert "secret internal detail" not in response.get_data(as_text=True)
