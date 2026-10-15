import logging
import os
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from agent import build_agent
from calculator import calculate_installment
from ingest import DATABASE_DIR, MODEL_NAME, COLLECTION_NAME
from rag_chat import BASE_DIR


logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    operation: str
    elapsed_seconds: float


class CalculationRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    principal: float = Field(gt=0, strict=True)
    annual_rate_percent: float = Field(ge=0, strict=True)
    months: int = Field(gt=0, strict=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources once when the server starts."""
    load_dotenv(BASE_DIR / ".env", encoding="utf-8-sig")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    database = chromadb.PersistentClient(path=str(DATABASE_DIR))
    collection = database.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )

    if collection.count() == 0:
        raise RuntimeError("The database is empty. Run ingest.py first.")

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
    client = genai.Client(api_key=api_key)

    app.state.agent = build_agent(client, embedding_model, collection)
    app.state.chunk_count = collection.count()

    # Limit this local demo to one active chat request at a time.
    app.state.chat_lock = Lock()

    logger.info("Policy assistant API ready.")
    yield


app = FastAPI(
    title="Loan & Policy Assistant",
    description=(
        "RAG and installment calculations using fictional demo policies. "
        "Each question is independent; conversation memory is not enabled."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": app.state.chunk_count,
        "note": "Startup succeeded. This does not check Gemini availability.",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    acquired = app.state.chat_lock.acquire(blocking=False)

    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="The assistant is busy. Try again shortly.",
        )

    started = perf_counter()

    try:
        result = app.state.agent.invoke(
            {"question": request.question}
        )

        elapsed = round(perf_counter() - started, 2)

        logger.info(
            "Chat completed: operation=%s elapsed_seconds=%.2f",
            result["plan"]["action"],
            elapsed,
        )

        return ChatResponse(
            answer=result["answer"],
            operation=result["plan"]["action"],
            elapsed_seconds=elapsed,
        )

    except Exception as error:
        # Log the error type, without exposing prompts or credentials.
        logger.error("Chat failed: %s", type(error).__name__)

        raise HTTPException(
            status_code=500,
            detail=(
                "The assistant could not complete the request. "
                "Check the server terminal and retry."
            ),
        ) from None

    finally:
        app.state.chat_lock.release()


@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        return calculate_installment(**request.model_dump())

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from None