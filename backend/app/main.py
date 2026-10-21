import hashlib
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import engine
from app.models.document import Document
from app.models.user import User
from app.services.answer_service import generate_grounded_answer
from app.services.embedding_client import JinaEmbeddingClient
from app.services.ingestion_service import ingest_document
from app.services.llm_client import OpenAICompatibleClient, LLMProviderError
from app.services.query_service import search_question


app = FastAPI(title="ResearchVault API", version="0.1.0")


def get_db():
    with Session(engine) as db:
        yield db


class AskRequest(BaseModel):
    question: str
    limit: int = 5


@app.get("/health")
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from exc


@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    owner_id: uuid.UUID = Form(...),
    db: Session = Depends(get_db),
):
    """Development-only upload endpoint; authentication is not implemented yet."""
    if not settings.jina_api_key:
        raise HTTPException(503, "Jina API key is not configured")

    owner = db.get(User, owner_id)
    if owner is None:
        raise HTTPException(404, "Owner not found")

    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are supported")

    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "PDF exceeds the upload size limit")

    if not content.startswith(b"%PDF-"):
        raise HTTPException(415, "Uploaded file is not a valid PDF")

    document_id = uuid.uuid4()
    content_hash = hashlib.sha256(content).hexdigest()

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.upload_dir / f"{document_id}.pdf"
    file_path.write_bytes(content)

    document = Document(
        id=document_id,
        owner_id=owner_id,
        title=filename[:255] or "document.pdf",
        content_hash=content_hash,
        status="uploaded",
    )
    db.add(document)
    db.commit()

    embedding_client = JinaEmbeddingClient(api_key=settings.jina_api_key)
    try:
        result = await ingest_document(
            db=db,
            document_id=document_id,
            file_path=file_path,
            embedding_client=embedding_client,
        )
        return {
            **result,
            "title": document.title,
            "content_hash": content_hash,
        }
    except Exception as exc:
        raise HTTPException(
            502,
            "Document ingestion failed; check document status and server logs",
        ) from exc
    finally:
        await embedding_client.close()


@app.post("/documents/{document_id}/ask")
async def ask_document(
    document_id: uuid.UUID,
    request: AskRequest,
    db: Session = Depends(get_db),
):
    """Development-only Q&A endpoint; add authentication before deployment."""
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    if document.status != "ready":
        raise HTTPException(409, "Document is not ready for questions")

    if not settings.jina_api_key or not settings.llm_api_key:
        raise HTTPException(
            503,
            "Jina and Gemini API keys must be configured",
        )

    if not 1 <= request.limit <= 20:
        raise HTTPException(422, "limit must be between 1 and 20")

    embedding_client = JinaEmbeddingClient(api_key=settings.jina_api_key)
    llm_client = OpenAICompatibleClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )

    try:
        search_result = await search_question(
            db=db,
            question=request.question,
            embedding_client=embedding_client,
            limit=request.limit,
            document_id=document_id,
        )
        answer = await generate_grounded_answer(
            question=search_result["question"],
            passages=search_result["results"],
            llm_client=llm_client,
        )
        return {
            **answer,
            "retrieved_passage_count": len(search_result["results"]),
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(502, "Gemini answer generation failed") from exc
    finally:
        await embedding_client.close()
        await llm_client.close()
