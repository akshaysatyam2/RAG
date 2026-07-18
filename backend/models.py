from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    status: str
    message: str


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    original_name: str
    file_type: str
    file_size: int
    page_count: int = 0
    chunk_count: int = 0
    status: str
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]
    total: int


class DeleteResponse(BaseModel):
    id: str
    deleted: bool
    message: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    history: list[dict] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)


class SourceChunk(BaseModel):
    document_id: str
    document_name: str
    chunk_index: int
    page_number: Optional[int] = None
    content: str
    relevance_score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_metadata: dict = Field(default_factory=dict)


class IngestionPhaseProgress(BaseModel):
    phase: str
    current_step: int
    total_steps: int
    message: str
    updated_at: str


class IngestionStatusResponse(BaseModel):
    document_id: str
    status: str
    phases: list[IngestionPhaseProgress]


class ErrorResponse(BaseModel):
    detail: str
