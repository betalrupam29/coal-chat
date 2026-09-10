from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    language: str = "auto"


class Source(BaseModel):
    document: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    language: str
    category: str
    session_id: str
    sources: list[Source]