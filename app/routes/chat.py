import uuid

from fastapi import APIRouter

from app.schemas import (
    ChatRequest,
    ChatResponse
)

from app.services.chatbot import generate_answer


router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post(
    "",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    session_id = request.session_id

    if not session_id:

        session_id = str(
            uuid.uuid4()
        )

    result = generate_answer(
        message=request.message,
        session_id=session_id,
        requested_language=request.language
    )

    return result