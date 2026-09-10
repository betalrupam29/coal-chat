from groq import Groq

from app.config import settings
from app.services.rag import get_rag
from app.services.memory import conversation_memory
from app.services.safety import is_emergency
from app.utils.language import (
    detect_language,
    language_name
)


client = Groq(
    api_key=settings.GROQ_API_KEY
)


MODEL_NAME = "openai/gpt-oss-120b"


SYSTEM_PROMPT = """
You are a Coal Mine Worker Assistant.

Your job is to help coal mine workers understand:
- company policies
- safety procedures
- benefits
- workplace rules
- general information

IMPORTANT RULES:

1. Company-specific policy information must come from the
   supplied company policy documents.

2. Never invent company-specific policies.

3. If the company policy documents do not contain relevant
   information, do not refuse automatically.

4. If policy information is unavailable, answer using
   general knowledge when appropriate.

5. When answering using general knowledge, clearly state:
   "This is general information, not a company-specific policy."

6. Never present general knowledge as company policy.

7. Explain policies in very simple language.

8. Use short sentences.

9. Prefer bullet points when explaining procedures.

10. Do not use complicated legal or technical language
    unless necessary.

11. Always respond in the requested language.

12. Supported languages:
    English
    Hindi
    Bengali

13. If the worker asks about Indian law, regulations,
    or legal rights, clearly state that the information
    is general information and should be verified with
    the appropriate official authority, HR, or legal
    department.

14. For emergency situations, prioritize immediate safety.

15. Never invent mine-specific:
    - emergency phone numbers
    - evacuation routes
    - assembly points
    - emergency procedures

16. If emergency information is not available in the
    supplied company documents, tell the worker to follow
    the official mine emergency procedure and contact
    the designated mine emergency authority.

17. Never claim that you contacted emergency services.

18. Do not request unnecessary personal information.

19. Do not identify or track individual workers.

20. The assistant is common for all workers.
"""


def classify_question(message: str) -> str:

    if is_emergency(message):
        return "emergency"

    policy_words = [

        "policy",
        "rule",
        "salary",
        "leave",
        "bonus",
        "ppe",
        "helmet",
        "safety",
        "attendance",
        "benefit",
        "insurance",
        "shift",
        "working hours",
        "overtime",
        "medical",
        "training",
        "holiday",

        # Hindi
        "नीति",
        "नियम",
        "वेतन",
        "छुट्टी",
        "सुरक्षा",
        "हेलमेट",

        # Bengali
        "নীতি",
        "নিয়ম",
        "বেতন",
        "ছুটি",
        "নিরাপত্তা",
        "হেলমেট"
    ]

    message_lower = message.lower()

    for word in policy_words:

        if word.lower() in message_lower:
            return "policy"

    return "general"


def build_context(results):

    if not results:
        return "No relevant company policy information was found."

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
DOCUMENT: {result['document']}
PAGE: {result['page']}

CONTENT:
{result['text']}
"""
        )

    return "\n".join(context_parts)


def build_history(session_id):

    history = conversation_memory.get_history(
        session_id
    )

    if not history:
        return ""

    recent_history = history[-10:]

    history_text = []

    for item in recent_history:

        history_text.append(
            f"{item['role'].upper()}: {item['content']}"
        )

    return "\n".join(history_text)


def generate_answer(
    message: str,
    session_id: str,
    requested_language: str = "auto"
):

    # --------------------------------
    # Language detection
    # --------------------------------

    if requested_language == "auto":

        language_code = detect_language(
            message
        )

    else:

        language_code = requested_language

    language = language_name(
        language_code
    )

    # --------------------------------
    # Question classification
    # --------------------------------

    category = classify_question(
        message
    )

    # --------------------------------
    # Load RAG only when needed
    # --------------------------------

    rag = get_rag()

    search_results = rag.search(
        message,
        top_k=4
    )

    # --------------------------------
    # RAG similarity threshold
    # --------------------------------

    MIN_SCORE = 0.45

    relevant_results = [

        result

        for result in search_results

        if result["score"] >= MIN_SCORE

    ]

    # --------------------------------
    # Build context
    # --------------------------------

    context = build_context(
        relevant_results
    )

    # --------------------------------
    # Conversation memory
    # --------------------------------

    history = build_history(
        session_id
    )

    # --------------------------------
    # Decide answer mode
    # --------------------------------

    if relevant_results:

        answer_mode = "COMPANY_POLICY"

    else:

        answer_mode = "GENERAL_KNOWLEDGE"

    # --------------------------------
    # Prompt
    # --------------------------------

    prompt = f"""
RESPONSE LANGUAGE:
{language}

QUESTION CATEGORY:
{category}

ANSWER MODE:
{answer_mode}

COMPANY POLICY CONTEXT:
{context}

PREVIOUS CONVERSATION:
{history}

WORKER QUESTION:
{message}

IMPORTANT INSTRUCTIONS:

If ANSWER MODE is COMPANY_POLICY:

- Answer using the supplied company policy context.
- Treat the context as the source of company-specific
  information.
- Do not invent company-specific rules.
- Explain the policy in simple language.
- Use bullet points when useful.
- If useful, mention the relevant document and page.

If ANSWER MODE is GENERAL_KNOWLEDGE:

- The company policy documents do not contain sufficiently
  relevant information for this question.
- Answer using your general knowledge.
- Do NOT present the answer as company policy.
- Start the answer with:

  "This is general information, not a company-specific policy."

- If the question is about Indian law, regulations, or
  workers' legal rights, clearly say that this is general
  information and should be verified with the appropriate
  official authority, HR, or legal department.
- Do not invent company-specific benefits, salaries,
  leave rules, emergency numbers, procedures, or policies.

For emergency questions:

- Prioritize immediate safety.
- Use company policy context if relevant information
  is available.
- Never invent mine-specific emergency numbers,
  evacuation routes, assembly points, or procedures.
- If the required emergency information is not available,
  instruct the worker to follow the official mine emergency
  procedure and contact the designated mine emergency
  authority.

GENERAL RULES:

- Use simple language.
- Use short sentences.
- Use bullet points when useful.
- Respond in the requested language.
- Never claim that you contacted emergency services.

Answer the worker now.
"""

    # --------------------------------
    # Groq LLM
    # --------------------------------

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    # --------------------------------
    # Get answer
    # --------------------------------

    answer = response.choices[0].message.content

    # --------------------------------
    # Save conversation
    # --------------------------------

    conversation_memory.add_message(
        session_id,
        "user",
        message
    )

    conversation_memory.add_message(
        session_id,
        "assistant",
        answer
    )

    # --------------------------------
    # Sources
    # --------------------------------

    sources = []

    for result in relevant_results:

        sources.append({

            "document": result["document"],

            "page": result["page"]

        })

    # --------------------------------
    # Return API response
    # --------------------------------

    return {

        "answer": answer,

        "language": language_code,

        "category": category,

        "session_id": session_id,

        "sources": sources

    }
