from groq import Groq

from app.config import settings
from app.services.rag import rag
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

Your job is to help coal mine workers understand
company policies, safety procedures, benefits,
workplace rules and general information.

IMPORTANT RULES:

1. Company policy information must come from the
   supplied policy documents.

2. Never invent a company policy.

3. If the requested policy information is not present
   in the supplied documents, clearly say:

   "I could not find this information in the available
   company policy documents."

4. Explain policies in very simple language.

5. Use short sentences.

6. Prefer bullet points when explaining procedures.

7. Do not use complicated legal or technical language
   unless necessary.

8. Always respond in the requested language.

9. Supported languages:
   English
   Hindi
   Bengali

10. If the worker asks a general question that is not
   related to company policy, you may answer using
   general knowledge.

11. Do not pretend that general knowledge is company policy.

12. For emergency situations, prioritize immediate safety.

13. Never invent mine-specific emergency phone numbers,
   evacuation routes, assembly points or procedures.

14. If emergency information is not available in the
   supplied company documents, tell the worker to follow
   the official emergency procedure and contact the
   designated mine emergency authority.

15. Never claim that you contacted emergency services.

16. Do not request unnecessary personal information.

17. Do not identify or track individual workers.

18. The assistant is common for all workers.
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

    if requested_language == "auto":
        language_code = detect_language(message)
    else:
        language_code = requested_language

    language = language_name(language_code)

    category = classify_question(message)

    search_results = rag.search(
        message,
        top_k=4
    )

    # Only use RAG if similarity is good enough
    MIN_SCORE = 0.45

    relevant_results = [
        result
        for result in search_results
        if result["score"] >= MIN_SCORE
    ]

    context = build_context(relevant_results)
    history = build_history(session_id)

    if relevant_results:
        answer_mode = "COMPANY_POLICY"
    else:
        answer_mode = "GENERAL_KNOWLEDGE"

    prompt = f"""
{SYSTEM_PROMPT}

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
- Treat the context as the source of company-specific information.
- Do not invent company-specific rules.
- Explain the policy in simple language.
- If useful, mention the relevant policy document/page.

If ANSWER MODE is GENERAL_KNOWLEDGE:

- The company policy documents do not contain sufficiently
  relevant information for this question.
- Answer using your general knowledge.
- Do NOT present the answer as company policy.
- Start the answer with:
  "This is general information, not a company-specific policy."
- If the question is about Indian law, regulations, or workers'
  legal rights, make it clear that this is general information
  and that the worker should verify the current law with the
  appropriate official authority or company HR/legal department.
- Do not invent company-specific benefits, salaries, leave rules,
  emergency numbers, procedures, or policies.

For emergency questions:

- Prioritize immediate safety.
- Use company policy context if relevant information is available.
- Never invent mine-specific emergency numbers, evacuation routes,
  assembly points, or procedures.
- If the required emergency information is not available,
  instruct the worker to follow the official mine emergency
  procedure and contact the designated mine emergency authority.

GENERAL RULES:

- Use simple language.
- Use short sentences.
- Use bullet points when useful.
- Respond in the requested language.
- Never claim that you contacted emergency services.

Answer the worker now.
"""

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

    answer = response.choices[0].message.content

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

    sources = []

    for result in relevant_results:
        sources.append({
            "document": result["document"],
            "page": result["page"]
        })

    return {
        "answer": answer,
        "language": language_code,
        "category": category,
        "session_id": session_id,
        "sources": sources
    }