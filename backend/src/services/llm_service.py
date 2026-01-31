"""LLM service for chat completions using Google Gemini."""

import google.generativeai as genai

from src.config import settings

# Initialize Google AI
_initialized = False
_model = None

# Model configuration
CHAT_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

# System prompt for the textbook assistant
SYSTEM_PROMPT = """You are an expert AI assistant for the "Physical AI & Humanoid Robotics" textbook. Your role is to help learners understand concepts related to:

- ROS 2 (Robot Operating System 2)
- Robot perception (computer vision, sensor fusion, SLAM)
- Motion planning and navigation
- Robot control systems
- Machine learning for robotics
- Humanoid robot design and locomotion
- System integration

When answering questions:
1. Use the provided context from the textbook to give accurate answers
2. Always cite your sources by referencing the chapter and section
3. If the context doesn't contain enough information, acknowledge limitations
4. Explain complex concepts in a clear, educational manner
5. Use code examples when relevant, especially for ROS 2 concepts
6. Be encouraging and supportive to learners at all levels

Format your responses with clear structure using markdown when helpful."""


def _init_google_ai():
    """Initialize Google AI client and model."""
    global _initialized, _model
    if not _initialized:
        genai.configure(api_key=settings.google_api_key)
        _model = genai.GenerativeModel(
            CHAT_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        _initialized = True
    return _model


async def generate_response(
    user_message: str,
    context: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Generate a response using the LLM with RAG context.

    Args:
        user_message: User's question or message
        context: Retrieved context from the textbook
        conversation_history: Optional list of previous messages

    Returns:
        Generated response text
    """
    model = _init_google_ai()

    # Build prompt with context
    prompt_parts = []

    if context:
        prompt_parts.append(
            "Here is relevant content from the textbook that may help answer "
            f"the user's question:\n\n{context}\n\n"
            "Use this information to provide an accurate, well-cited response."
        )

    # Add conversation history
    if conversation_history:
        prompt_parts.append("\nPrevious conversation:")
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {msg['content']}")

    # Add current user message
    prompt_parts.append(f"\nUser question: {user_message}")

    full_prompt = "\n".join(prompt_parts)

    # Generate response
    response = model.generate_content(full_prompt)

    return response.text or ""


async def generate_response_with_selected_text(
    user_message: str,
    selected_text: str,
    context: str,
) -> str:
    """
    Generate a response for a question about selected text.

    Args:
        user_message: User's question about the selection
        selected_text: Text selected by the user
        context: Additional context from RAG search

    Returns:
        Generated response text
    """
    model = _init_google_ai()

    prompt = (
        "The user has selected the following text from the textbook and "
        f"has a question about it:\n\n---\nSelected Text:\n{selected_text}\n---\n\n"
        f"Additional context:\n{context}\n\n"
        f"User question: {user_message}"
    )

    response = model.generate_content(prompt)

    return response.text or ""
