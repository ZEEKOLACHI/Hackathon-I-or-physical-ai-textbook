"""LLM service for chat completions using OpenAI."""

from openai import AsyncOpenAI

from src.config import settings

# OpenAI client
_client: AsyncOpenAI | None = None

# Model configuration
CHAT_MODEL = "gpt-4o-mini"
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


def get_openai_client() -> AsyncOpenAI:
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


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
    client = get_openai_client()

    # Build messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add context as a system message
    if context:
        context_message = (
            "Here is relevant content from the textbook that may help answer "
            f"the user's question:\n\n{context}\n\n"
            "Use this information to provide an accurate, well-cited response."
        )
        messages.append({"role": "system", "content": context_message})

    # Add conversation history
    if conversation_history:
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Generate response
    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content or ""


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
    client = get_openai_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "The user has selected the following text from the textbook and "
                f"has a question about it:\n\n---\nSelected Text:\n{selected_text}\n---\n\n"
                f"Additional context:\n{context}"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return response.choices[0].message.content or ""
