"""Authentication service for user management and session handling."""

import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import Session, User

# Session configuration
SESSION_TOKEN_BYTES = 32
SESSION_EXPIRY_DAYS = 30


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_session_token() -> str:
    """Generate a cryptographically secure session token."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    programming_level: str = "beginner",
    robotics_level: str = "beginner",
    hardware_available: list[str] | None = None,
) -> User:
    """
    Create a new user.

    Args:
        db: Database session
        email: User email
        password: Plain text password
        programming_level: Programming experience level
        robotics_level: Robotics experience level
        hardware_available: List of available hardware

    Returns:
        Created user

    Raises:
        ValueError: If email is already registered
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise ValueError("Email already registered")

    # Create user
    user = User(
        email=email,
        password_hash=hash_password(password),
        programming_level=programming_level,
        robotics_level=robotics_level,
        hardware_available=hardware_available or [],
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


async def authenticate_user(
    db: AsyncSession,
    email: str,
    password: str,
) -> User | None:
    """
    Authenticate a user with email and password.

    Args:
        db: Database session
        email: User email
        password: Plain text password

    Returns:
        User if authentication successful, None otherwise
    """
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user


async def create_session(db: AsyncSession, user_id: uuid.UUID) -> Session:
    """
    Create a new session for a user.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        Created session
    """
    session = Session(
        user_id=user_id,
        token=generate_session_token(),
        expires_at=datetime.now(timezone.utc) + timedelta(days=SESSION_EXPIRY_DAYS),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    return session


async def get_session_by_token(db: AsyncSession, token: str) -> Session | None:
    """
    Get a session by its token.

    Args:
        db: Database session
        token: Session token

    Returns:
        Session if found and valid, None otherwise
    """
    result = await db.execute(
        select(Session).where(Session.token == token)
    )
    session = result.scalar_one_or_none()

    if not session:
        return None

    if session.is_expired:
        # Delete expired session
        await db.delete(session)
        await db.commit()
        return None

    return session


async def get_user_by_session_token(db: AsyncSession, token: str) -> User | None:
    """
    Get a user by their session token.

    Args:
        db: Database session
        token: Session token

    Returns:
        User if session is valid, None otherwise
    """
    session = await get_session_by_token(db, token)
    if not session:
        return None

    result = await db.execute(select(User).where(User.id == session.user_id))
    return result.scalar_one_or_none()


async def invalidate_session(db: AsyncSession, token: str) -> bool:
    """
    Invalidate a session by deleting it.

    Args:
        db: Database session
        token: Session token

    Returns:
        True if session was deleted, False if not found
    """
    result = await db.execute(select(Session).where(Session.token == token))
    session = result.scalar_one_or_none()

    if not session:
        return False

    await db.delete(session)
    await db.commit()
    return True


async def invalidate_all_user_sessions(db: AsyncSession, user_id: uuid.UUID) -> int:
    """
    Invalidate all sessions for a user.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        Number of sessions deleted
    """
    result = await db.execute(select(Session).where(Session.user_id == user_id))
    sessions = result.scalars().all()

    count = len(sessions)
    for session in sessions:
        await db.delete(session)

    await db.commit()
    return count


async def update_user_profile(
    db: AsyncSession,
    user: User,
    programming_level: str | None = None,
    robotics_level: str | None = None,
    hardware_available: list[str] | None = None,
) -> User:
    """
    Update a user's profile.

    Args:
        db: Database session
        user: User to update
        programming_level: New programming level (optional)
        robotics_level: New robotics level (optional)
        hardware_available: New hardware list (optional)

    Returns:
        Updated user
    """
    if programming_level is not None:
        user.programming_level = programming_level
    if robotics_level is not None:
        user.robotics_level = robotics_level
    if hardware_available is not None:
        user.hardware_available = hardware_available

    await db.commit()
    await db.refresh(user)

    return user
