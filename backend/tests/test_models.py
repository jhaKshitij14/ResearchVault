
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.models import Document, User


@pytest.fixture
def session():
    # Each test uses a separate temporary SQLite database.
    # PostgreSQL-specific behavior will be tested separately.
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        yield db

    engine.dispose()


def create_user(session, email="test@example.com"):
    user = User(
        email=email,
        password_hash="test-hash",
    )
    session.add(user)
    session.commit()
    return user


def test_user_can_be_created(session):
    user = create_user(session)

    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.password_hash == "test-hash"


def test_document_defaults_to_uploaded(session):
    user = create_user(session)

    document = Document(
        owner_id=user.id,
        title="Networking Notes",
        content_hash="a" * 64,
    )
    session.add(document)
    session.commit()

    assert document.id is not None
    assert document.status == "uploaded"


def test_user_email_must_be_unique(session):
    create_user(session)
    session.add(
        User(
            email="test@example.com",
            password_hash="another-hash",
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()


def test_document_rejects_invalid_status(session):
    user = create_user(session)

    session.add(
        Document(
            owner_id=user.id,
            title="Invalid Document",
            content_hash="b" * 64,
            status="invalid",
        )
    )

    with pytest.raises(IntegrityError):
        session.commit()

    session.rollback()
