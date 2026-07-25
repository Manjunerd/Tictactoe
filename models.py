from sqlalchemy import Column, Integer, String, DateTime, func
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    google_id = Column(String, unique=True, nullable=False)

    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    picture = Column(String)

    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    games = Column(Integer, default=0)

    highest_streak = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)

    created_at = Column(DateTime(timezone=True), server_default=func.now())