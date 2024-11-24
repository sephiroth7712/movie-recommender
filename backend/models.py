from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    ForeignKey,
    TIMESTAMP,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# User Model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)  # Email as username
    password = Column(String, nullable=False)  # Hashed password


# Movie Model
class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(
        Integer, primary_key=True, index=True, autoincrement=True
    )  # SERIAL in PostgreSQL
    title = Column(String(255), nullable=False)  # VARCHAR(255)
    plot_summary = Column(Text, nullable=True)  # TEXT
    release_year = Column(Integer, nullable=True)  # INTEGER
    runtime = Column(Integer, nullable=True)  # INTEGER
    language = Column(String(50), nullable=True)  # VARCHAR(50)
    genres = Column(ARRAY(String), nullable=True)  # TEXT[]
    plot_vector = Column(TSVECTOR, nullable=True)  # tsvector
    feature_vector = Column(ARRAY(Float), nullable=True)  # float[]


# Rating Model
class Rating(Base):
    __tablename__ = "ratings"

    rating_id = Column(
        Integer, primary_key=True, autoincrement=True
    )  # SERIAL in PostgreSQL
    user_id = Column(
        Integer, ForeignKey("users.id"), nullable=False
    )  # Foreign key to users(id)
    movie_id = Column(
        Integer, ForeignKey("movies.movie_id"), nullable=False
    )  # Foreign key to movies(movie_id)
    rating = Column(Float, nullable=False)  # FLOAT for user ratings
    timestamp = Column(
        TIMESTAMP, nullable=True
    )  # TIMESTAMP for when the rating was given

    # Unique constraint to ensure a user can rate a movie only once
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="unique_user_movie"),
    )
