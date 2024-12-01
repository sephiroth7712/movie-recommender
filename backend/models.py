from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, UniqueConstraint, Index, Boolean
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


# User Model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)  # Email as username
    password = Column(String, nullable=False)  # Hashed password
    is_dataset_user = Column(Boolean, nullable=False)
    movies_watched = Column(ARRAY(Integer), nullable=False)


class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    plot_summary = Column(Text, nullable=True)
    release_year = Column(String, nullable=True)
    runtime = Column(Float, nullable=True)
    genres = Column(ARRAY(String), nullable=True)
    imdb_id = Column(String, nullable=True)
    tmdb_id = Column(String, nullable=True)

    __table_args__ = (
        Index('idx_movie_title', title),
        Index('idx_movie_year', release_year),
    )

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
    is_dataset_rating = Column(Boolean, nullable=False, default=False)
    # Unique constraint to ensure a user can rate a movie only once
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="unique_user_movie"),
    )
