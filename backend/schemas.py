from pydantic import BaseModel, EmailStr
from typing import List


# User Schemas
class UserCreate(BaseModel):
    username: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: EmailStr
    movies_watched: List[int] = []

    class Config:
        orm_mode = True


# Movie Schemas
class MovieBase(BaseModel):
    title: str
    plot_summary: str = None
    release_year: int = None
    runtime: int = None
    language: str = None
    genres: List[str] = None


class MovieCreate(MovieBase):
    pass


class MovieResponse(MovieBase):
    movie_id: int

    class Config:
        orm_mode = True


# Rating Schemas
class RatingBase(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class RatingCreate(RatingBase):
    pass


class RatingResponse(RatingBase):
    rating_id: int

    class Config:
        orm_mode = True


class GenreClassificationRequest(BaseModel):
    plot_summary: str


class GenrePrediction(BaseModel):
    genre: str
    confidence: float


class GenreClassificationResponse(GenreClassificationRequest):
    predictions: List[GenrePrediction]
