from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import IntegrityError
from models import User, Movie, Rating
from schemas import UserCreate, MovieCreate, RatingCreate
from fastapi import HTTPException
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Hash password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# Verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# Create a new user
async def create_user(db: AsyncSession, user: UserCreate):
    hashed_password = hash_password(user.password)
    new_user = User(username=user.username, password=hashed_password)
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        return new_user
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")


# Get user by username
async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(User).where(User.username == username))
    return result.scalars().first()


# Create a new movie
async def create_movie(db: AsyncSession, movie: MovieCreate):
    new_movie = Movie(**movie.dict())
    db.add(new_movie)
    await db.commit()
    await db.refresh(new_movie)
    return new_movie


# Get all movies
async def get_movies(db: AsyncSession):
    result = await db.execute(select(Movie))
    return result.scalars().all()


# Get a specific movie
async def get_movie(db: AsyncSession, movie_id: int):
    result = await db.execute(select(Movie).where(Movie.movie_id == movie_id))
    return result.scalars().first()


# Search a movie
async def search_movie_by_name(db: AsyncSession, name: str):
    # Use `ILIKE` for case-insensitive search in PostgreSQL
    result = await db.execute(select(Movie).where(Movie.title.ilike(f"%{name}%")))
    return result.scalars().all()


# Create a rating
async def create_rating(db: AsyncSession, rating: RatingCreate):
    new_rating = Rating(**rating.dict())
    db.add(new_rating)
    await db.commit()
    await db.refresh(new_rating)
    return new_rating


# Get all ratings
async def get_ratings(db: AsyncSession):
    result = await db.execute(select(Rating))
    return result.scalars().all()


# Get a specific rating
async def get_rating(db: AsyncSession, rating_id: int):
    result = await db.execute(select(Rating).where(Rating.rating_id == rating_id))
    return result.scalars().first()
