from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, update
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
    new_user = User(
        username=user.username,
        password=hashed_password,
        is_dataset_user=False,
        movies_watched=[],
    )
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        return new_user
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")


# Get user
async def get_user(db: AsyncSession, user_id: int):
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalars().first()


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
    result = await db.execute(
        select(Movie).where(Movie.title.ilike(f"%{name}%")).limit(10)
    )
    return result.scalars().all()


# Create a rating
async def create_rating(db: AsyncSession, rating: RatingCreate):
    new_rating = Rating(**rating.dict())
    db.add(new_rating)
    await db.commit()
    await db.refresh(new_rating)

    # add the rating to users watched list
    user = await get_user(db, rating.user_id)
    movies_watched = [movie_id for movie_id in user.movies_watched]
    movies_watched.append(rating.movie_id)
    update_query = (
        update(User)
        .where(User.id == rating.user_id)
        .values(movies_watched=movies_watched)
        .returning(User)
    )
    result = await db.execute(update_query)
    await db.commit()
    return new_rating


# Get all ratings
async def get_ratings(user_id: int, movie_id: int, db: AsyncSession):
    result = await db.execute(
        select(Rating)
        .where(and_(Rating.movie_id == movie_id, Rating.user_id == user_id))
        .limit(1)
    )
    return result.scalars().all()


# Get a specific rating
async def get_rating(db: AsyncSession, rating_id: int):
    result = await db.execute(select(Rating).where(Rating.rating_id == rating_id))
    return result.scalars().first()
