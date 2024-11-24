from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import timedelta, datetime
from database import get_db, config
from sqlalchemy.ext.asyncio import AsyncSession
from schemas import UserResponse
import crud
from crud import verify_password, get_user_by_username
from schemas import (
    UserCreate,
    UserResponse,
    MovieCreate,
    MovieResponse,
    RatingResponse,
    RatingCreate,
)
from database import engine
from models import Base
from datetime import datetime, timedelta, timezone
from fastapi import Query

app = FastAPI()


def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    # Use timezone-aware UTC datetime
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, config["SECRET_KEY"], algorithm=config["ALGORITHM"]
    )
    return encoded_jwt


# Create tables
@app.on_event("startup")
async def startup():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise e


# User Register
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_user(db, user)


# User Login
@app.post("/login/")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=int(config["ACCESS_TOKEN_EXPIRE_MINUTES"]))
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
):
    try:
        payload = jwt.decode(
            token, config["SECRET_KEY"], algorithms=[config["ALGORITHM"]]
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
        user = await get_user_by_username(db, username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )
        return user
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


# Movie Routes
@app.post("/movies/", response_model=MovieResponse)
async def create_movie(movie: MovieCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_movie(db, movie)


@app.get("/movies/", response_model=list[MovieResponse])
async def read_movies(db: AsyncSession = Depends(get_db)):
    return await crud.get_movies(db)


@app.get("/movies/{movie_id}", response_model=MovieResponse)
async def read_movie(movie_id: int, db: AsyncSession = Depends(get_db)):
    movie = await crud.get_movie(db, movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie

#Movie Search
@app.get("/movies/search/", response_model=list[MovieResponse])
async def search_movies(
    name: str = Query(
        ..., description="Search movie"
    ),
    db: AsyncSession = Depends(get_db),
):
    movies = await crud.search_movie_by_name(db, name)
    if not movies:
        raise HTTPException(
            status_code=404, detail="No movies found with the given name"
        )
    return movies


# Rating Routes
@app.post("/ratings/", response_model=RatingResponse)
async def create_rating(rating: RatingCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_rating(db, rating)


@app.get("/ratings/", response_model=list[RatingResponse])
async def read_ratings(db: AsyncSession = Depends(get_db)):
    return await crud.get_ratings(db)


@app.get("/ratings/{rating_id}", response_model=RatingResponse)
async def read_rating(rating_id: int, db: AsyncSession = Depends(get_db)):
    rating = await crud.get_rating(db, rating_id)
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    return rating
