from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db, engine
from models import Base
from schemas import UserCreate, UserResponse
import crud
from crud import verify_password, get_user_by_username
from schemas import (
    UserCreate,
    UserResponse,
    MovieCreate,
    MovieResponse,
    RatingResponse,
    RatingCreate,
    GenreClassificationRequest,
    GenreClassificationResponse,
)
from database import engine
from models import Base
from content_based_recommendation_service import ContentBasedRecommendationService
from collaborative_recommendation_service import CollaborativeRecommendationService
from genre_prediction_service import GenrePredictionService


app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
cbf_recommender_service = ContentBasedRecommendationService()
cobf_recommender_service = CollaborativeRecommendationService()
genre_prediction_service = GenrePredictionService()


# Create tables
@app.on_event("startup")
async def startup():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Preparing collaborative recommender")
        await cobf_recommender_service.setup()
        print("Preparing content-based recommender")
        await cbf_recommender_service.setup()
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise e


# User Register
@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    return await crud.create_user(db, user)


@app.get("/users/{user_id}", response_model=UserResponse)
async def create_user(user_id: int, db: AsyncSession = Depends(get_db)):
    return await crud.get_user(db, user_id)


# User Login
@app.post("/login/", response_model=UserResponse)
async def login(form_data: UserCreate, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


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


# Movie Search
@app.get("/search/movies", response_model=list[MovieResponse])
async def search_movies(
    name: str = Query(..., description="Search movie"),
    db: AsyncSession = Depends(get_db),
):
    movies = await crud.search_movie_by_name(db, name)
    if not movies:
        raise HTTPException(
            status_code=404, detail="No movies found with the given name"
        )
    return [
        {
            "movie_id": movie.movie_id,
            "title": movie.title,
        }
        for movie in movies
    ]


# Recommendation routes
@app.get("/movies/{movie_id}/recommendations")
async def get_movie_recommendations(
    movie_id: int,
    n_recommendations: int = 4,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    genres: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    genre_list = genres.split(",") if genres else None
    recommendations = await cbf_recommender_service.get_recommendations(
        db, movie_id, n_recommendations, min_year, max_year, genre_list
    )
    return {"recommendations": recommendations}


@app.get("/users/{user_id}/recommendations")
async def get_user_recommendations(
    user_id: int,
    n_recommendations: int = 4,
    min_rating: Optional[float] = 3.5,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    genres: Optional[str] = None,
    type: Optional[str] = "cbf",
    db: AsyncSession = Depends(get_db),
):
    genre_list = genres.split(",") if genres else None
    if type == "cbf":
        recommendations = await cbf_recommender_service.get_user_recommendations(
            db, user_id, n_recommendations, min_rating, min_year, max_year, genre_list
        )
    elif type == "cobf":
        recommendations = await cobf_recommender_service.get_recommendations_for_user(
            db, user_id, n_recommendations
        )
    return {"recommendations": recommendations}


# Rating Routes
@app.post("/ratings/", response_model=RatingResponse)
async def create_rating(
    rating: RatingCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    try:
        response = await crud.create_rating(db, rating)
        await cobf_recommender_service.update_user_rating()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ratings/", response_model=list[RatingResponse])
async def read_ratings(user_id: int, movie_id: int, db: AsyncSession = Depends(get_db)):
    return await crud.get_ratings(user_id, movie_id, db)


@app.get("/ratings/{rating_id}", response_model=RatingResponse)
async def read_rating(rating_id: int, db: AsyncSession = Depends(get_db)):
    rating = await crud.get_rating(db, rating_id)
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    return rating


# Genre Classification routes
@app.post("/classify", response_model=GenreClassificationResponse)
async def classify_genre(request: GenreClassificationRequest):
    predicted_genres = genre_prediction_service.predict_genres(request.plot_summary)
    response = dict()
    response["plot_summary"] = request.plot_summary
    response["predictions"] = []
    print(predicted_genres)
    for genre in predicted_genres.keys():
        prediction = dict()
        prediction["genre"] = genre
        prediction["confidence"] = predicted_genres[genre]
        response["predictions"].append(prediction)

    return response
