# Movie Genre Classification and Recommendation System

A full-stack application that combines automated movie genre classification with personalized movie recommendations. The system uses machine learning to analyze movie plots and predict genres, while offering both content-based and collaborative filtering approaches for movie recommendations.

## Features

### Genre Classification
- Automated multi-label genre classification based on movie plot summaries
- TF-IDF vectorization and Support Vector Machines for classification
- Real-time genre predictions with confidence scores
- Support for multiple genres per movie

### Movie Recommendations
- Hybrid recommendation system combining:
  - Content-based filtering using movie plots and metadata
  - Collaborative filtering based on user ratings
- Personalized recommendations based on user watch history
- Similar movie suggestions
- Configurable recommendation parameters

### User Features
- User authentication and profile management
- Personal watchlist
- Movie rating system
- Search functionality
- Detailed movie information display

## Technical Stack

### Backend
- FastAPI for the REST API
- SQLAlchemy for database ORM
- PostgreSQL for data storage
- Python-based machine learning models:
  - scikit-learn for classification
  - NLTK and spaCy for text processing
  - NumPy and Pandas for data manipulation

### Frontend
- React with TypeScript
- Tailwind CSS for styling
- React Query for data fetching
- React Router for navigation
- Zustand for state management

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- PostgreSQL 13+
- Git

### Backend Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd movie-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. Set up environment variables:
Create a `.env` file in the backend directory with:
```
DATABASE_USER=your_db_user
DATABASE_PASSWORD=your_db_password
DATABASE_DOMAIN=localhost
DATABASE_NAME=movie_recommender
```

5. Initialize the database:
```bash
python data_processing/preprocessing/preprocessing.py
python backend/data_loader.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file:
```
VITE_API_URL=http://localhost:8000
```

4. Start the development server:
```bash
npm run dev
```

## Usage Examples

### Genre Classification
```python
from genre_prediction_service import GenrePredictionService

# Initialize the service
service = GenrePredictionService()

# Predict genres for a movie plot
plot = "A police officer must save hostages from a building taken over by terrorists."
genres = service.predict_genres(plot)
```

### Movie Recommendations
```python
from recommendation_service import RecommendationService

# Initialize the service
service = RecommendationService()

# Get recommendations for a user
recommendations = await service.get_user_recommendations(
    user_id=1,
    n_recommendations=5,
    min_rating=3.5
)
```

## API Endpoints

### Authentication
- `POST /login/` - User login
- `POST /users/` - User registration

### Movies
- `GET /movies/{movie_id}` - Get movie details
- `GET /search/movies` - Search movies
- `GET /movies/{movie_id}/recommendations` - Get similar movies

### Recommendations
- `GET /users/{user_id}/recommendations` - Get personalized recommendations

### Genre Classification
- `POST /classify` - Predict genres for a plot summary

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- MovieLens dataset for movie data and ratings
- CMU Movie Summary Corpus for plot summaries
- IMDb for additional movie metadata
