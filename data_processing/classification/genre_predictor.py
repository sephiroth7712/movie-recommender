from genre_classifier import MovieGenreClassifier
from enhanced_genre_classifier import EnhancedMovieGenreClassifier

plot = "A cyborg assassin from the future attempts to find and kill a young woman who is destined to give birth to a warrior that will lead a resistance to save humankind from extinction."

mgc = MovieGenreClassifier()
cls0 = mgc.load_model("./models/genre_classifier_model.pkl")

emgc = EnhancedMovieGenreClassifier()
cls1 = emgc.load_model("./models/enhanced_genre_classifier_model.pkl")

genres = cls0.predict(plot)
print(genres)

genres, scores = cls1.predict(plot)
print(genres, scores)