from genre_classifier import MovieGenreClassifier
from enhanced_genre_classifier import EnhancedMovieGenreClassifier

plot = "Marty McFly, a 17-year-old high school student, is accidentally sent 30 years into the past in a time-traveling DeLorean invented by his close friend, the maverick scientist Doc Brown."

mgc = MovieGenreClassifier()
cls0 = mgc.load_model("./models/genre_classifier_model.pkl")

emgc = EnhancedMovieGenreClassifier()
cls1 = emgc.load_model("./models/enhanced_genre_classifier_model.pkl")

genres = cls0.predict(plot)
print(genres)

genres, scores = cls1.predict(plot)
print(genres, scores)