import pandas as pd
from surprise import Dataset, Reader, SVD, dump

# Load the ratings dataset
rating = pd.read_csv('rating.csv')

# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

# Create a trainset
trainset = data.build_full_trainset()

# Build and train the SVD model
model = SVD()
model.fit(trainset)

# Save the trained model
dump.dump('recommendation_model', algo=model)

# Load the trained model (if you want to make predictions)
# loaded_model = dump.load('recommendation_model')[1]

# Example: Get recommendations for a user
user_id = 1  # Replace with the user ID you want to recommend movies for

# Create a list of all movie IDs
movie_ids = rating['movieId'].unique()

# Predict ratings for the user for all movies
user_predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]

# Sort predictions by estimated rating in descending order
user_predictions.sort(key=lambda x: x.est, reverse=True)

# Get top N movie recommendations (e.g., top 10)
top_n = user_predictions[:10]

# Extract movie titles for the recommendations
movie_titles = [rating[rating['movieId'] == prediction.iid]['title'].values[0] for prediction in top_n]

# Print recommended movie titles
for i, movie_title in enumerate(movie_titles):
    print(f"Recommendation {i + 1}: {movie_title}")
