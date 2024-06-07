from surprise import dump, Dataset, Reader, SVD
import pandas as pd

# Load the trained model
model = dump.load('recommendation_model')[1]

# Load the movies dataset
movies = pd.read_csv('movie.csv')

# Create a Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

# Build a basic recommendation function
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get recommendations for a user
user_id = 1  # Replace with the user ID you want to recommend movies for
testset = data.build_full_trainset()
predictions = model.test(testset.build_testset())

top_n = get_top_n_recommendations(predictions, n=10)
user_recommendations = top_n[user_id]

# Print recommended movies
for movie_id, estimated_rating in user_recommendations:
    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(f'Recommended: {movie_title} (Estimated rating: {estimated_rating:.2f})')
