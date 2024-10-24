import numpy as np
import pandas as pd

# Load the dataset
data_path = "C:/Users/User/Desktop/MACHINE LEARNING/ml-1m/ratings.dat"
df = pd.read_csv(data_path, header=None, sep="::", engine="python")
df.columns = ["user_id", "movie_id", "rating", "timestamp"]

# Get the number of unique users and movies
n_users = df["user_id"].nunique()
n_movies = df["movie_id"].nunique()


# Function to load user rating data and map movie IDs
def load_user_rating_data(df):  # type: ignore
    # Create movie ID to column index mapping
    movie_id_mapping = {
        movie_id: i for i, movie_id in enumerate(df["movie_id"].unique())
    }

    # Initialize a matrix with zeros for user ratings
    data = np.zeros((n_users, len(movie_id_mapping)), dtype=int)

    # Fill in the matrix with user ratings
    for row in df.itertuples(index=False):
        data[row.user_id - 1, movie_id_mapping[row.movie_id]] = row.rating

    return data, movie_id_mapping


# Load the rating data and movie ID mapping
data, movie_id_mapping = load_user_rating_data(df)

# Analyze the data distribution (check for class imbalance)
unique_ratings, rating_counts = np.unique(data, return_counts=True)
for rating, count in zip(unique_ratings, rating_counts):
    print(f"Number of rating {rating}: {count}")

# Find the movie with the most ratings
most_rated_movie_id = df["movie_id"].value_counts().idxmax()
print(f"Most rated movie: {most_rated_movie_id}")

# Construct the dataset for prediction
X_raw = np.delete(data, movie_id_mapping[most_rated_movie_id], axis=1)  # Features
Y_raw = data[:, movie_id_mapping[most_rated_movie_id]]  # Target

# Filter out the known ratings for the target movie
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

# Display the shapes of X and Y
print("Feature set (X):", X.shape)
print("Target ratings (Y):", Y.shape)

# Convert ratings to binary classification
recommend = 3
Y_binary = np.where(Y <= recommend, 0, 1)  # 0 for ratings <= 3, 1 for ratings > 3

# Count positive and negative samples
n_pos = (Y_binary == 1).sum()
n_neg = (Y_binary == 0).sum()
print(f"{n_pos} positive samples and {n_neg} negative samples")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Load the dataset
data_path = "C:/Users/User/Desktop/MACHINE LEARNING/ml-1m/ratings.dat"
df = pd.read_csv(data_path, header=None, sep="::", engine="python")
df.columns = ["user_id", "movie_id", "rating", "timestamp"]

# Get the number of unique users and movies
n_users = df["user_id"].nunique()
n_movies = df["movie_id"].nunique()


# Function to load user rating data and map movie IDs
def load_user_rating_data(df):
    # Create movie ID to column index mapping
    movie_id_mapping = {
        movie_id: i for i, movie_id in enumerate(df["movie_id"].unique())
    }

    # Initialize a matrix with zeros for user ratings
    data = np.zeros((n_users, len(movie_id_mapping)), dtype=int)

    # Fill in the matrix with user ratings
    for row in df.itertuples(index=False):
        data[row.user_id - 1, movie_id_mapping[row.movie_id]] = row.rating

    return data, movie_id_mapping


# Load the rating data and movie ID mapping
data, movie_id_mapping = load_user_rating_data(df)

# Select the most rated movie (for example, movie ID 2858)
target_movie_id = 2858

# Define a threshold for binary classification (e.g., recommend rating >= 4)
threshold = 3

# Exclude the target movie from features (X)
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)

# Target movie ratings (Y)
Y_raw = data[:, movie_id_mapping[target_movie_id]]

# Only include rows where the user has rated the target movie (non-zero ratings)
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

# Convert Y into a binary classification (1 for recommend, 0 for not recommend)
Y_binary = (Y > threshold).astype(int)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y_binary, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier (MultinomialNB)
clf = MultinomialNB(alpha=2.0, fit_prior=True)
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)
print(y_pred[:10])

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the confusion matrix and classification report for further evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report (includes precision, recall, f1-score)
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
