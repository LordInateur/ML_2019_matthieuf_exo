import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# We'll use the famous SVD algorithm.
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

"""
print(predictions)

Prediction(uid='752', iid='539', r_ui=4.0, est=2.655070017379331, details={'was_impossible': False}), 
Prediction(uid='916', iid='163', r_ui=3.0, est=3.2915252200292375, details={'was_impossible': False}),

"""

graph = []
for prediciton in predictions:
    graph.append( prediciton.r_ui - prediciton.est)


plt.hist(graph, 120)
plt.show()


