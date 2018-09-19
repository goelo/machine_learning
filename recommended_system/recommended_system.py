import json
import numpy as np

# Returns the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
	if user1 not in dataset:
		raise TypeError('User ' + user1 + 'not present in dataset')

	if user2 not in dataset:
		raise TypeError('User ' + user2 + 'not present in dataset')


	#Movies rated by both user1 and user2
	rated_both = {}

	for item in dataset[user1]:
		if item in dataset[user2]:
			rated_both[item] = 1

	num_ratings = len(rated_both)

	#If there are no common movies, the score is 0
	if num_ratings == 0:
		return 0

	# Compute the sum of ratings of all the common preferences
	user1_sum = np.sum([dataset[user1][item] for item in rated_both])
	user2_sum = np.sum([dataset[user2][item] for item in rated_both])

	# Compute the sum of squared ratings of all the common preferences
	user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_both])
	user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_both])

	# Compute the sum of products of the common ratings
	total_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_both])

	# Computer the Pearson correlation
	Sxy = total_sum - (user1_sum * user2_sum / num_ratings)
	Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
	Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

	if Sxx * Syy == 0:
		return 0

	return Sxy / np.sqrt(Sxx * Syy)

def find_similar_users(dataset, user, num_users):
	if user not in dataset:
		raise TypeError('User ' + user + 'not present in the dataset')

	# Compute Pearson scores for all the users
	scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if user != x])

	# Sort the scores based on second column
	score_sorted = np.argsort(scores[:, 1])

	# Sort the scores in decreasing order (highest score first)
	scored_sorted_dec = scores_sorted[::-1]

	# Extract top 'k' indices
	top_k = scored_sorted_dec[0:num_users]

	return scores[top_k]

# Generate recommendations for a given user
def generate_recommendations(dataset, user):
	if user not in dataset:
		raise TypeError('User ' + user + ' not present in the dataset')

	total_scores = {}
	similarity_sums = {}

	for u in [x for x in dataset if x != user]:
		similarity_score = pearson_score(dataset, user, u)

		if similarity_score <= 0:
			continue

		for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
			total_scores.update({item: dataset[u][item] * similarity_score})
			similarity_sums.update({item: similarity_score})

	if len(total_scores) == 0:
		return ['No recommendations possible']

	# Create the normalized list
	movie_ranks = np.array([[total/similarity_sums[item], item]
			for item, total in total_scores.items()])

	# Sort in decreasing order based on the first column
	movie_ranks = movies_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

	# Extract the recommended movies
	recommendations = [movie for _, movie in movie_ranks]

	return recommendations
	
if __name__=='__main__':
	data_file = 'movie_ratings.json'

	with open(data_file, 'r') as f:
		data = json.loads(f.read())

	user = 'Michael Henry'
	print("\nRecommendations for " + user + ":")
	movies = generate_recommendations(data, user) 
    for i, movie in enumerate(movies):
    	print(str(i+1) + '. ' + movie)

    user = 'John Carson' 
    print("\nRecommendations for " + user + ":")
    movies = generate_recommendations(data, user) 
    for i, movie in enumerate(movies):
        print(str(i+1) + '. ' + movie)


