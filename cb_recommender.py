import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Load the restaurant data
try:
    restaurant_data = pd.read_csv('restaurant_data.csv')
except FileNotFoundError:
    print("The file 'restaurant_data.csv' was not found. Please make sure the file exists and try again.")

# Get user input for cuisine preference and location
cuisine_preference = ['Italian', 'Mediterranean']
latitude = 30.1061028
longitude = 31.365905

# Filter out the restaurants that do not serve the user's preferred cuisines
restaurant_data = restaurant_data[restaurant_data['cuisines'].apply(lambda x: any(cuisine in x for cuisine in cuisine_preference))]

# Prioritize restaurants that serve multiple cuisine types
restaurant_data['num_cuisines'] = restaurant_data['cuisines'].apply(lambda x: len(x.split(',')))
restaurant_data.sort_values(by=['num_cuisines'], ascending=False, inplace=True)

# Find the optimal number of clusters for this specific data
ssd = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(restaurant_data[['Latitude', 'Longitude']])
    ssd.append(kmeans.inertia_)
optimal_clusters = np.argmin(np.diff(ssd)) + 1

# Use K-Means to cluster the restaurants based on their location (latitude and longitude)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
kmeans.fit(restaurant_data[['Latitude', 'Longitude']])
restaurant_data['cluster'] = kmeans.predict(restaurant_data[['Latitude', 'Longitude']])

# Find the closest cluster to the user's location
user_cluster = kmeans.predict([[latitude, longitude]])[0]

# Filter out the restaurants that are not in the closest cluster
recommended_restaurants = restaurant_data[restaurant_data['cluster'] == user_cluster]

# Use KNN to find the nearest restaurants to the user's location in the closest cluster.
knn = NearestNeighbors(n_neighbors=len(recommended_restaurants), algorithm='ball_tree')
knn.fit(recommended_restaurants[['Latitude', 'Longitude']])
distances, indices = knn.kneighbors([[latitude, longitude]])

# Select the highest-rated restaurants and rearrange them randomly.
recommended_restaurants = recommended_restaurants.iloc[indices[0]].sort_values(by=['rate'], ascending=False).reset_index(drop=True)
recommended_restaurants = recommended_restaurants.sample(frac=1).reset_index(drop=True)

# Ask the user for the number of recommended restaurants
num_recommendations = int(input("How many restaurants would you like to be recommended? "))

# Print the recommended restaurants in a nice format
print(f"\nHere are {num_recommendations} recommended restaurants based on your preferences and location:\n")
for i in range(num_recommendations):
    print(f"{i+1}. Name: {recommended_restaurants['Name'][i]}")
    print(f"   Cuisine: {recommended_restaurants['cuisines'][i]}")
    print(f"   Location: {recommended_restaurants['location'][i]}")
    print(f"   Rating: {recommended_restaurants['rate'][i]} ({recommended_restaurants['number of reviews'][i]} Reviews)")
    print()
