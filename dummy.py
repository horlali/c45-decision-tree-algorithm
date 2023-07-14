import random

import requests

# Make a request to the US cities API
response = requests.get(
    "https://public.opendatasoft.com/api/records/1.0/search/?dataset=us-cities-top-1k&rows=1000"
)

# Get the JSON data from the response
data = response.json()

print(data)
# # Extract the city names from the data
# cities = [record['fields']['city'] for record in data['records']]

# # Choose 100 random cities from the list
# random_cities = random.sample(cities, 100)

# # Print the list of random cities
# print(random_cities)
