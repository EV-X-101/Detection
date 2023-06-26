import os
from dotenv import load_dotenv
import requests
import polyline

# Load environment variables from .env file
load_dotenv()

# Retrieve the Mapbox API key from the environment variable
api_key = os.getenv('MAPBOX_API_KEY')

# Set the destination coordinates
destination = '8.557840,39.287434'  # Example: New York, NY

# Get the current location coordinates using GPS or other means
current_location = '8.563830,39.289224'  # Example: 37.7749,-122.4194

# Construct the API request URL
url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{current_location};{destination}?alternatives=true&geometries=polyline&language=en&overview=full&steps=true&access_token={api_key}'

# Send the API request
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the route data from the response
    data = response.json()
    
    # Get the distance and duration of the route
    distance = data['routes'][0]['distance']
    duration = data['routes'][0]['duration']
    
    # Extract the polyline coordinates
    polyline_data = data['routes'][0]['geometry']
    path = polyline.decode(polyline_data, geojson=True)
    
    # Print the distance, duration, and path coordinates
    print('Distance:', distance, 'meters')
    print('Duration:', duration, 'seconds')
    print('Path:', path)
    
    # Extract the bearing information for each step
    steps = data['routes'][0]['legs'][0]['steps']
    bearings = [step['maneuver']['bearing_after'] for step in steps]
    
    # Print the bearings for each step
    print('Bearings:', bearings)
    
else:
    print('Error:', response.status_code)
