import os
import socket
import requests
import time
import dotenv
import polyline

# Load environment variables from .env file
dotenv.load_dotenv()

# Retrieve the Mapbox API key from the environment variable
api_key = os.getenv('MAPBOX_API_KEY')

# Get the Raspberry Pi IP address from the environment variable
RPI_IP_ADDRESS = os.getenv('RPI_IP_ADDRESS')

# Set up socket connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((RPI_IP_ADDRESS, 5000))  # Replace with Raspberry Pi IP address
print('Socket connection established')

# Set the destination coordinates
destination = '8.557840,39.287434'  # Example: New York, NY

# Set the desired distance threshold for reaching the destination (in meters)
distance_threshold = 300  # Example: 5 meters

# Set the duration between updates (in seconds)
update_interval = 1  # Example: 1 second

# Function to calculate the distance between two GPS coordinates

# Construct the API request URL

url = f'https://api.mapbox.com/directions/v5/mapbox/driving/{destination};{destination}?access_token={api_key}'

# Send the API request to obtain the route data
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract the route data from the response
    data = response.json()

    # Extract the path coordinates
    polyline_data = data['routes'][0]['geometry']
    path = polyline.decode(polyline_data)

    # Start following the path
    for coordinate in path:
        # Convert the coordinate to a string
        coordinate_str = ','.join(str(c) for c in coordinate)

        # Send the coordinate to the Raspberry Pi
        sock.send(coordinate_str.encode())

        # Receive the distance to the destination from the Raspberry Pi
        # received_distance = sock.recv(1024).decode()
        # print('Received distance:', received_distance)

        # Convert the received distance to float
        try:
            # received_distance = float(received_distance)

            # Check if the distance to the destination is below the threshold
            if 1 >= distance_threshold:
                print('Destination reached')
                sock.sendall(b'stop')
            else:
                sock.sendall(b'forward')
            distance_threshold -=1
            print(distance_threshold)

            # Receive GPS data from the Raspberry Pi
            # gps_data = sock.recv(1024).decode()
            # print('Received GPS data:', gps_data)

            # Process the GPS data as per your requirements
            # Example: Extract latitude, longitude, etc. from the GPS data

            # Wait for the next update interval
            time.sleep(update_interval)
        except:
            pass

else:
    print('Error:', response.status_code)

# Close the socket connection
sock.close()
