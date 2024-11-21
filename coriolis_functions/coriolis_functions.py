import numpy as np

def main():
    print("This file is not intended to be run.")

# Calculation of Earth's radius at a given latitude
def radius_at_latitude(lat):
    """
    Calculates the Earth's radius at a specific latitude.
    
    Args:
        lat (float): Latitude in degrees.

    Returns:
        float: The interpolated Earth radius at the given latitude, in kilometers.
    """
    # Convert latitude from degrees to radians
    lat_rad = np.radians(lat)
    
    # Earth's polar radius = 6371.0 km, equatorial radius = 6378.137 km
    # Interpolating between the two based on the latitude
    return 6371.0 * (1 - 0.5 * (1 - np.cos(lat_rad))) + \
           6378.137 * (0.5 * (1 - np.cos(lat_rad)))

# Haversine formula to calculate the great-circle distance
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points 
    on the Earth using the Haversine formula.

    Args:
        lat1, lon1 (float): Latitude and Longitude of the first point in degrees.
        lat2, lon2 (float): Latitude and Longitude of the second point in degrees.

    Returns:
        float: The distance between the two points in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Get Earth's radius based on average latitude of the two points
    avg_lat = np.degrees((lat1 + lat2) / 2)
    r = radius_at_latitude(avg_lat)
    
    # Distance in kilometers
    return c * r

def direction_vector(lat1, lon1, lat2, lon2):
    """
    Calculates a unit vector in direction of a trajectory

    Args:
        lat1, lon1 (float): Latitude and Longitude of the first point in degrees.
        lat2, lon2 (float): Latitude and Longitude of the second point in degrees.
    
    Returns:
        np.array: A three dimensional unit vector, representing the direction along 
              a trajectory between two points.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Convert to Cartesian coordinates
    x1, y1, z1 = np.cos(lat1) * np.cos(lon1), np.cos(lat1) * np.sin(lon1), np.sin(lat1)
    x2, y2, z2 = np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)
    
    # Calculate the difference
    diff = np.array([x2 - x1, y2 - y1, z2 - z1])

    # Normalize to get the direction vector
    direction = diff / np.linalg.norm(diff)  # This gives a unit vector
    
    return direction

# Rotation matrix based on latitude
def rotation_matrix(latitude):
    """
    Calculates the Earth's rotation matrix at a given latitude.
    
    Args:
        latitude (float): Latitude in degrees.

    Returns:
        np.array: A 3x3 rotation matrix with elements in [1/s], representing the 
                  Earth's angular velocity at the given latitude.
    """
    # Convert latitude to radians
    lat_rad = np.radians(latitude)
    
    # Earth's angular velocity in rad/s
    omega = 7.2921159e-5  # rad/s

    # Define the rotation matrix (units of [1/s] for non-zero elements)
    return np.array([[0, -omega * np.cos(lat_rad), omega * np.sin(lat_rad)],
                     [omega * np.cos(lat_rad), 0, 0],
                     [-omega * np.sin(lat_rad), 0, 0]])   

# Coriolis acceleration calculation
def coriolis_acc(lat1, lat2, scaled_direction, time, num_steps):
    """
    Calculates the Coriolis acceleration over time as an object moves from one latitude to another.

    Args:
        lat1, lat2 (float): Latitudes in degrees between which the object moves.
        scaled_direction (np.array): Direction vector scaled by velocity in km/s.
        time (float): Total time of movement (units can be arbitrary but consistent).
        num_steps (int): Number of time steps to integrate over.

    Returns:
        np.array: An array of vectors representing Coriolis accelerations at each time step (in km/s²).
    """
    coriolis_accelerations = []
    
    for t in np.linspace(0, time, num_steps):
        # Interpolate latitude at current time step
        current_latitude = lat1 + (t / time) * (lat2 - lat1)
        
        # Get the rotation matrix at the current latitude
        R = rotation_matrix(current_latitude)
        
        # Calculate Coriolis acceleration (resulting in km/s²)
        coriolis_acceleration = R @ scaled_direction
        
        coriolis_accelerations.append(coriolis_acceleration)
    
    return np.array(coriolis_accelerations)


# Coriolis integral for velocity and drift distance
def coriolis_integral(coriolis_acceleration, time, num_steps):
    """
    Integrates Coriolis acceleration to obtain velocity and drift distance over time.

    Args:
        coriolis_acceleration (np.array): An array of Coriolis accelerations (in km/s²).
        time (float): Total time for integration.
        num_steps (int): Number of time steps for integration.

    Returns:
        tuple: 
            - np.array: Coriolis velocity (in km/s).
            - np.array: Coriolis drift distance (in km).
    """
    time_steps = np.linspace(0, time, num_steps)
    coriolis_velocity = np.zeros((num_steps, 3))  # Assuming a 3D vector
    coriolis_drift_distance = np.zeros((num_steps, 3))  # Same shape for distance

    # Integrate Coriolis acceleration to get velocity
    for i in range(1, num_steps):
        coriolis_velocity[i] = coriolis_velocity[i - 1] \
            + 0.5 * (coriolis_acceleration[i] + coriolis_acceleration[i - 1]) \
            * (time_steps[i] - time_steps[i - 1])

    # Integrate velocity to get drift distance
    for i in range(1, num_steps):
        coriolis_drift_distance[i] = coriolis_drift_distance[i - 1] \
            + 0.5 * (coriolis_velocity[i] + coriolis_velocity[i - 1]) \
            * (time_steps[i] - time_steps[i - 1])

    return coriolis_velocity, coriolis_drift_distance

# Calculate total drift based on input DataFrame row
def calculate_total_drift(row, airtime, num_steps=100):
    """
    Calculates the total drift distance due to Coriolis effect for a given row, 
    using code written in python.

    Args:
        row (pd.Series): A row from the DataFrame containing required data:
                         - 'haversine_distance': Distance in kilometers.
                         - 'x_direction', 'y_direction', 'z_direction': Direction components.
                         - 'LATITUDE_ORIGIN', 'LATITUDE_DEST': Latitude values.
        airtime (float): Total time over which the drift occurs (in seconds).
        num_steps (int): Number of steps for integration (default is 100).

    Returns:
        float: Total drift distance magnitude at the last time step (in kilometers).
    """
    
    # Step 2: Calculate the average velocity
    average_velocity = row["haversine_distance"] / airtime
    
    # Step 3: Scale the direction components by the average velocity
    scaled_direction = np.array([
        row["x_direction"] * average_velocity,
        row["y_direction"] * average_velocity,
        row["z_direction"] * average_velocity
    ])
    
    # Step 4: Calculate Coriolis acceleration at each step
    coriolis_accelerations = coriolis_acc(
        row["LATITUDE_ORIGIN"], row["LATITUDE_DEST"], 
        scaled_direction, airtime, num_steps
    )
    
    # Step 5: Integrate to get drift distance
    _, coriolis_drift_distance = coriolis_integral(coriolis_accelerations, airtime, num_steps)
    
    # Step 6: Return the final drift distance at the last time step
    return np.linalg.norm(coriolis_drift_distance[-1])  # Total drift distance magnitude

if __name__ == "__main__":
    main()