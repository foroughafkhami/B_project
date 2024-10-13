import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from LightHandControl import LightHandControl

# Parameters
num_sensors = 4
num_luminaires = 4
T_s = 1  # Sampling period in seconds
time_steps = 100  # Total simulation time steps
hold_time = 5  # ZOH hold time in steps

# Initial desired illuminance levels
desired_illuminance_occupied = 500  # Default value, will be updated by light control
desired_illuminance_unoccupied = 300

# System matrix G (4x4) - the effect of luminaires on sensors
G = np.array([[450, 60, 30, 25],
              [55, 470, 40, 35],
              [25, 40, 440, 50],
              [30, 35, 55, 480]])

# Daylight contribution vector (4x1)
d = np.array([[280], [140], [80], [40]])  # Daylight contribution for each sensor

# Initialize luminaires (control input)
u = np.zeros((time_steps, num_luminaires))  # Control signals over time

# Initialize estimated illumination values
y_hat = np.zeros((time_steps, num_sensors))

# Function to apply Zero-Order Hold (ZOH)
def apply_zoh(u, last_u, hold_time, k, m):
    if k % hold_time == 0:
        last_u[m] = u[k, m]  # Update the last control value at step k for luminaire m
    else:
        u[k, m] = last_u[m]  # Hold the previous control value for luminaire m
    return u, last_u

# Store the last control values for ZOH (for each luminaire)
last_u = np.zeros(num_luminaires)

# Initialize LightHandControl
light_control = LightHandControl()

# Function to update desired illuminance based on hand gestures
def update_desired_illuminance():
    global desired_illuminance_occupied
    while True:
        light = light_control.get_light()
        if light is not None:
            desired_illuminance_occupied = light
            print(f"Updated desired illuminance: {desired_illuminance_occupied} lux")

# Start the light control thread
light_thread = threading.Thread(target=update_desired_illuminance, daemon=True)
light_thread.start()

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, time_steps)
ax.set_ylim(0, 1000)
lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(num_sensors)]

# Add desired illuminance lines
line_desired_occupied, = ax.plot([], [], 'r--', label='Desired Illuminance (Occupied)')
line_desired_unoccupied, = ax.plot([], [], 'g--', label='Desired Illuminance (Unoccupied)')
ax.set_title('Estimated Light Sensor Readings Over Time with ZOH on Control Signal')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Illuminance (lux)')
ax.legend()
ax.grid()

# Simulation loop
def simulation_step(k):
    global u, last_u
    prev_k = k - 1
    
    # Simulate the contribution of daylight
    daylight_contribution = d.flatten()  # Flatten the matrix to vector

    # Compute the estimated illumination
    for m in range(num_sensors):
        y_hat[k, m] = G[m, :] @ u[prev_k] + daylight_contribution[m]  # Estimate illuminance at sensor m

    # Control strategy (PI controller) and ZOH applied here
    for m in range(num_luminaires):
        error = desired_illuminance_occupied - y_hat[k, m]  # Error for each sensor

        # Update control input using the provided formula
        u[k, m] = (error / G[m, m]) + u[prev_k, m]  # Control input update

        # Ensure control signal is within bounds
        u[k, m] = np.clip(u[k, m], 0, 100)  # Brightness limits

        # Apply ZOH to the control signal after updating it
        u, last_u = apply_zoh(u, last_u, hold_time, k, m)

# Update function for the animation
def update(frame):
    simulation_step(frame)

    for sensor_index, line in enumerate(lines):
        line.set_data(np.arange(frame), y_hat[:frame, sensor_index])

    # Update desired illuminance line
    line_desired_occupied.set_data(np.arange(frame), [desired_illuminance_occupied]*frame)
    line_desired_unoccupied.set_data(np.arange(frame), [desired_illuminance_unoccupied]*frame)

    # Print current desired illuminance for visibility
    print(f"Current desired illuminance: {desired_illuminance_occupied} lux")

    return lines + [line_desired_occupied, line_desired_unoccupied]

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(1, time_steps), blit=True, interval=200)

# Show the plot with animation
plt.show()

# Clean up
del light_control