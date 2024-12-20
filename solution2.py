import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Constants
g = np.array([0, -9.8])  # Gravitational acceleration (m/s^2)
R = 0.00132  # Drag coefficient (1/m)
v0 = 100  # Initial velocity (m/s)
angle = np.pi / 4  # Launch angle (radians)
x0 = np.array([0, 0])  # Initial position
v0_vector = v0 * np.array([np.cos(angle), np.sin(angle)])  # Initial velocity vector
t_final = 10  # Total simulation time
m = 20  # Number of intervals

def drag_force(v):
    return -R * np.linalg.norm(v) * v

def acceleration(v):
    return drag_force(v) + g

def euler_method(dt):
    t_values = np.arange(0, t_final + dt, dt)
    x = np.zeros((len(t_values), 2))
    v = np.zeros((len(t_values), 2))

    x[0] = x0
    v[0] = v0_vector

    for i in range(1, len(t_values)):
        v[i] = v[i-1] + dt * acceleration(v[i-1])
        x[i] = x[i-1] + dt * v[i-1]

    return t_values, x



#had to do it like this because I was having trouble when using an approach that used the drag force and acceleration functions
def f(t, x, v):
    v_magnitude = np.linalg.norm(v)
    dxdt = v
    dvdt = -R * v_magnitude * v + g
    return dxdt, dvdt

# RK4 Step Function
def rk4_step(t, x, v, h):
    k1_dx, k1_dv = f(t, x, v)
    k2_dx, k2_dv = f(t + h / 2, x + h / 2 * k1_dx, v + h / 2 * k1_dv)
    k3_dx, k3_dv = f(t + h / 2, x + h / 2 * k2_dx, v + h / 2 * k2_dv)
    k4_dx, k4_dv = f(t + h, x + h * k3_dx, v + h * k3_dv)

    x_next = x + h / 6 * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
    v_next = v + h / 6 * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv)
    return x_next, v_next

# RK4 Full Method
def rk4_method(dt):
    t_values = np.arange(0, t_final + dt, dt)
    x = [x0]
    v = [v0_vector]

    for t in t_values[:-1]:
        x_next, v_next = rk4_step(t, x[-1], v[-1], dt)
        x.append(x_next)
        v.append(v_next)

    return t_values, np.array(x)

# Error estimation
def estimate_error(x_m, x_2m):
    error = np.linalg.norm(x_m - x_2m)
    relative_error = error / np.linalg.norm(x_2m)
    return error, relative_error


# does all the calculations depending on the number of steps for both methods
def calculateBoth(m):
    # Solve the problem
    dt = t_final / m

    # Euler Method
    t_euler, x_euler = euler_method(dt)

    # RK4 Method
    t_rk4, x_rk4 = rk4_method(dt)

    # Change the number of steps
    m2 = 2 * m
    dt2 = t_final / m2

    # Recalculate with doubled steps
    t_euler_2m, x_euler_2m = euler_method(dt2)
    t_rk4_2m, x_rk4_2m = rk4_method(dt2)

    # Estimate errors for Euler and RK4 methods
    error_euler, rel_error_euler = estimate_error(x_euler[-1], x_euler_2m[-1])
    error_rk4, rel_error_rk4 = estimate_error(x_rk4[-1], x_rk4_2m[-1])

    return t_euler, x_euler, t_rk4, x_rk4, t_euler_2m, x_euler_2m, t_rk4_2m, x_rk4_2m, error_euler, rel_error_euler, error_rk4, rel_error_rk4

# Plot convergence
def plot_convergence(initial_steps, cap_steps):
    steps = np.arange(initial_steps, cap_steps, 10)
    evaluations_euler = np.arange(initial_steps + 1, cap_steps + 1, 10)
    evaluations_rk4 = np.arange(4 * (initial_steps + 1), 4 * (cap_steps + 1), 40)
    
    errors_euler = [estimate_error(euler_method(t_final / s)[1][-1], euler_method(t_final / (2 * s))[1][-1])[0] for s in steps]
    errors_rk4 = [estimate_error(rk4_method(t_final / s)[1][-1], rk4_method(t_final / (2 * s))[1][-1])[0] for s in steps]

    plt.figure(figsize=(10, 6))
    plt.loglog(evaluations_euler, errors_euler, label="Euler Method", marker="o")
    plt.loglog(evaluations_rk4, errors_rk4, label="RK4 Method", marker="x")
    plt.title("Convergence Plot")
    plt.xlabel("log(Number of Function Evaluations)")
    plt.ylabel("log(Error)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()


def system(t, y):
    x, y_pos, vx, vy = y
    v = np.array([vx, vy])
    ax, ay = acceleration(v)
    return [vx, vy, ax, ay]

# Solve the problem using solve_ivp
def solve_ivp_method(maxError):
    initial_conditions = [0, 0, v0_vector[0], v0_vector[1]]
    sol = solve_ivp(system, [0, t_final], initial_conditions, method='RK45', t_eval=np.linspace(0, t_final, 100), rtol=maxError, atol=maxError)
    return sol.t, sol.y, sol.nfev


# Event function to detect when projectile hits the ground
def hit_ground(t, y):
    return y[1]  # y-position
hit_ground.terminal = True
hit_ground.direction = -1

# Solve the problem using solve_ivp
def solve_ivp_with_event(theta):
    v0_vector = v0 * np.array([np.cos(theta), np.sin(theta)])
    initial_conditions = [0, 0, v0_vector[0], v0_vector[1]]
    sol = solve_ivp(system, [0, 100], initial_conditions, method='RK45', events=hit_ground, rtol=1e-12, atol=1e-12)
    return sol.t_events[0][0], sol.y_events[0][0, 0]


# Function to calculate horizontal distance for a given theta
def calculate_distance(theta):
    _, distance_ground = solve_ivp_with_event(theta)
    return distance_ground

# Find the optimal theta to hit a target distance
def find_optimal_theta(target_distance):
    def objective(theta):
        return abs(calculate_distance(theta) - target_distance)

    result = minimize_scalar(objective, bounds=(0, np.pi/2), method='bounded')
    optimal_theta = result.x
    optimal_distance = calculate_distance(optimal_theta)
    return optimal_theta, optimal_distance

t_euler, x_euler, t_rk4, x_rk4, t_euler_2m, x_euler_2m, t_rk4_2m, x_rk4_2m, error_euler, rel_error_euler, error_rk4, rel_error_rk4 = calculateBoth(m)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_euler[:, 0], x_euler[:, 1], label="Euler Method", marker="o")
plt.plot(x_rk4[:, 0], x_rk4[:, 1], label="RK4 Method", marker="x")
plt.title("Projectile Motion Trajectory")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Vertical Distance (m)")
plt.legend()
plt.grid()
plt.show()

# Position at t = 10 seconds for both methods
print("--------- EX 1 --------")
print("Euler Method position at t=10 seconds:", x_euler[-1])
print("RK4 Method position at t=10 seconds:", x_rk4[-1])
print()

# Print results
print("--------- EX 2 --------")
print("Error estimation for m = 20:")
print(f"Euler Method: Absolute Error = {error_euler:.6f}, Relative Error = {rel_error_euler:.6f}")
print(f"RK4 Method: Absolute Error = {error_rk4:.6f}, Relative Error = {rel_error_rk4:.6f}")
print()
print("Error estimation for m = 200:")

m = 200
t_euler, x_euler, t_rk4, x_rk4, t_euler_2m, x_euler_2m, t_rk4_2m, x_rk4_2m, error_euler, rel_error_euler, error_rk4, rel_error_rk4 = calculateBoth(m)

print(f"Euler Method: Absolute Error = {error_euler:.12f}, Relative Error = {rel_error_euler:.12f}")
print(f"RK4 Method: Absolute Error = {error_rk4:.12f}, Relative Error = {rel_error_rk4:.12f}")
print()


plot_convergence(20, 200)


print("--------- EX 4 --------")
# Solve using solve_ivp
t_ivp, y_ivp, nfev_ivp = solve_ivp_method(1e-12)
print("Euler Method position at t=10 seconds:", x_euler[-1])
print("RK4 Method position at t=10 seconds:", x_rk4[-1])
print("solve_ivp (RK45) position at t=10 seconds:", [y_ivp[0][-1], y_ivp[1][-1]])
print("Number of evaluations of the EDO to get an error lower than 1e-12:", nfev_ivp)

t_ivp, y_ivp, nfev_ivp = solve_ivp_method(error_euler)
print("Number of evaluations of the EDO to get a solution as good as Euler with m = 200:", nfev_ivp)

t_ivp, y_ivp, nfev_ivp = solve_ivp_method(error_rk4)
print("Number of evaluations of the EDO to get a solution as good as RK4 with m = 200:", nfev_ivp)


print("--------- EX 5 --------")

# calculated with the initial conditions
t_ground, distance_ground = solve_ivp_with_event(angle)

print(f"Time when projectile hits the ground (solve_ivp): {t_ground:.6f} seconds")
print(f"Horizontal distance when projectile hits the ground (solve_ivp): {distance_ground:.6f} meters")

print("--------- EX 6 --------")
target_distance = 500
optimal_theta, optimal_distance = find_optimal_theta(target_distance)

# Print the results
print(f"Optimal Launch Angle: {np.degrees(optimal_theta):.6f} degrees")
print(f"Distance Achieved at Optimal Angle: {optimal_distance:.6f} meters")

# Test for d(theta) using different angles
theta_values = np.linspace(0, np.pi/2, 50)
distances = [calculate_distance(theta) for theta in theta_values]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta_values), distances, label="Distance vs Angle")
plt.axvline(np.degrees(optimal_theta), color='r', linestyle='--', label=f"Optimal Angle ({np.degrees(optimal_theta):.2f}°)")
plt.axhline(target_distance, color='g', linestyle='--', label=f"Target Distance ({target_distance} m)")
plt.title("Horizontal Distance as a Function of Launch Angle")
plt.xlabel("Launch Angle (degrees)")
plt.ylabel("Horizontal Distance (m)")
plt.legend()
plt.grid()
plt.show()
