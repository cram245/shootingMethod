import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

def rk4_method(dt):
    t_values = np.arange(0, t_final + dt, dt)
    x = np.zeros((len(t_values), 2))
    v = np.zeros((len(t_values), 2))

    x[0] = x0
    v[0] = v0_vector

    for i in range(1, len(t_values)):
        k1_v = acceleration(v[i-1])
        k1_x = v[i-1]

        k2_v = acceleration(v[i-1] + 0.5 * dt * k1_v)
        k2_x = v[i-1] + 0.5 * dt * k1_x

        k3_v = acceleration(v[i-1] + 0.5 * dt * k2_v)
        k3_x = v[i-1] + 0.5 * dt * k2_x

        k4_v = acceleration(v[i-1] + dt * k3_v)
        k4_x = v[i-1] + dt * k3_x

        v[i] = v[i-1] + (dt / 6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        x[i] = x[i-1] + (dt / 6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)

    return t_values, x

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
    steps = [initial_steps, cap_steps]
    evaluations_euler = [initial_steps + 1, cap_steps + 1]
    evaluations_rk4 = [4 * (initial_steps + 1), 4 * (cap_steps + 1)]
    
    errors_euler = [estimate_error(euler_method(t_final / s)[1][-1], euler_method(t_final / (2 * s))[1][-1])[0] for s in steps]
    errors_rk4 = [estimate_error(rk4_method(t_final / s)[1][-1], rk4_method(t_final / (2 * s))[1][-1])[0] for s in steps]

    plt.figure(figsize=(10, 6))
    plt.loglog(evaluations_euler, errors_euler, label="Euler Method", marker="o")
    plt.loglog(evaluations_rk4, errors_rk4, label="RK4 Method", marker="x")
    plt.title("Convergence Plot")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()


def system(t, y):
    x, y_pos, vx, vy = y
    v = np.array([vx, vy])
    ax, ay = acceleration(v)
    return [vx, vy, ax, ay]

# Solve the problem using solve_ivp
def solve_ivp_method():
    initial_conditions = [0, 0, v0_vector[0], v0_vector[1]]
    sol = solve_ivp(system, [0, t_final], initial_conditions, method='RK45', t_eval=np.linspace(0, t_final, 100), rtol=1e-12, atol=1e-12)
    return sol.t, sol.y, sol.nfev

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
t_ivp, y_ivp, nfev_ivp = solve_ivp_method()
print("Euler Method position at t=10 seconds:", x_euler[-1])
print("RK4 Method position at t=10 seconds:", x_rk4[-1])
print("solve_ivp (RK45) position at t=10 seconds:", [y_ivp[0][-1], y_ivp[1][-1]])
print("Number of evaluations of ivp:", nfev_ivp)
