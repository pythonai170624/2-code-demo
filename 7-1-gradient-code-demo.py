import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set seed for reproducibility
np.random.seed(42)


# Define the linear function: f(x) = mx + b
def linear_function(x, m, b):
    return m * x + b


# Loss function (Mean Squared Error)
def mse_loss(m, b, X, y):
    predictions = linear_function(X, m, b)
    return np.mean((predictions - y) ** 2)


# Compute gradients for the linear function parameters
def compute_gradients(m, b, X, y):
    n = len(X)
    predictions = linear_function(X, m, b)
    errors = predictions - y

    # Gradient for slope (m)
    dm = (2 / n) * np.sum(errors * X)

    # Gradient for intercept (b)
    db = (2 / n) * np.sum(errors)

    return dm, db


# Gradient descent algorithm
def gradient_descent(X, y, learning_rate=0.01, iterations=100):
    # Initialize parameters randomly
    m = np.random.randn()
    b = np.random.randn()

    # Lists to store history for visualization
    m_history = [m]
    b_history = [b]
    loss_history = [mse_loss(m, b, X, y)]

    # Perform gradient descent
    for i in range(iterations):
        # Calculate gradients
        dm, db = compute_gradients(m, b, X, y)

        # Update parameters
        m = m - learning_rate * dm
        b = b - learning_rate * db

        # Store history
        m_history.append(m)
        b_history.append(b)
        loss_history.append(mse_loss(m, b, X, y))

    return m, b, m_history, b_history, loss_history


# Generate synthetic data
def generate_data(n=50, noise=3):
    # True parameters we want to discover
    true_m, true_b = 2.5, -5

    # Generate X values
    X = np.linspace(-5, 5, n)

    # Generate Y values with some noise
    y = true_m * X + true_b + noise * np.random.randn(n)

    return X, y, true_m, true_b


# Generate data
X, y, true_m, true_b = generate_data(n=50)

# Perform gradient descent
m, b, m_history, b_history, loss_history = gradient_descent(
    X, y, learning_rate=0.02, iterations=50)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Data points and fitting line
ax1.scatter(X, y, color='blue', alpha=0.6, label='Data points')

# X values for plotting the line
x_line = np.linspace(-6, 6, 100)

# Initial line (first guess)
initial_line, = ax1.plot(x_line, linear_function(x_line, m_history[0], b_history[0]),
                         'r--', alpha=0.5, label='Initial guess')

# True line (what we're trying to discover)
ax1.plot(x_line, linear_function(x_line, true_m, true_b),
         'g-', label='True line')

# Current line that will be updated in the animation
line, = ax1.plot(x_line, linear_function(x_line, m_history[0], b_history[0]),
                 'r-', linewidth=2, label='Current fit')

ax1.set_xlim(-6, 6)
ax1.set_ylim(min(y) - 5, max(y) + 5)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title('Linear Regression with Gradient Descent', fontsize=14)
ax1.legend()
ax1.grid(True)

# Text to display current iteration and parameters
iteration_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                          bbox=dict(facecolor='white', alpha=0.8))

# Plot 2: Loss history
ax2.plot(loss_history, 'b-', linewidth=2)
ax2.set_xlabel('Iterations', fontsize=12)
ax2.set_ylabel('Mean Squared Error', fontsize=12)
ax2.set_title('Loss over Iterations', fontsize=14)
ax2.grid(True)


# Animation update function
def update(frame):
    # Update the line
    line.set_ydata(linear_function(x_line, m_history[frame], b_history[frame]))

    # Update the text
    iteration_text.set_text(
        f'Iteration: {frame}\nm: {m_history[frame]:.4f}\nb: {b_history[frame]:.4f}\nLoss: {loss_history[frame]:.4f}')

    # Update loss plot - fill area under curve up to current iteration
    ax2.clear()
    ax2.plot(loss_history, 'b-', linewidth=2)
    ax2.fill_between(range(frame + 1), loss_history[:frame + 1], color='blue', alpha=0.2)
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Mean Squared Error', fontsize=12)
    ax2.set_title('Loss over Iterations', fontsize=14)
    ax2.grid(True)

    return line, iteration_text


# Create the animation
anim = FuncAnimation(fig, update, frames=len(m_history), interval=200, blit=False)

plt.tight_layout()
plt.show()

# Print final results
print(f"True parameters: m = {true_m}, b = {true_b}")
print(f"Found parameters: m = {m:.4f}, b = {b:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")

