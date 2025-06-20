import numpy as np
import matplotlib.pyplot as plt

# field function dy/dx = f(x, y)
def f(x, y):
    return y - x

# plot region
x = np.linspace(-5, 5, 20)
y = np.linspace(-5, 5, 20)

X, Y = np.meshgrid(x, y)

# direction field components
U = 1              # dx component is always 1
V = f(X, Y)        # dy component

# normalizing the vectors for better visualization
N = np.sqrt(U**2 + V**2)
U = U / N
V = V / N

# plotting the direction field
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, U, V, angles="xy")
plt.title(r"Direction Field for $\frac{dy}{dx} = y - x$")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.axis('equal')
plt.show()
