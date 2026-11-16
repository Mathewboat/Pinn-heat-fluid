def analytical(x, t):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

# Generate 10,000 test points
x_test = np.random.uniform(0, 1, 10000)
t_test = np.random.uniform(0, 1, 10000)
T_true = analytical(x_test, t_test)   # this is your "ground truth data"