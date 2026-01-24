import numpy as np

# Create test data
signal_size = 10
T = [1, 3, 5, 7]  # Indices where we want to assign values

# Create a source array with some values
source = np.arange(10) * 2  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

print("Original source array:", source)
print("Indices to assign (T):", T)

# Method 1: Using for loop (old way)
b_loop = np.zeros(signal_size)
for index in T:
    b_loop[index] = source[index]
print("\nUsing for loop:")
print("Result:", b_loop)

# Method 2: Vectorized operation (efficient way)
b_vectorized = np.zeros(signal_size)
b_vectorized[T] = source[T]
print("\nUsing vectorized indexing:")
print("Result:", b_vectorized)

# Verify they're the same
print("\nAre results equal?", np.array_equal(b_loop, b_vectorized))
