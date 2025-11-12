import numpy as np

# Path to your .npz file
npz_file = "mnist_test_data.npz"

# Load the .npz file
data = np.load(npz_file)

# Print the arrays and their shapes
for name, array in data.items():
    print(f"{name}: shape={array.shape}, dtype={array.dtype}")