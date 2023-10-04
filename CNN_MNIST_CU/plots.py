import matplotlib.pyplot as plt
import numpy as np

input_w = 28
input_h = 28
kernel_size = 3
output_w = input_w - kernel_size + 1
output_h = input_h - kernel_size + 1
output_channels = 16

num_blocks_max = 11

# Sample data (replace with your actual data)
#execution_times = [0.05, 0.02, 0.02, 0.17, 0.16, 0.15, 0.13, 0.13, 0.14, 0.14, 0.15, 0.21, 0.16, 0.16, 0.16, 0.16, 0.18, 0.15, 0.14, 0.18, 0.13, 0.19, 0.13, 0.27, 0.16, 0.19, 0.24, 0.17, 0.19, 0.16, 0.24, 0.17, 0.19, 0.20, 0.16, 0.18, 0.16, 0.16, 0.16, 0.18, 0.17, 0.18, 0.16, 0.17, 0.24, 0.17, 0.22, 0.18, 0.19]
execution_times = [0.02, 0.04, 0.05, 0.17, 0.16, 0.15, 0.14, 0.13, 0.14, 0.14]
                   
num_blocks = np.arange(1, num_blocks_max)  # Convert to NumPy array

# Calculate the number of threads per block
# threads = (output_w + num_blocks - 1) // num_blocks  # Use // for integer division
threads_x = (output_w + num_blocks - 1) // num_blocks
threads_y = (output_h + num_blocks - 1) // num_blocks
threads_z = output_channels
threads = threads_x * threads_y * threads_z

# Create two subplots
plt.figure(figsize=(12, 5))

# Subplot 1: Number of Blocks vs. Execution Time
plt.subplot(1, 2, 1)
plt.plot(num_blocks, execution_times, marker='o')
plt.xlabel("Number of Blocks")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs. Number of Blocks")

# Subplot 2: Number of Blocks vs. Threads per Block
plt.subplot(1, 2, 2)
plt.plot(num_blocks, threads, marker='o', color='r', linestyle='--')
plt.xlabel("Number of Blocks")
plt.ylabel("Threads per Block")
plt.title("Threads per Block vs. Number of Blocks")

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()