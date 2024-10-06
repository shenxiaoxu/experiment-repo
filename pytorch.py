import torch
import time

# Set the device to GPU
device = torch.device("cuda")

# Create two random matrices A and B of size 1000x1000
A = torch.randn(1000, 1000, device=device)
B = torch.randn(1000, 1000, device=device)

# Perform matrix multiplication with PyTorch on GPU
start_time = time.time()
C = torch.mm(A, B)
end_time = time.time()

print(f"PyTorch matrix multiplication result shape: {C.shape}")
print(f"Time taken by PyTorch on GPU: {end_time - start_time:.6f} seconds")


# Run experiment on p3.8xlarge instance. The time takes
# PyTorch matrix multiplication result shape: torch.Size([1000, 1000])
# Time taken by PyTorch on GPU: 0.330934 seconds
