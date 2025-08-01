import torch
import time
import numpy as np
 
# Cuda check
if not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU")
    device = torch.device("cpu")
else:
    print("CUDA is available. Running on GPU")
 
# Get the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")
 
# Set the matrix size
matrix_size = 5000
 
def create_and_multiply_matrices(device):
    matrix1 = torch.rand(matrix_size, matrix_size, device=device)
    matrix2 = torch.rand(matrix_size, matrix_size, device=device)
 
    # Perform matrix multiplication
    start_time = time.time()
    result = torch.matmul(matrix1, matrix2)
    torch.cuda.synchronize(device)
    end_time = time.time()
 
    elapsed_time = end_time - start_time
    return elapsed_time
 
try:
    while True:
        if num_gpus > 0:
            for i in range(num_gpus):
                device = torch.device(f"cuda:{i}")
                elapsed_time = create_and_multiply_matrices(device)
        else:
            device = torch.device("cpu")
            elapsed_time = create_and_multiply_matrices(device)
        time.sleep(0.5)
 
except KeyboardInterrupt:
    print("Script terminated by user")