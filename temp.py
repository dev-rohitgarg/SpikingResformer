import torch

def check_cuda_and_gpu():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
        print("No GPU detected.")

check_cuda_and_gpu()
