import torch
import os
import time
import argparse

def allocate_gpu_memory(size_gb, gpus):
    # 检查系统中的可用GPU数量
    available_gpus = torch.cuda.device_count()
    for gpu in gpus:
        if gpu >= available_gpus:
            raise ValueError(f"Invalid GPU device ordinal: {gpu}. Available GPUs: 0 to {available_gpus - 1}")

    # 设置可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    
    # 每个 float32 数字占用 4 字节 (32 bits)
    bytes_per_gb = 1024 ** 3  # 1GB 的字节数
    num_elements = (size_gb * bytes_per_gb) // 4  # 计算需要多少个 float32 元素来占用 size_gb GB 的内存

    # 为每个指定的 GPU 分配内存
    tensors = []
    for gpu in gpus:
        device = torch.device(f'cuda:{gpu}')
        dummy_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        tensors.append(dummy_tensor)
        print(f"Allocated a tensor of size {dummy_tensor.numel() * 4 / bytes_per_gb:.2f} GB on GPU {gpu}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allocate GPU memory.')
    parser.add_argument('--size_gb', type=int, default=10, help='The size of GPU memory to allocate in GB')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='The list of GPUs to allocate memory on')
    args = parser.parse_args()

    allocate_gpu_memory(args.size_gb, args.gpus)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allocate GPU memory.')
    parser.add_argument('--size_gb', type=int, default=10, help='The size of GPU memory to allocate in GB')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='The list of GPUs to allocate memory on')
    args = parser.parse_args()

    allocate_gpu_memory(args.size_gb, args.gpus)

    while True:
        time.sleep(60)  # 每60秒休眠一次，保持程序运行