import os
n = 1
command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.5 --beta 0.5 --max_data_size 100000 --topk 0.5"

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.0 --beta 0.0 --max_data_size 100000 --topk 0.5"
    os.system(command)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.5 --beta 0.0 --max_data_size 100000 --topk 0.5"
    os.system(command)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.0 --beta 0.5 --max_data_size 100000 --topk 0.5"
    os.system(command)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.5 --beta 0.5 --max_data_size 100000 --topk 0.5"
    os.system(command)
