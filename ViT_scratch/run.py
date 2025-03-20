import os
n = 5
command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 4 --epochs 30 --alpha 0.5 --beta 0.5 --max_data_size 100000 --topk 0.5"

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 8 --epochs 30 --alpha 0.0 --beta 0.0 --max_data_size 100000 --topk 0.5"
    print(command)
    os.system(command)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 8 --epochs 30 --alpha 0.5 --beta 0.0 --max_data_size 100000 --topk 0.5"
    print(command)
    os.system(command)
    if i == 1:
        print(aaa)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 8 --epochs 30 --alpha 0.0 --beta 0.5 --max_data_size 100000 --topk 0.5"
    print(command)
    os.system(command)

for i in range(n):
    print(f"Running iteration {i+1}/10...")
    command = "python train.py --method 2 --dataset rgbd-dataset-10k --batch_size 8 --epochs 30 --alpha 0.5 --beta 0.5 --max_data_size 100000 --topk 0.5"
    print(command)
    os.system(command)
