# DepthViT

テスト実行
```
$ cd ViT_scratch
$ python train.py --method 2 --dataset rod_sample --batch_size 4 --epochs 3 --max_data_size 10
```

テスト実行/ VSCode Windows11
1. install library
　pip install -r requirements.txt

2. change directry to ViT_scratch
 cd .\ViT_scratch

3. run_trial
3-1. if want to use GPU
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 python train.py --method 2 --dataset rod_sample --batch_size 4 --epochs 3 --max_data_size 10 

3-2. if only CPU
 editting now...

4. run_actual
4-1. if want to use GPU
 pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
 python train.py --method 2 --dataset rgbd-dataset --batch_size 16 --epochs 100 