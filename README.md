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

3.  run_actual

| 3-1. use WRGBD
    python train.py --method 2 --dataset_type 0 --alpha 0.5 --beta 0.5 --dataset rod_sample --batch_size 16 --epochs 30 --max_data_size 20000000
　3-2. use NYUv2
    python train.py --method 2 --dataset_type 1 --alpha 0.5 --beta 0.5 --dataset nyu_data_sample --batch_size 16 --epochs 30 --max_data_size 20000000
  3-3. use TinyImageNet
    python train.py --method 2 --dataset_type 2 --alpha 0.5 --beta 0.5 --dataset rgbd_tinyimagenet --batch_size 16 --epochs 30 --max_data_size 20000000
