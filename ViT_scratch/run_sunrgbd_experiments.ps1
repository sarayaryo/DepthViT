

Write-Host "=== SUN RGB-D 19-cat Experiments (3 conditions, 100 epochs, lr 5e-3->1e-3 at ep10, dropout=0.2) ===" -ForegroundColor Cyan
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# 1/3: Late-Fusion
Write-Host "[1/3] Late-Fusion lr=5e-3->1e-3 step" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --use_19 --exp_name SUNRGBD_19cat_latefusion_lr5e-3_step_do0.2_ep100 --alpha 0.0 --beta 0.0 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 100 --lr 5e-3 --max_data_size 20000000 --scheduler step
Write-Host "[1/3] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

# 2/3: Share-Fusion (alpha/beta fixed)
Write-Host "[2/3] Share-Fusion a=0.5 b=0.5 fixed lr=5e-3->1e-3 step" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --use_19 --exp_name SUNRGBD_19cat_sharefusion_a0.5_b0.5_lr5e-3_step_do0.2_ep100 --alpha 0.5 --beta 0.5 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 100 --lr 5e-3 --max_data_size 20000000 --scheduler step
Write-Host "[2/3] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

# 3/3: Share-Fusion (alpha/beta learnable)
Write-Host "[3/3] Share-Fusion a=0.5 b=0.5 learnable lr=5e-3->1e-3 step" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --use_19 --exp_name SUNRGBD_19cat_sharefusion_a0.5_b0.5_learnable_lr5e-3_step_do0.2_ep100 --alpha 0.5 --beta 0.5 --learnable_alpha_beta --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 100 --lr 5e-3 --max_data_size 20000000 --scheduler step
Write-Host "[3/3] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

Write-Host "=== All experiments finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Green
