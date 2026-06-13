Write-Host "=== SUN RGB-D Experiments (4 conditions) ===" -ForegroundColor Cyan
Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

# 1/4: Late-Fusion lr=1e-3
Write-Host "[1/4] Late-Fusion lr=1e-3" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_latefusion_lr1e-3 --alpha 0.0 --beta 0.0 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3 --max_data_size 20000000
Write-Host "[1/4] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

# 2/4: Late-Fusion lr=1e-4
Write-Host "[2/4] Late-Fusion lr=1e-4" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_latefusion_lr1e-4 --alpha 0.0 --beta 0.0 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-4 --max_data_size 20000000
Write-Host "[2/4] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

# 3/4: Share-Fusion lr=1e-3
Write-Host "[3/4] Share-Fusion lr=1e-3" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.5_b0.5_lr1e-3 --alpha 0.5 --beta 0.5 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3 --max_data_size 20000000
Write-Host "[3/4] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

# 4/4: Share-Fusion lr=1e-4
Write-Host "[4/4] Share-Fusion lr=1e-4" -ForegroundColor Yellow
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.5_b0.5_lr1e-4 --alpha 0.5 --beta 0.5 --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-4 --max_data_size 20000000
Write-Host "[4/4] Done: $(Get-Date -Format 'HH:mm:ss')"
Write-Host ""

Write-Host "=== All experiments finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" -ForegroundColor Green
