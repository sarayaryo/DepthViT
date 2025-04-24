import shutil
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import glob
import random

dataset_path = f"../data/rgbd-dataset/"
import os

cwd = os.getcwd()
print(f"📂 現在のカレントディレクトリ: {cwd}")

def getlabels(image_files):
    """ラベルを取得し、エンコードする"""
    le = LabelEncoder()
    global label_mapping
    labels = []
    
    for image_file in image_files:
        filename = os.path.splitext(os.path.basename(image_file))[0]
        classlabel = filename.split("_")[0]  # クラス名取得
        labels.append(classlabel)

    if not labels:
        print("⚠️ ラベルが取得できていません")
        return [], []

    encoded_labels = le.fit_transform(labels)
    label_mapping = {index: label for index, label in enumerate(le.classes_)}
    
    return encoded_labels, labels  # エンコードと元ラベル両方返す

# 画像ファイルのパス取得
image_files = glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
print(f"🔍 取得した画像ファイル数: {len(image_files)}")

# RGB画像と深度画像を分類
image_paths, depth_paths = [], []
for file_path in image_files:
    filename = os.path.basename(file_path)
    if "depth" in filename:
        depth_paths.append(file_path)
    elif "maskcrop" not in filename:
        image_paths.append(file_path)

# 画像データの確認
print(f"📸 RGB画像数: {len(image_paths)}, 深度画像数: {len(depth_paths)}")
assert len(image_paths) == len(depth_paths), "🚨 Image and depth paths must have the same length!"

# ラベル取得
encoded_labels, original_labels = getlabels(image_paths)
print(f"🏷️ 取得したラベル数: {len(set(original_labels))}, サンプル数: {len(original_labels)}")

# クラスごとにデータを分割
label_dict = {}
for img, depth, label in zip(image_paths, depth_paths, original_labels):
    if label not in label_dict:
        label_dict[label] = []
    label_dict[label].append((img, depth))

# クラスごとのデータ数を確認
print(f"📊 クラス数: {len(label_dict)}")
for label, files in label_dict.items():
    print(f"  - {label}: {len(files)} 件")

# 各クラスごとにバランスをとってサンプリング（合計1万件）
target_size = 10000
num_labels = len(label_dict)  # クラス数

if num_labels == 0:
    raise ValueError("🚨 クラス数が0です。画像のパスやラベル取得を再確認してください！")

samples_per_label = target_size // num_labels  # 1クラスあたりのサンプル数
print(f"📏 各クラスあたりの取得数: {samples_per_label}")

selected_images, selected_depths = [], []
for label, files in label_dict.items():
    random.shuffle(files)
    sampled_files = files[:samples_per_label]  # 各クラスから均等にサンプル
    if sampled_files:
        img_paths, depth_paths = zip(*sampled_files)
        selected_images.extend(img_paths)
        selected_depths.extend(depth_paths)

# シャッフル
paired_data = list(zip(selected_images, selected_depths))
random.shuffle(paired_data)
selected_images, selected_depths = zip(*paired_data)
selected_images, selected_depths = list(selected_images), list(selected_depths)

# 訓練・検証・テストデータに分割（80%:10%:10%）
image_train, image_temp, depth_train, depth_temp = train_test_split(selected_images, selected_depths, test_size=0.2, random_state=42)
image_valid, image_test, depth_valid, depth_test = train_test_split(image_temp, depth_temp, test_size=0.5, random_state=42)

# ラベル取得
train_labels = getlabels(image_train)[0]
valid_labels = getlabels(image_valid)[0]
test_labels = getlabels(image_test)[0]

print(f"✔️ 分割結果 -> Train: {len(image_train)}, Valid: {len(image_valid)}, Test: {len(image_test)}")

# 保存先ディレクトリ
save_dir = "../data/rgbd-dataset-10k/"
os.makedirs(save_dir, exist_ok=True)

# 各フォルダを作成
for split in ["train", "valid", "test"]:
    os.makedirs(os.path.join(save_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, split, "depth"), exist_ok=True)

# データコピー関数
def copy_files(image_list, depth_list, split):
    for img_path, depth_path in zip(image_list, depth_list):
        try:
            shutil.copy(img_path, os.path.join(save_dir, split, "images", os.path.basename(img_path)))
            shutil.copy(depth_path, os.path.join(save_dir, split, "depth", os.path.basename(depth_path)))
        except Exception as e:
            print(f"⚠️ コピーエラー: {img_path}, {depth_path} -> {e}")

# データをコピー
copy_files(image_train, depth_train, "train")
copy_files(image_valid, depth_valid, "valid")
copy_files(image_test, depth_test, "test")

print(f"✅ データを {save_dir} に保存完了！")
