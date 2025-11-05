import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# CSV読み込み
df = pd.read_csv("fixed_images.csv")

def extract_category(path):
    # 正規化（Windowsパスのバックスラッシュも処理）
    norm_path = os.path.normpath(path)
    # フォルダ部分のリスト
    parts = norm_path.split(os.sep)
    # 末尾: ファイル名
    # -1: フォルダ名（kitchen_xxx_out）
    category_folder = parts[-2]
    # 分割
    category = category_folder.split("_")[0]
    return category

# カテゴリ抽出
df["category"] = df["path"].apply(extract_category)
print(df)

# カテゴリ別件数
category_counts = df["category"].value_counts()
print("カテゴリ別改善件数:")
print(category_counts)

# 件数の多い上位5カテゴリを表示
top_categories = category_counts.head(5).index.tolist()
print("\n上位5カテゴリ:", top_categories)

# 各カテゴリから代表例（ランダムに1枚）を表示
for category in top_categories:
    sample_paths = df[df["category"] == category]["path"].sample(1).values[0]
    print(f"\n代表画像 ({category}): {sample_paths}")

    # # 画像表示
    # img = Image.open(sample_paths)
    # plt.imshow(img)
    # plt.title(f"Category: {category}")
    # plt.axis("off")
    # plt.show()
