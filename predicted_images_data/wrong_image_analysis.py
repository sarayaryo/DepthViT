import pandas as pd
import os

# モデル1の不正解画像のCSV
df = pd.read_csv(r"onlyRGB\wrong_images.csv")

# カテゴリ抽出
def extract_category(path):
    norm_path = os.path.normpath(path)
    parts = norm_path.split(os.sep)
    category_folder = parts[-2]
    category = category_folder.split("_")[0]
    return category

df["category"] = df["path"].apply(extract_category)

# カテゴリ別件数を集計
category_counts = df["category"].value_counts()

# 結果表示
print("モデル1で不正解だったカテゴリ別件数:")
print(category_counts)

# 必要なら上位Nカテゴリを取得
top_categories = category_counts.head(5)
print("\n上位5カテゴリ:")
print(top_categories)
