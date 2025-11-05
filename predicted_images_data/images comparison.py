import pandas as pd

# モデル1で不正解の画像
wrong_df = pd.read_csv("wrong_images.csv")

# モデル2で正解の画像
correct_df = pd.read_csv("correct_images.csv")

# モデル1で不正解だったもののうち、モデル2で正解したものを抽出
# path列をキーに照合する
fixed_paths = pd.merge(
    wrong_df,
    correct_df,
    on="path",
    how="inner",
    suffixes=("_model1", "_model2")
)

# 結果を確認
print(fixed_paths)

# 必要に応じてパスだけのCSVを作る
fixed_paths[["path"]].to_csv("fixed_images.csv", index=False)
