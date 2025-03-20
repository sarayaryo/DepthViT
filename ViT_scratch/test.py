
from datasets import load_dataset
from PIL import Image
import numpy as np

dataset = load_dataset("sayakpaul/nyu_depth_v2",split="train[:5%]", trust_remote_code=True)
from datasets import load_dataset

print(dataset.cache_files)

# print(dataset)

# sample = dataset['train'][0]
# print(sample.keys())  # ['image', 'depth_map']


# image = sample['image']
# depth_map = sample['depth_map']

# image.show()  # RGB画像
# Image.fromarray((depth_map / np.max(depth_map) * 255).astype(np.uint8)).show()  # 深度マップ
