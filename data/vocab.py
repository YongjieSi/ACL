import pandas as pd
import json

# 读取 train.csv 文件
train_data = pd.read_csv('/new_data/syj/CD-FCAC/data/s2s/l2n2f_fscil_test.csv')

# 获取所有类别并按顺序排序
categories = sorted(train_data['label'].unique())

for category, count in categories.items():
    print(f'类别 "{category}" 的样本数量: {count}')
# # 为每个类别分配一个标签
# category_to_label = {category: idx for idx, category in enumerate(categories)}

# # 将映射关系保存为 vocab.json 文件
# with open('/data/syj/CIL-RepLearning-main/data/esc/test_vocab.json', 'w') as f:
#     json.dump(category_to_label, f, indent=4)

# print("vocab.json 文件已生成")
