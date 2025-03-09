import json
import random

input_file = '../stanford_alpaca/alpaca_data.json'
train_output_file = './data/alpaca_train.json'
test_output_file = './data/alpaca_test.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

subset_size = 1000
subset = random.sample(data, subset_size)

test_ratio = 0.1
test_size = int(subset_size * test_ratio)
train_size = subset_size - test_size

random.shuffle(subset)
train_subset = subset[:train_size]
test_subset = subset[train_size:]

with open(train_output_file, 'w', encoding='utf-8') as f:
    json.dump(train_subset, f, ensure_ascii=False, indent=4)

with open(test_output_file, 'w', encoding='utf-8') as f:
    json.dump(test_subset, f, ensure_ascii=False, indent=4)

print(f"成功生成 {train_size} 条训练数据和 {test_size} 条测试数据，分别保存在 {train_output_file} 和 {test_output_file} 中。")
