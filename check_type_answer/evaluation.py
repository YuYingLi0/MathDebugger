import json
from sklearn.metrics import accuracy_score, f1_score
import random

labels_file_path = f"check_type_answer/data/simple.json"
predictions_file_path = f"check_type_answer/output/cot_simple_GPT_deepseek-reasoner.json"
predictions = []

random.seed(42)

# 标签映射字典
label_mapping = {
    "logic error": 0,
    "computing error": 1,
    "expression error": 2,
    "Other": random.choice([0, 1, 2])  # 随机选择一个已有标签
}

# 读取预测结果
with open(predictions_file_path, "r", encoding="utf-8") as file:
    predict = json.load(file)

with open(labels_file_path, "r", encoding="utf-8") as file:
    label = json.load(file)

predict_list = []
label_list = []

for i in range(len(predict)):
    predict_list.append(label_mapping[predict[i]['result']])
    label_list.append(label_mapping[label[i]['type']])

# 确保预测结果和标签的长度一致
assert len(predict_list) == len(label_list), "Predictions and labels length mismatch!"

# 计算总准确率
accuracy = accuracy_score(label_list, predict_list)

f1_macro = f1_score(label_list, predict_list, average='macro')  # 宏平均 F1-Score

# 输出结果
print(f"Total Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
# print(f"&{accuracy:.4f}&{f1_weighted:.4f}&{f1_macro:.4f}")
