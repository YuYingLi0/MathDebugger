import json
from sklearn.metrics import accuracy_score, f1_score
import random

labels_file_path = f"check_correct_question/data/challenging.json"
predictions_file_path = f"check_correct_question/output/cot_math_GPT_deepseek-reasoner.json"
predictions = []

# 标签映射字典
label_mapping = {
    "correct": 1,
    "expression_error": 0,
    "lack_of_conditions": 0,
    "contradictions": 0,
    "unrealistic": 0
}

predict_mapping = {
    "correct": 1,
    "incorrect": 0
}

# 读取预测结果
with open(predictions_file_path, "r", encoding="utf-8") as file:
    predict = json.load(file)

with open(labels_file_path, "r", encoding="utf-8") as file:
    label = json.load(file)

predict_list = []
label_list = []

for i in range(len(predict)):
    predict_list.append(predict_mapping[predict[i]['result']])
    label_list.append(label_mapping[label[i]['type']])

# 确保预测结果和标签的长度一致
assert len(predict_list) == len(label_list), "Predictions and labels length mismatch!"

# 计算总准确率
accuracy = accuracy_score(label_list, predict_list)

# 计算加权平均和宏平均的 F1-Score
f1 = f1_score(label_list, predict_list)  # 加权平均 F1-Score

# 输出结果
print(f"Total Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
# print(f"&{accuracy:.4f}&{f1_weighted:.4f}&{f1_macro:.4f}")
