import json

# 读取 JSON 文件
with open('/global_data/sft_intern/lh/lyy/check_type_answer/data/math.json', 'r') as f:
    data = json.load(f)

# 遍历数据并替换"type"字段的值
for item in data:
    if item.get("type") == "mathematical method":
        item["type"] = "logic error"
    elif item.get("type") == "computing problem":
        item["type"] = "computing error"
    elif item.get("type") == "ability of expression":
        item["type"] = "expression error"

# 保存修改后的数据到新的 JSON 文件
with open('/global_data/sft_intern/lh/lyy/check_type_answer/data/modified_math.json', 'w') as f:
    json.dump(data, f, indent=4)

print("替换完成，修改后的数据已保存到 'modified_file.json'")
