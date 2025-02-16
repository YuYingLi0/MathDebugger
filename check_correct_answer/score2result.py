import json

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 根据 score 添加 result 字段
def add_result(data):
    for entry in data:
        if 'score' in entry:
            if entry['score'] > 0.5:
                entry['result'] = 'correct'
            else:
                entry['result'] = 'incorrect'
        else:
            entry['result'] = 'unknown'  # 如果 score 字段不存在
    return data

# 保存为新的 JSON 文件
def save_as_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 主函数
def main(input_file, output_file):
    # 读取 JSON 文件
    data = read_json(input_file)
    
    # 根据 score 添加 result 字段
    updated_data = add_result(data)
    
    # 保存为新的 JSON 文件
    save_as_json(updated_data, output_file)
    print(f"数据已处理并保存到 {output_file}")

# 示例调用
if __name__ == "__main__":
    input_file = "check_correct_answer/prm_score/cot_prm_Skywork-PRM-7B_math.json"  # 输入 JSON 文件路径
    output_file = "check_correct_answer/output/cot_prm_Skywork-PRM-7B_math.json"  # 输出 JSON 文件路径
    main(input_file, output_file)