import re
import json
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
import argparse
import os
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run error classification with a model and type")
parser.add_argument('--model', required=True, help='Model name')
parser.add_argument('--dataset', required=True, help='Problem type')

args = parser.parse_args()

# 设置输出文件路径
output_path = f"output/cot_prm_{args.model}_{args.dataset}.json"

# Load the existing JSON file if it exists
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
        next_id = len(existing_data) + 1  # Continue from the last ID
else:
    existing_data = []
    next_id = 1  # Start from ID 1

# 生成prompts的路径
with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

total_entries = len(data)
processed_entries = next_id - 1

# 获取命令行传入的模型名称和问题类型
good_token = '+'
bad_token = '-'
step_tag = 'ки'

tokenizer = AutoTokenizer.from_pretrained(f'huggingface_models/{args.model}')
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:] # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1] # 12902
model = AutoModelForCausalLM.from_pretrained('huggingface_models/math-shepherd-mistral-7b-prm').eval()

system_prompt = 'You are an intelligent chatbot designed for evaluating math questions and answers.'

for entry in tqdm(data[processed_entries:], total=total_entries, initial=processed_entries, desc="Processing", unit="entry"):
    question = entry.get("question", "Unknown question")
    answer = entry.get("answer", "Unknown answer")

    formatted_prompt = f'''
    Question: {question}\n
    Answer: {answer}\n
    '''
    # message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_prompt + "ки"}]
    # text = tokenizer.apply_chat_template(
    #     message,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # prompts.append(text)
    prompt = formatted_prompt + "ки"

    input_id = torch.tensor([tokenizer.encode(prompt)])

    with torch.no_grad():
        logits = model(input_id).logits[:,:,candidate_tokens]
        scores = logits.softmax(dim=-1)[:,:,0] 
        step_scores = scores[input_id == step_tag_id]
        print(step_scores)

    existing_data.append({"id": next_id, "answer": answer, "score": step_scores.item()})

    next_id = next_id + 1

    with open(output_path, "w", encoding="utf-8") as output_file:      
        # 写入 JSON 文件
        json.dump(existing_data, output_file, ensure_ascii=False, indent=4)
