import re
import json
from transformers import AutoTokenizer
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards
import argparse
import os
from tqdm import tqdm
import torch

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run error classification with a model and type")
parser.add_argument('--model', required=True, help='Model name')
parser.add_argument('--dataset', required=True, help='Problem type')

args = parser.parse_args()

device = torch.device("cuda")
print(device)

# 设置输出文件路径
output_path = f"/global_data/sft_intern/lh/lyy/check_correct_answer/output/cot_prm_{args.model}_{args.dataset}.json"

# Load the existing JSON file if it exists
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
        next_id = len(existing_data) + 1  # Continue from the last ID
else:
    existing_data = []
    next_id = 1  # Start from ID 1

# 生成prompts的路径
with open(f'/global_data/sft_intern/lh/lyy/check_correct_answer/data/{args.dataset}.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

total_entries = len(data)
processed_entries = next_id - 1

prm_model_path = f'/global_data/sft_intern/lh/huggingface_models/{args.model}'
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(f'/global_data/sft_intern/lh/huggingface_models/{args.model}', trust_remote_code=True)
model = PRM_MODEL.from_pretrained(prm_model_path, device_map="auto").eval()

# model = f"/global_data/sft_intern/lh/huggingface_models/{args.model}"
# tokenizer = AutoTokenizer.from_pretrained(model)

system_prompt = 'You are an intelligent chatbot designed for evaluating math questions and answers.'

for entry in tqdm(data[processed_entries:], total=total_entries, initial=processed_entries, desc="Processing", unit="entry"):
    question = entry.get("question", "Unknown question")
    answer = entry.get("answer", "Unknown answer")

    formatted_prompt = f'''
    I want you to evaluate the following math question and answer:\n
    Question: {question}\n
    '''

    processed_data = [prepare_input(question, answer, tokenizer=tokenizer, step_token="<extra_0>")]
    input_ids, steps, reward_flags = zip(*processed_data)

    input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, tokenizer.pad_token_id)

    print("input_ids device:", input_ids.device)
    print("attention_mask device:", attention_mask.device)
    print("model device:", next(model.parameters()).device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    _, _, rewards = model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
    step_rewards = derive_step_rewards(rewards, reward_flags)
    print("step_rewards:",step_rewards)
    print("step_rewards:",step_rewards[0])

    existing_data.append({"id": next_id, "answer": answer, "score": step_rewards[0][0]})
    next_id = next_id + 1

    with open(output_path, "w", encoding="utf-8") as output_file:      
        # 写入 JSON 文件
        json.dump(existing_data, output_file, ensure_ascii=False, indent=4)
