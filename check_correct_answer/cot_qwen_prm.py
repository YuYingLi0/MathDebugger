import re
import json
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
import torch

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run error classification with a model and type")
parser.add_argument('--model', required=True, help='Model name')
parser.add_argument('--dataset', required=True, help='Problem type')

args = parser.parse_args()

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

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
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(f'huggingface_models/{args.model}', trust_remote_code=True)
model = AutoModel.from_pretrained(
    f'huggingface_models/{args.model}',
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

system_prompt = 'You are an intelligent chatbot designed for evaluating math questions and answers.'

for entry in tqdm(data[processed_entries:], total=total_entries, initial=processed_entries, desc="Processing", unit="entry"):
    question = entry.get("question", "Unknown question")
    answer = entry.get("answer", "Unknown answer")

    formatted_prompt = f'''
    I want you to evaluate the following math question and answer:\n
    Question: {question}\n
    '''

    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )

    input_ids = tokenizer.encode(
        conversation_str, 
        return_tensors="pt", 
    ).to(model.device)

    outputs = model(input_ids=input_ids)
    print("outputs")

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    print(step_reward)

    existing_data.append({"id": next_id, "answer": answer, "score": step_reward[0][0]})
    next_id = next_id + 1


    with open(output_path, "w", encoding="utf-8") as output_file:      
        # 写入 JSON 文件
        json.dump(existing_data, output_file, ensure_ascii=False, indent=4)