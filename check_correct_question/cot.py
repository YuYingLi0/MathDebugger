import re
import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Run error classification with a model and type")
parser.add_argument('--model', required=True, help='Model name')
parser.add_argument('--dataset', required=True, help='Problem type')

args = parser.parse_args()

# 获取命令行传入的模型名称和问题类型
model = f"/global_data/sft_intern/lh/huggingface_models/{args.model}"
tokenizer = AutoTokenizer.from_pretrained(model)
# 生成prompts的路径
with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
prompts = []

system_prompt = 'You are an intelligent chatbot designed for evaluating math questions.'

for entry in data:
    question = entry.get("output", "Unknown question")

    formatted_prompt = f'''
    I want you to evaluate the following math question:\n
    Question: {question}\n
    Please determine if the question is correct or not. Answer Step By Step, and finally answer 'Correct' or 'Incorrect'.\n
    '''
    message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": formatted_prompt}]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    prompts.append(text)

# 设置输出文件路径
output_path = f"output/cot_{args.model}_{args.dataset}.json"

# 设置生成参数
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4000)
llm = LLM(model=model, tensor_parallel_size=4)
outputs = llm.generate(prompts, sampling_params)

# 将结果写入文件
results = []
with open(output_path, "w", encoding="utf-8") as output_file:
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        #print(f"Generated Text {i+1}: {generated_text}")

        last_incorrect_pos = generated_text.rfind("Incorrect")
        last_correct_pos = generated_text.rfind("Correct")

        # 判断是正确还是错误
        result = "correct" if last_incorrect_pos < last_correct_pos else "incorrect"

        # 存储结果
        results.append({"generated_text": generated_text, "result": result})
    
    # 写入 JSON 文件
    json.dump(results, output_file, ensure_ascii=False, indent=4)
