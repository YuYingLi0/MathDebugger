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

system_prompt = '''
You are an intelligent chatbot designed for evaluating math questions.
'''
for entry in data:
    question = entry.get("output", "Unknown question")

    formatted_prompt = f'''
    I want you to evaluate the following math question:\n
    Question: {question}\n
    you need to determine the type of error in the question. The question is incorrect, and your task is to classify the type of error. 

    Return one of the following error types:
    "expression_error": Contains pronouns or expressions that are unclear or ambiguous, leading to multiple possible interpretations of the problem; includes grammatical errors, unclear expressions, or incoherent writing that is difficult to read and understand, or even nonsensical; contains unnecessary or irrelevant conditions; misuses technical terms, jargon, or concepts.
    "lack_of_conditions": Missing necessary charts, diagrams, or illustrations; lacking key conditions required to solve the problem.
    "contradictions": Contains contradictory conditions; the solution to the problem does not exist.
    "unrealistic": The result should be an integer, but is presented as a non-integer, which is not feasible in the real world (e.g., 0.5 people); includes conditions that violate common sense or natural laws, making the problem absurd or meaningless.

    Answer Step By Step, and finally return only one of these error types.
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

        last_expression_error_pos = generated_text.rfind("expression_error")
        last_lack_of_conditions_pos = generated_text.rfind("lack_of_conditions")
        last_contradictions_pos = generated_text.rfind("contradictions")
        last_unrealistic_pos = generated_text.rfind("unrealistic")

        # 如果所有错误类型都没有出现
        if last_expression_error_pos == -1 and last_lack_of_conditions_pos == -1 and last_contradictions_pos == -1 and last_unrealistic_pos == -1:
            result = "Other"
        else:
            # 取最后出现的错误类型
            last_position = max(last_expression_error_pos, last_lack_of_conditions_pos, last_contradictions_pos, last_unrealistic_pos)

            if last_position == last_expression_error_pos:
                result = "expression_error"
            elif last_position == last_lack_of_conditions_pos:
                result = "lack_of_conditions"
            elif last_position == last_contradictions_pos:
                result = "contradictions"
            else:
                result = "unrealistic"

        # 存储结果
        results.append({"generated_text": generated_text, "result": result})
    
    # 写入 JSON 文件
    json.dump(results, output_file, ensure_ascii=False, indent=4)
