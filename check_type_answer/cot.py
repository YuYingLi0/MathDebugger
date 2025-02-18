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
model = f"huggingface_models/{args.model}"
tokenizer = AutoTokenizer.from_pretrained(model)
# 生成prompts的路径
with open(f'data/{args.dataset}.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
prompts = []

system_prompt = '''
You are an intelligent chatbot designed for evaluating math questions and answers.
'''
for entry in data:
    question = entry.get("question", "Unknown question")
    answer = entry.get("answer", "Unknown answer")

    formatted_prompt = f'''
    I want you to evaluate the following math question and answer:\n
    Question: {question}\n
    Answer: {answer}\n
    you need to determine the type of error in the answer. The answer is incorrect, and your task is to classify the type of error. 

    Return one of the following error types:
    "logic error": Errors in problem-solving methods and mathematical logic, including failure to correctly understand the problem, incorrect logic, or steps that are not feasible (e.g., 0.5 people), etc.
    "computing error": Calculation errors, including mistakes in program steps, numerical approximation issues, etc.
    "expression error": Language expression errors, including unclear expression, grammatical issues, too little or excessive response, adding extra conditions, etc.
    
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

        last_mathematical_method_pos = generated_text.rfind("logic error")
        last_computing_problem_pos = generated_text.rfind("computing error")
        last_ability_of_expression_pos = generated_text.rfind("expression error")

        # 如果都不存在
        if last_mathematical_method_pos == -1 and last_computing_problem_pos == -1 and last_ability_of_expression_pos == -1:
            result = "Other"
        else:
            # 取最后出现的那个关键字
            last_position = max(last_mathematical_method_pos, last_computing_problem_pos, last_ability_of_expression_pos)

            if last_position == last_mathematical_method_pos:
                result = "logic error"
            elif last_position == last_computing_problem_pos:
                result = "computing error"
            else:
                result = "expression error"

        # 存储结果
        results.append({"generated_text": generated_text, "result": result})
    
    # 写入 JSON 文件
    json.dump(results, output_file, ensure_ascii=False, indent=4)
