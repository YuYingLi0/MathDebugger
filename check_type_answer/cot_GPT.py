import csv
import json
import openai
from openai import OpenAI
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run error classification with a model and type")
parser.add_argument('--model', required=True, help='Problem type')
parser.add_argument('--dataset', required=True, help='Problem type')

args = parser.parse_args()

dataset = args.dataset

json_filename = f"data/{args.dataset}.json"
output_filename = f"output/cot_{args.dataset}_GPT_{args.model}.json"

# Load the existing JSON file if it exists
if os.path.exists(output_filename):
    with open(output_filename, 'r', encoding='utf-8') as json_file:
        existing_data = json.load(json_file)
        next_id = len(existing_data) + 1  # Continue from the last ID
else:
    existing_data = []
    next_id = 1  # Start from ID 1

# 系统提示
system_prompt = '''
You are an intelligent chatbot designed for evaluating math questions and answers.
'''
with open(json_filename, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

client = OpenAI(
    api_key="sk-",
    base_url=""
)

# Get total entries and processed count
total_entries = len(data)
processed_entries = next_id - 1

# Process each entry with progress bar
for entry in tqdm(data[processed_entries:], total=total_entries, initial=processed_entries, desc="Processing", unit="entry"):
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
    retry = 0
    while retry < 5 :
        try:
            # Request OpenAI API
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=1,  # Control randomness
                max_tokens=4000  # Max response length
            )

            generated_text = response.choices[0].message.content
            
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

            # Prepare the data to be added
            new_entry = {
                "id": next_id,
                "generated_text": generated_text,
                "result": result
            }

            # Add the new entry to the existing data
            existing_data.append(new_entry)

            # Increment the ID for the next entry
            next_id += 1

            # Save the updated data back to the JSON file immediately after processing the entry
            with open(output_filename, 'w', encoding='utf-8') as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

            print(generated_text)
            print(result)

            break
        except Exception as e:
            print("Error:", e)
            retry = retry + 1
            pass
    if retry >= 5:
        break
