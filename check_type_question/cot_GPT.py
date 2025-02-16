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
You are an intelligent chatbot designed for evaluating math questions.
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
    retry = 0
    while retry<5 :
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
            retry=retry+1
            pass
    if retry >= 5:
        break