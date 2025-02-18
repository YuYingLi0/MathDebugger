# MathClean
Code for the Paper ["MathClean: A Benchmark for Synthetic Mathematical Data Cleaning"]()

## Outlines
- [News](#News)
- [MathClean](#MathClean)
- [Data Format and Examples](#Data_Format_and_Examples)
  - [Data Format](#Data_Format)
  - [Example](#Example)
- [Usage](#Usage)
  - [Environment](#Environment)
  - [Model Inference](#Model_Inference)
  - [Evaluation](#Evaluation)

## News

## MathClean

<p align="center">
    <img src="fig/cover/cover_page.jpg"> <br>
  overview of <b>MathClean</b>.
</p>


MathClean benchmark is designed to evaluate the effectiveness of mathematical data cleaning models. It was proposed to address the issue of quality contamination caused by incorrect mathematical synthetic data.

<p align="center">
    <img src="fig/answer_dataset/answer_dataset.jpg" width="40%"> <br>
  distribution of <b>answer_dataset</b>.
</p>

<p align="center">
    <img src="fig/question_dataset/question_dataset.jpg" width="40%"> <br>
  distribution of <b>question_dataset</b>.
</p>


The MathClean benchmark consists of 2,000 correct questions and 2,000 erroneous questions with additional 2,000 correct and erroneous answers sourced from augmented data based on GSM8K and MATH. Moreover, we also annotate error types for each question or answer, since it can assess whether models can correctly identify the error categories for future improvements.

## Data Format and Examples

MathClean is divided into 4 categories:Answer Error Detection, Answer Error Type Detection, Question Error Detection, and Question Error Type Detection.

The four types data of benchmark are placed under the `check_correct_answer/data`, `check_type_answer/data`, `check_correct_question/data` and `check_type_question/data` directory.

### Data Format
[string] The type of question: "multi_choice" or "free_form"
check_correct_answer and check_type_answer
```python
  {
        "question": [string] The question text,
        "answer": [string] The correct answer for the problem,
        "extent": [string] How obvious the error is, "obvious error" or "not obvious error" or "no error",
        "type": [string] "mathematical method" or "mathematical method" or null
    },
```
check_correct_question and check_type_question
```python
    {
        "output": [string] The question text,
        "type": [string] Error type of problem, "contradictions" or "expression_error" or "lack_of_conditions" or "unrealistic" or "correct"
    },
```

### Example
#### check_correct_answer
```python
  {
        "question": "Consider a scenario where a rectangular box has dimensions of 3 by 4 by 5 units. Now, imagine that there is a special event where the box expands by 1 unit in all directions, but this expansion happens only 8 times a year. For the rest of the year, the box remains in its original size. Calculate the total volume of the set of points that are inside or within one unit of the box for the entire year, expressed as $\\displaystyle {{m+n\\pi}\\over p}$, where $m$, $n$, and $p$ are positive integers, and $n$ and $p$ are relatively prime. Find $m+n+p$.",
        "answer": "I'd be happy to help you with that!\n\nLet's break down the problem step by step:\n\n**Step 1: Find the volume of the original box**\n\nThe dimensions of the box are 3 by 4 by 5 units, so its volume is:\n\nVolume = length × width × height = 3 × 4 × 5 = 60 cubic units\n\n**Step 2: Find the volume of the expanded box**\n\nThe box expands by 1 unit in all directions, so its new dimensions are 4 by 5 by 6 units. Its volume is:\n\nVolume = length × width × height = 4 × 5 × 6 = 120 cubic units\n\n**Step 3: Find the volume of the expanded box for the 8 times it expands**\n\nSince the box expands 8 times a year, we need to find the total volume of the expanded box for those 8 times. The volume of the expanded box is 120 cubic units, and it expands 8 times, so the total volume is:\n\n120 cubic units × 8 = 960 cubic units\n\n**Step 4: Find the volume of the original box for the remaining 44 weeks**\n\nSince the box remains in its original size for the remaining 44 weeks, we need to find the volume of the original box for those 44 weeks. The volume of the original box is 60 cubic units, and it remains in that size for 44 weeks, so the total volume is:\n\n60 cubic units × 44 = 2640 cubic units\n\n**Step 5: Find the total volume of the set of points**\n\nThe total volume of the set of points that are inside or within one unit of the box for the entire year is the sum of the volumes of the expanded box for the 8 times it expands and the volume of the original box for the remaining 44 weeks:\n\nTotal volume = 960 cubic units + 2640 cubic units = 3600 cubic units\n\n**Step 6: Write the answer in the required form**\n\nThe answer can be written in the form $\\displaystyle {{m+n\\pi}\\over p}$ as follows:\n\n3600 = $\\displaystyle {{3600}\\over 1}$ = $\\displaystyle {{3600+0\\pi}\\over 1}$\n\nSo, $m=3600$, $n=0$, and $p=1$. Therefore, $m+n+p = 3601$.\n\nThe answer is 3601.",
        "extent": "obvious error",
        "type": "mathematical method"
    },
```
#### check_correct_question
```python
    {
        "output": "A circle has a radius of 3 inches. The product of this radius and the circumference of the circle is equal to the circle's area. What is the length of the circumference of the circle, in inches?",
        "type": "contradictions"
    },
```
#### check_type_answer

```python
    {
        "question": "One piece of string is 2.5 inches long and another piece of string is 5.5 inches long. If a third piece of string is added that is 3.5 inches long, what is the average length, in inches, of these three pieces of string?",
        "answer": "I'd be happy to help you with that!\n\nTo solve this problem, we'll follow a step-by-step approach:\n\n1. Determine the total length of the three pieces of string: We have the lengths of the first two pieces: 2.5 inches and 5.5 inches. For the third piece, we're given a length of 3.5 inches. To find the total length, we add these values together:\n\n2.5 + 5.5 + 3.5 = 11.5 inches\n\n2. Find the average length: To find the average length, we divide the total length by the number of pieces. Since we have three pieces, we divide the total length by 3:\n\n11.5 \u00f7 3 = 3.83 inches\n\nSo, the average length of the three pieces of string is approximately 3.83 inches.\n\nTherefore, the answer is: \"3.83 inches\"",
        "extent": "not obvious error",
        "type": "logic error"
    },
```

#### check_type_question

```python
    {
        "output": "At Beaumont High School, there are 12 players on the baseball team. All 12 players are taking at least one of biology or chemistry. If 7 players are taking biology and 2.5 players are taking both sciences, how many players are taking chemistry?",
        "type": "unrealistic"
    },
```

## Usage

### Environment

requirement
```python
pip install -r requirements.txt
```

### Model Inference

We support three types of model inference: API, local model inference, and PRMs inference.

#### API

```python
cd check_correct_answer # check_type_answer, check_correct_question or check_type_question

python cot_GPT.py --model gpt-4o --dataset math
python cot_GPT.py --model gpt-4o --dataset gsm8k
```

The model generates text and we extracts the final answer.

```python
    {
        "generated_text": [string] The result of model inference,
        "result": [string] "correct" or "incorrect"
    },
```

The result is saved in `output/cot_{args.dataset}_GPT_{args.model}.json`

#### Local Model

```python
cd check_correct_answer # check_type_answer, check_correct_question or check_type_question

CUDA_VISIBLE_DEVICES=0,1 python cot.py --model Llama-3.1-8B-Instruct --dataset gsm8k
CUDA_VISIBLE_DEVICES=3,4 python cot.py --model Llama-3.1-8B-Instruct --dataset math
```

The model generates text and we extracts the final answer.

```python
    {
        "generated_text": [string] The result of model inference,
        "result": [string] "correct" or "incorrect"
    },
```

The result is saved in `output/cot_{args.dataset}_GPT_{args.model}.json`

#### PRM

```python
cd check_correct_answer # check_type_answer, check_correct_question or check_type_question

CUDA_VISIBLE_DEVICES=1 python cot_shepherd_prm.py --model math-shepherd-mistral-7b-prm --dataset math
CUDA_VISIBLE_DEVICES=0 python cot_shepherd_prm.py --model math-shepherd-mistral-7b-prm --dataset gsm8k

CUDA_VISIBLE_DEVICES=3 python cot_skywork_prm.py --model Skywork-PRM-7B --dataset math
CUDA_VISIBLE_DEVICES=1 python cot_skywork_prm.py --model Skywork-PRM-7B --dataset gsm8k

CUDA_VISIBLE_DEVICES=2 python cot_qwen_prm.py --model Qwen2.5-Math-PRM-7B --dataset math
CUDA_VISIBLE_DEVICES=3 python cot_qwen_prm.py --model Qwen2.5-Math-PRM-7B --dataset gsm8k
```
The model generates scores.

```python
    {
       "answer": [string] The answer for the problem,,
        "score": [float] The score of answer
    },
```

The preliminary result is saved in `output/cot_prm_{args.model}_{args.dataset}.json`

Then we need to run `score2result.py` to get the final result.

### Evaluation

Modify the `predictions_file_path` for the model output path you want to evaluate, like `output/cot_{args.dataset}_GPT_{args.model}.json`, then run `evaluation.py`

```python
    cd check_correct_answer # check_type_answer, check_correct_question or check_type_question

    python evaluation.py
```
