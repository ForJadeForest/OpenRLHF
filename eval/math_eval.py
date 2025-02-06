from vllm import LLM,SamplingParams
from datasets import load_dataset,load_from_disk
import os
from argparse import ArgumentParser
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
from transformers import AutoTokenizer

SYSTEM_PROMPT='You are a helpful assistant good at solving math problems with step-by-step reasoning. You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$.The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means your output should start with <think> and end with </answer>.'

SYSTEM_PROMPT_BASE=SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The answer is in latex format and wrapped in $...$."
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> Since $1+1=2$, so the answer is $2$. </think><answer> $2$ </answer>, which means assistant's output should start with <think> and end with </answer>.\nUser: __PROMPT__\n\nAssistant: "
)

def get_dataset(dataset,split=None):
    data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
    dataset = dataset.split("@")[0].strip()
    dataset_basename = os.path.basename(dataset)
    ext = os.path.splitext(dataset)[-1]
    # local python script
    if ext == ".py" or (
        os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
    ):
        data = load_dataset(dataset, trust_remote_code=True)
    # local text file
    elif ext in [".json", ".jsonl", ".csv"]:
        ext = ext.lower().strip(".")
        if ext == "jsonl":
            ext = "json"
        data = load_dataset(ext, data_files=dataset)
    # local dataset saved with `datasets.Dataset.save_to_disk`
    elif os.path.isdir(dataset):
        data = load_from_disk(dataset)
    # remote/local folder or common file
    else:
        data = load_dataset(dataset, data_dir=data_dir)
    if split:
        data = data[split]
    return data

def verify_math(content,sol):
    gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        result = float(verify(answer_parsed, gold_parsed))
    else:
        result = 0.0
        print("Failed to parse gold solution: ", sol)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model",type=str,required=True)
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--split",type=str,default=None)
    parser.add_argument("--problem_key",type=str,default="problem")
    parser.add_argument("--answer_key",type=str,default="answer")
    parser.add_argument("--system_prompt",type=str,default=SYSTEM_PROMPT)
    args = parser.parse_args()
    data = get_dataset(args.dataset,args.split)
    model = LLM(args.model,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = []
    is_base = "base" in args.model.lower()
    for d in data:
        problem = d[args.problem_key]
        if not is_base:
            message = [{"role":"system","content":args.system_prompt},{"role":"user","content":problem}]
        else:
            message = SYSTEM_PROMPT_BASE.replace("__PROMPT__",problem)
        messages.append(message)
    if not is_base:
        messages = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=0,max_tokens=3000,skip_special_tokens=False)
    outputs = model.generate(messages,sampling_params)
    results = []
    for output,d in zip(outputs,data):
        pred = output.outputs[0].text
        answer = d[args.answer_key]
        if answer[0] != "$":
            answer = '$' + answer + '$'
        results.append(verify_math(pred,answer))
    acc = sum(results) / len(results)
    print(f"Acc: {acc}")
