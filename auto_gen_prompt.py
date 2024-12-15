from __future__ import absolute_import, division, print_function
import argparse
import json
import os
import random
import re
import subprocess
import warnings
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='model to use for code generation. should be one of [CodeLlama,WizardCoder,'
                         'DeepSeek-Coder,OpenCodeInterpreter,Magicoder,Llama-3,Phi-2, Mistral]',
                    type=str)
parser.add_argument("--n_example", type=int, default=2)
parser.add_argument("--sample", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--k',
                    help='The number of highest probability vocabulary tokens to keep '
                         'for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.',
                    type=int, default=50)
parser.add_argument('--p',
                    help='Only the most probable tokens with probabilities that add up to top_p '
                         'or higher are considered during decoding. The valid range is 0.0 to 1.0. '
                         '1.0 is equivalent to disabled and is the default. Only applies to sampling '
                         'mode. Also known as nucleus sampling.',
                    type=float, default=0.95)
parser.add_argument('--temperature',
                    help='A value used to warp next-token probabilities in sampling mode. Values less '
                         'than 1.0 sharpen the probability distribution, resulting in "less random" output.'
                         ' Values greater than 1.0 flatten the probability distribution, resulting in "more '
                         'random" output. A value of 1.0 has no effect and is the default. '
                         'The allowed range is 0.0 to 2.0.',
                    type=float, default=1)
args = parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, device_map='auto')
num_return_sequences = 5


def read_file_with_indentation(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            # 读取文件的全部内容，保留原始缩进
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"错误：文件 {filename} 未找到。")
        return None
    except IOError:
        print(f"错误：读取文件 {filename} 时发生 IO 错误。")
        return None


def few_shot_inject(args, prompt, tokenizer, model):
    if prompt is not None:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # print("inputs", inputs['input_ids'])
        raw_outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=512,
            do_sample=True,
            top_p=args.p,
            top_k=args.k,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )
        output_list = []
        for raw_output in raw_outputs:
            output = tokenizer.decode(raw_output[len(inputs[0]):])
            output = output[:output.find("// End")].strip()
            output = output[:output.rfind("}") + 1]
            output_list.append(output)
            # print("=======================")
            # print(output)
        torch.cuda.empty_cache()
        return output_list


def multiple_replace(original, replacements):
    for old, new in replacements.items():
        original = original.replace(old, new)
    return original


replacements = {
    "/test/": "/contracts/",
    ".t.sol": ".sol"
}

with open('prompt_template.txt', 'r', encoding='utf-8') as file:
    prompt_temp = file.read()

# 读取 JSON 文件
with open('parsed_results.json', 'r') as file:
    data = json.load(file)

real_path_cargo = {}
for file_path, file_content in tqdm(data.items()):
    # print("file_path:\n", file_path)
    # if not file_path.endswith("Governor.sol"):
    #     continue
    if "forge" in file_path:
        continue
    if not (file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
            or "test" in file_path or "forge" in file_path):
        continue
    # print("file_path", file_path)
    real_file_path = multiple_replace(file_path, replacements)
    if not os.path.exists(real_file_path):
        print("Path not found error!!!!!!!!!!!!!!!")
        exit(666)
    real_path_cargo[real_file_path] = file_path
    # print("real_file_path", real_file_path)
print("filter Over!")


def update_id(identifier, file_cont):
    flag = False
    for method in file_cont['methods']:
        if identifier in method['body']:
            method['id'].append(identifier)
            flag = True

    return flag


repo_dir_path = "/root/openzeppelin-contracts"
number_total = 0
number_pass = 0
number_fail = 0
number_compiled_total = 0
number_compiled_fail = 0
for file_path, file_content in tqdm(data.items()):
    # if not file_path.endswith("Governor.sol"):
    #     continue
    if file_path not in real_path_cargo.keys():
        continue
    if not file_content or not file_content[0]['methods']:
        continue
    if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
            or "test" in file_path or "forge" in file_path:
        continue
    print("file_path:\n", file_path)

    for method in file_content[0]['methods']:
        # if "schedule" not in method['full_signature']:
        #     continue
        identifier = method['identifier']

        flag = update_id(identifier, file_content[0])
        comment = method['comment']
        if not comment or not flag:
            continue
        function_full_sig = method['full_signature'].strip() + '{' + '\n'
        prompt = prompt_temp + '\n' + comment + '\n' + "// Function"
        prompt += '\n' + function_full_sig
        # print(prompt)
        # print("===========================")
        output_list = few_shot_inject(args, prompt, tokenizer, model)
        output_list = [function_full_sig + output for output in output_list]
        # print("output_list: ", output_list)
        start = int(method['start'])
        end = int(method['end'])
        with open(f"{file_path}", 'r') as f:
            source = f.readlines()

        # print("-----------------")
        # pprint(source)
        # print("-----------------")
        PASS = False
        COMPILE_PASS = False
        for out in output_list:
            # print(type(patch))
            # print(type(source))
            with open(f"patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}.txt", 'w') as f:
                f.write(out)
            with open(f"patch_{real_path_cargo[file_path].split('/')[-1]}_function_{identifier}.txt", 'r') as f:
                patch = f.readlines()
            source_p = "\n".join(source[:start - 1] + patch + source[end:])
            # pprint(source)
            # print("type:", type(source))
            with open(f"{file_path}", 'r') as f:
                source_bk = f.read()
            file_path_bk = file_path.replace(".sol", ".sol.bak")
            with open(f"{file_path_bk}", 'w') as f:
                f.write(source_bk)
            with open(f"{file_path}", 'w') as f:
                f.write(source_p)
            match_path = real_path_cargo[file_path].split('/')[-1]
            print("match_path", match_path)
            test_process = subprocess.run(['forge', 'test', '--match-path', f'{match_path}'], capture_output=True,
                                          cwd=repo_dir_path,
                                          timeout=90)
            captured_stdout = test_process.stdout.decode()
            # print("captured_stdout", captured_stdout)
            with open(f"{file_path}", 'w') as f:
                f.write(source_bk)
            print("captured_stdout", captured_stdout)
            if "Compiler run failed:" in captured_stdout:
                continue
            COMPILE_PASS = COMPILE_PASS or True
            pattern = re.compile(
                r'Ran\s+(-?\d+)\s+test\s+suites?\s+in\s+([\d.]+)\s*(ms|s)\s+'
                r'\(([\d.]+)\s*(ms|s)\s+CPU time\):\s+'
                r'(\d+)\s+tests passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped\s+\((\d+)\s+total tests\)'
            )

            # 使用正则表达式进行匹配
            matches = pattern.findall(captured_stdout)
            print("matches:", matches)
            print("length:", len(matches))
            passes = int(matches[-1][5])
            fails = int(matches[-1][6])
            skips = int(matches[-1][7])
            total = int(matches[-1][8])
            PASS = PASS or True if fails == 0 else PASS or False
            print("============================")
            print("PASS:", PASS)
            print("passes:", passes)
            print("failures:", fails)
            print("skips:", skips)
            print("total:", total)
        number_pass += int(PASS)
        number_fail += int(not PASS)
        number_total += 1
        number_compiled_total += 1
        number_compiled_fail += int(not COMPILE_PASS)
        print("-----------------------------")
        print("number_pass:", number_pass)
        print("number_failures:", number_fail)
        print("number_total:", number_total)
        print(f"Pass@{num_return_sequences}:", number_pass / number_total)
        print(f"COMPILE successful rate:", (number_compiled_total - number_compiled_fail) / number_compiled_total)

    # pprint(file_content[0]['methods'])
