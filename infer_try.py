from __future__ import absolute_import, division, print_function
import os
import torch
import random
import argparse
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def few_shot_inject(args, tokenizer, model):
    #     origin_prompt = """
    #     I will give you a requirement, please generate a solidity function to solve the requirement.
    #     """
    # 读取 prompt.txt 文件
    filename = 'prompt.txt'
    file_content = read_file_with_indentation(filename)

    if file_content is not None:
        # print("文件内容：")
        # print(file_content)
        prompt = file_content
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        raw_outputs = model.generate(
            inputs,
            max_new_tokens=512,
            # do_sample=True,
            # top_p=args.p,
            # top_k=args.k,
            # temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        output = tokenizer.decode(raw_outputs[0][len(inputs[0]):])
        # output = tokenizer.decode(raw_outputs[0])
        output = output[:output.find("// End")].strip()
        output = output[:output.rfind("}") + 1]
        # print("raw_outputs: >>>")
        # print(raw_outputs)
        print(output)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='model to use for clone generation. should be one of [CodeLlama,WizardCoder,'
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
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', use_fast=True)
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-coder-6.7b-instruct', trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16, device_map='auto')
    few_shot_inject(args, tokenizer, model)


if __name__ == "__main__":
    main()
