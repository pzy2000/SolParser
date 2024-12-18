import json
import os
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
from auto_gen_prompt import update_id
from extract_function_from_solidity_project import serialize
from logger import MyLogger


# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="sk-03f8ceb10b22426bb235639e45aa1c91", base_url="https://api.deepseek.com")
# print(client.models.list())
real_path_cargo = {}
with open('parsed_results.json', 'r') as file:
    data = json.load(file)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"gen_log_{current_time}.txt"
logger = MyLogger(f"logs/{log_file}")


def multiple_replace(original, replacements):
    for old, new in replacements.items():
        original = original.replace(old, new)
    return original


replacements = {
    "/openzeppelin-contracts/test/": "/openzeppelin-contracts/contracts/",
    "/ethernaut.git/contracts/test/": "/ethernaut.git/contracts/src/",
    ".t.sol": ".sol"
}


for file_path, file_content in data.items():
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
        logger.error(real_file_path + " (real_file_path) not found error!!!!!!!!!!!!!!!")
        logger.log(file_path + " (file_path) not found error!!!!!!!!!!!!!!!")

    real_path_cargo[real_file_path] = file_path
    # print("real_file_path", real_file_path)
logger.log("filter Over!")
with open('prompt_comment_gen.txt', 'r', encoding='utf-8') as file:
    prompt_temp = file.read()


for file_path, file_content in tqdm(data.items()):
    # if not file_path.endswith("Governor.sol"):
    #     continue
    if file_path not in real_path_cargo.keys():
        logger.warn("jumping file_path:\n" + file_path)
        continue
    if not file_content or not file_content[0]['methods']:
        logger.warn("jumping file_path:\n" + file_path)
        continue
    if file_path.endswith(".t.sol") or file_path.endswith(".test.sol") \
            or "test" in file_path or "forge" in file_path:
        logger.warn("jumping file_path:\n" + file_path)
        continue
    logger.log("file_path:\n" + file_path)
    for method in file_content[0]['methods']:
        if "llm_comment" in method.keys() and method['llm_comment']:
            logger.warn("jumping file_path: " + file_path + " method: " + method['identifier'] + "since llm_comment "
                                                                                                 "already exists!")
            continue
        identifier = method['identifier']
        flag = update_id(identifier, data[real_path_cargo[file_path]][0])
        comment = method['comment']
        if not comment or not flag:
            continue
        prompt = prompt_temp + '\n' + "// Function"
        function_body = method['body']
        prompt += '\n' + function_body
        # print("prompt:", prompt)
        t_and_function = prompt
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "You are a experienced software engineer, you are required to summarize solidity codes into comments."},
                {"role": "user", "content": t_and_function},
            ],
            stream=False
        )
        # print("======================================")
        output = str(response.choices[0].message.content)
        # print("output0:", output)
        output = output[output.find("// Generated Comments") + 22:]
        # print("output1:", output)
        output = output[:output.rfind("// END") - 6]
        method['llm_comment'] = output
        # 将解析结果序列化并写入 JSON 文件
with open('parsed_results_with_comment.json', "w") as json_file:
    json.dump(serialize(data), json_file, indent=4)

