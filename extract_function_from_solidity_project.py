import json
import os
from tqdm import tqdm
from TestParser import TestParser
parser = TestParser("libtree-sitter-solidity.so", "solidity")

sol_files = []
# change root_path to include your dataset
root_path = "/root/openzeppelin-contracts"

# 遍历目录，获取所有 .sol 文件
for dirpath, dirnames, filenames in os.walk(root_path):
    for filename in filenames:
        if filename.endswith(".sol"):
            full_path = os.path.join(dirpath, filename)
            sol_files.append(full_path)

# 使用 tqdm 对文件进行解析
parsed_results = {}
for file_path in tqdm(sol_files, desc="Parsing .sol files"):
    try:
        parsed_classes = parser.parse_file(file_path)
        # pprint(parsed_classes)
        # if 'comment' in parsed_results.keys():
        parsed_results[file_path] = parsed_classes
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

# 导出结果到 JSON 文件
output_json_file = "/root/SolParser/parsed_results.json"


# 为了确保 JSON 可序列化，需要转换解析结果为基础类型
def serialize(obj):
    if isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    else:
        return str(obj)  # 将对象转为字符串


# 将解析结果序列化并写入 JSON 文件
with open(output_json_file, "w") as json_file:
    json.dump(serialize(parsed_results), json_file, indent=4)

print(f"Parsed results have been exported to {output_json_file}")
