import json
import os
from tqdm import tqdm
from TestParser import TestParser


# 为了确保 JSON 可序列化，需要转换解析结果为基础类型
def serialize(obj):
    if isinstance(obj, dict):
        return {key: serialize(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    else:
        return str(obj)  # 将对象转为字符串


if __name__ == '__main__':
    sol_files = ["demo.sol"]
    # 获取当前脚本的绝对路径，并拼接上 "libtree-sitter-solidity.so"
    current_dir = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件的绝对路径
    libtree_so_path = os.path.join(current_dir, "libtree-sitter-solidity.so")

    # 创建 TestParser 实例，传入拼接后的路径
    parser = TestParser(libtree_so_path, "solidity")
    # 遍历目录，获取所有 .sol 文件

    # 使用 tqdm 对文件进行解析
    parsed_results = {}
    for file_path in tqdm(sol_files, desc="Parsing .sol files"):
        parsed_classes = parser.parse_file(file_path)
        parsed_results[file_path] = parsed_classes

    # 导出结果到 JSON 文件
    output_json_file = "./demo_results.json"

    # 将解析结果序列化并写入 JSON 文件
    with open(output_json_file, "w") as json_file:
        json.dump(serialize(parsed_results), json_file, indent=4)

    print(f"Parsed results have been exported to {output_json_file}")
