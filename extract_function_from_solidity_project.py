from pprint import pprint
from TestParser import TestParser

parser = TestParser("/root/openzeppelin-contracts/libtree-sitter-solidity.so", "solidity")
buggy_function = dict()

# for project in projects:
buggy_classes = parser.parse_file("/root/openzeppelin-contracts/contracts/governance/Governor.sol")
buggy_methods = dict()
# print("buggy_classes:")
pprint(buggy_classes)
for buggy_class in buggy_classes:
    for method in buggy_class['methods']:
        buggy_methods[method["full_signature"]] = method["body"]
