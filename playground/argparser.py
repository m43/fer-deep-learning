import argparse
from typing import List

parser = argparse.ArgumentParser(description='Personal information')
parser.add_argument('--name', dest='name', type=str, help='Name of the candidate')
parser.add_argument('--surname', dest='surname', type=str, help='Surname of the candidate')
parser.add_argument('--age', dest='age', type=int, help='Age of the candidate')
parser.add_argument('--list', dest='list', nargs="+", type=int, help='Test list')

args = parser.parse_args()

print(args)
print(args.name)
print(vars(args))