# import argparse
#
# def main():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--input-path', required=True, type=str)
#     parser.add_argument('--vocab-size', required=True, type=int)
#     parser.add_argument('--special-tokens', nargs='+', type=str)

from datasets import load_dataset

# Load the TinyStories dataset
dataset = load_dataset("roneneldan/TinyStories", split="train")

# Inspect the first example
print(dataset[0])
