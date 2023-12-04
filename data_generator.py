import argparse
from data_schema import schema, tasks
from clrs_30_data import main_caller as data_getter
import os
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--part', type=str, default="train")
    args = parser.parse_args()

    if args.task not in tasks:
        print(f"error {args.task} not in {tasks}")
    task_schema = schema[args.task]
    print(f"Schema for this task is {task_schema}")

    returned_dict = data_getter(args.task, args.part, args.num_samples)

    returned_dict["len"] = args.num_samples
    if not os.path.exists("data"):
        os.makedirs("data")
    file_path = os.path.join("data", f"{args.task}_{args.part}.json")
    with open(file_path, 'w') as json_file:
        json.dump(returned_dict, json_file)

if __name__ == "__main__":
    main()