import json
import os
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--dp', type=str, default=5)
    args = parser.parse_args()
    task = args.task

    for partition in ["train","test"]:
        if not os.path.exists(f"upload_data/{partition}_{task}/"):
            os.makedirs(f"upload_data/{partition}_{task}/")
        with open(f"data/{task}_{partition}.json", 'r') as json_file:
            data = json.load(json_file)
        num_samples = data["len"]

        for i in range(0,num_samples):
            adj_mat = data["input"][i]
            temp = np.around(np.array(adj_mat), decimals=args.dp).tolist() # rounds to args.dp places
            # reduced 0.0 to 0 to save on input tokens
            for k in range(0, len(temp)):
                for j in range(0, len(temp)):
                    if temp[k][j] == 0.0:
                        temp[k][j] = 0
            adj_mat = temp

            with open(f"upload_data/{partition}_{task}/adj_mat_{partition}_{i}.txt", 'w+') as file:
                for row in adj_mat:
                    row_str = ', '.join(map(str, row))
                    file.write(row_str + '\n')

if __name__ == "__main__":
    main()