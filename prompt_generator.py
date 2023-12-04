import argparse
from data_schema import schema, tasks
import json
import numpy as np

def insertion_sort(input):
    return f"Insertion Sort: {input}."

def bubble_sort(input):
    return f"Bubble Sort: {input}."

def heap_sort(input):
    return f"Heap Sort: {input}."

def quick_sort(input):
    return f"Quick Sort: {input}."

def minimum(input):
    return f"Minimum: {input}."

def binary_search(input, target):
    return f"Binary Search: {input}, {target[0]}."

def quick_select(input):
    return f"Quick Select: {input}."

def maximum_subarray(input):
    return f"Maximum Subarray: {input}."

def activity_selection(input_s, input_f):
    return f"Activity Selection: {input_s}, {input_f}."

def task_scheduling(input_w, input_d):
    return f"Task Scheduling: penalties {input_w}, deadlines {input_d}."

def matrix_mul(input):
    return f"Matrix Chain Order: {input}."

def longest_common_subseq(input_d, input_string):
    # String 1 is the elements of input_d, marked by 0's in input_string
    # String 2 is the elements of input_d, marked by 1's in input_string
    input_string = np.array(input_string)
    input_d = np.array(input_d)
    string1 = input_d[np.where(input_string == 0)[0]]
    string2 = input_d[np.where(input_string == 1)[0]]
    string1 = [[int(element) for element in row] for row in string1]
    string2 = [[int(element) for element in row] for row in string2]
    return f"Longest Common Subsequence: {string1}, {string2}."

def opt_binary_search_tree(input_p, input_q):
    return f"Optimal Binary Search Tree: probabilities {input_p}, dummy probabilities {input_q}"

def bfs_printer(adj, start):
    return f"Breadth First Search: {adj}, start {start[0]}."

def dfs(adj):
    return f"Depth First Search: {adj}."

def topological_sort(adj):
    return f"Topological Sort: {adj}."

def articulation_points(adj):
    return f"FArticulation Points: {adj}."

def bridges(adj):
    return f"Bridges: {adj}."

def strongly_connected_comps(adj):
    return f"Strongly Connected Components: {adj}."

def kruskal(adj):
    return f"Kruscal's Minimum Spanning Tree: {adj}."

def prim(adj, start):
    return f"Prim's Minimum Spanning Tree: {adj}, start {start[0]}."

def bellmanford(adj, start):
    return f"Bellman-Ford: {adj}, start {start[0]}."

def dijkstras(adj, start):
    return f"Dijkstras Algorithm: {adj}, start {start[0]}."

def floydwarshall(adj):
    return f"Floyd-Warshall: {adj}."

def DAGsp(adj, start):
    return f"Directed Acyclic Graphs: {adj}, start {start[0]}."

def naive_strings(input, string):
    # String 1 is the elements of input, marked by 0's in string
    # String 2 is the elements of input, marked by 1's in string
    input_string = np.array(string)
    input = np.array(input)
    string1 = input[np.where(input_string == 0)[0]].tolist()
    string2 = input[np.where(input_string == 1)[0]].tolist()
    string1 = [[int(element) for element in row] for row in string1]
    string2 = [[int(element) for element in row] for row in string2]
    return f"Naive string matching: {string1} and {string2}."

def kmp_strings(input, string):
    # String 1 is the elements of input, marked by 0's in string
    # String 2 is the elements of input, marked by 1's in string
    input_string = np.array(string)
    input = np.array(input)
    string1 = input[np.where(input_string == 0)[0]].tolist()
    string2 = input[np.where(input_string == 1)[0]].tolist()
    string1 = [[int(element) for element in row] for row in string1]
    string2 = [[int(element) for element in row] for row in string2]
    return f"Knuth-Morris-Pratt: {string1}, {string2}."

def segment_intersect(x,y):
    return f"Segment Intersect: {x}, {y}."

def graham_scan(x,y):
    return f"Graham Scan: {x}, {y}."

def jarvis_march(x,y):
    return f"Jarvis March: {x}, {y}."

def prompt_gen(task, partition, max_samples):
    with open(f"data/{task}_{partition}.json", 'r') as json_file:
        data = json.load(json_file)
    num_samples = data["len"]
    if num_samples < max_samples:
        print(f"Only {num_samples} samples in data, you requested {max_samples}")
        exit()

    name = task
    prompts = []
    outputs = []
    for i in range(0,max_samples):    
        if name == "insertion_sort":
            prompt = insertion_sort(data["input"][i])
        elif name == "bubble_sort":
            prompt = bubble_sort(data["input"][i])
        elif name == "heap_sort":
            prompt = heap_sort(data["input"][i])
        elif name == "quick_sort":
            prompt = quick_sort(data["input"][i])
        elif name == "minimum":
            prompt = minimum(data["input"][i]) 
        elif name == "binary_search":
            prompt = binary_search(data["input"][i], data["target"][i])
        elif name == "quick_select":
            prompt = quick_select(data["input"][i]) 
        elif name == "maximum_subarray":
            prompt = maximum_subarray(data["input"][i]) 
        elif name == "activity_selection":
            prompt = activity_selection(data["input_s"][i], data["input_f"][i]) 
        elif name == "task_scheduling":
            prompt = task_scheduling(data["input_w"][i], data["input_d"][i])
        elif name == "matrix_chain_mul":
            prompt = matrix_mul(data["input"][i])
        elif name == "longest_common_subseq":
            prompt = longest_common_subseq(data["input_d"][i], data["input_string"][i])
        elif name == "opt_bst":
            prompt = opt_binary_search_tree(data["input_p"][i], data["input_q"][i])
        elif name == "bfs":
            prompt = bfs_printer(data["input"][i], data["start"][i])
        elif name == "dfs":
            prompt = dfs(data["input"][i])
        elif name == "topological_sort":
            prompt = topological_sort(data["input"][i])
        elif name == "articulation_points":
            prompt = articulation_points(data["input"][i])
        elif name == "bridges":
            prompt = bridges(data["input"][i])
        elif name == "strongly_connected_comps":
            prompt = strongly_connected_comps(data["input"][i])
        elif name == "kruskal":
            prompt = kruskal(data["input"][i])
        elif name == "prim":
            prompt = prim(data["input"][i], data["start"][i])
        elif name == "bellman_ford":
            prompt = bellmanford(data["input"][i], data["start"][i])
        elif name == "dijkstras":
            prompt = dijkstras(data["input"][i], data["start"][i])
        elif name == "floyd_warshall":
            prompt = floydwarshall(data["input"][i])
        elif name == "DAG_sp":
            prompt = DAGsp(data["input"][i], data["start"][i])
        elif name == "naive_strings":
            prompt = naive_strings(data["input"][i], data["string"][i])
        elif name == "kmp_strings":
            prompt = kmp_strings(data["input"][i], data["string"][i])
        elif name == "segment_intersect":
            prompt = segment_intersect(data["x"][i], data["y"][i])
        elif name == "graham_scan":
            prompt = graham_scan(data["x"][i], data["y"][i])
        elif name == "jarvis_march": 
            prompt = jarvis_march(data["x"][i], data["y"][i]) 
        prompts.append(prompt)
        # maximum_subarray and topological_sort have two outputs (longest_common_subseq has 2 options of how to view outputs)
        if name == "maximum_subarray":
            outputs.append([data["start"][i], data["end"][i]])
        elif name == "topological_sort":
            outputs.append([data["output_head"][i], data["output"][i]])
        elif name == "longest_common_subseq":
            outputs.append([data["output"][i], data["output_arrows"][i]])
        else:
            outputs.append(data["output"][i])
    return prompts, outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--part', type=str, default="train")
    parser.add_argument('--num_samples', type=int, default=1)
    args = parser.parse_args()

    if args.task not in tasks:
        print(f"Error {args.task} not in {tasks}")
        exit()
    task_schema = schema[args.task]
    print(f"Schema for this task is {task_schema}")
    prompts, outputs = prompt_gen(args.task, args.part, args.num_samples)
    print(prompts)
    print(outputs)

if __name__ == "__main__":
    main()
