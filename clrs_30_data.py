import clrs
import numpy as np
import argparse
from data_schema import schema, tasks
import sys

def pointers_to_list(inp, head=None):
    # helper method
    pointers = inp.copy()
    ordered_list = []
    temp_head = head
    if head is None:
        for i in range(0,len(pointers)):
            if pointers[i] == i:
                head = i
                break
        assert head != None
    pointers[head] = -1

    index_dict = {}
    for index, element in enumerate(pointers):
        if element in index_dict:
            index_dict[element].append(index)
        else:
            index_dict[element] = [index]

    current_list = [head]
    while len(current_list) != 0:
        current_index = current_list.pop(0)
        temp = []
        if current_index in index_dict:
            temp = sorted(index_dict[current_index])
        current_list = current_list + temp
        ordered_list.append(current_index)

    if len(ordered_list) != len(inp):
        print("lengths not equal in pointers to list")
        print(f"input {len(inp)}: ", inp)
        print(f"output {len(ordered_list)}: ", ordered_list)
        if (temp_head is None) or (len(inp)-len(ordered_list) > 1):#((3*len(inp))/4)):
            print("temp head ", temp_head)
            print("lengths not equal in pointers to list ... EXITING")
            exit()
    return np.array(ordered_list).tolist() # fixes weird json errors when list lengths differ 

def strong_comp_pointers(inp):
    pointers = inp.copy()
    ordered_list = []

    heads = []
    for i in range(0,len(pointers)):
        if pointers[i] == i:
            heads.append(i)
            pointers[i] = -1
    assert len(heads) > 0

    index_dict = {}
    for index, element in enumerate(pointers):
        if element in index_dict:
            index_dict[element].append(index)
        else:
            index_dict[element] = [index]
    for head in heads:
        temp_list = []
        current_list = [head]
        while len(current_list) != 0:
            current_index = current_list.pop(0)
            temp = []
            if current_index in index_dict:
                temp = sorted(index_dict[current_index])
            current_list = current_list + temp
            temp_list.append(current_index)
        ordered_list.append(temp_list)
    return ordered_list

def get_dataset(partition="train", alg="bubble_sort", batch_size_inp=1):
    ds, num_samples, spec = clrs.create_dataset(folder='CLRS30_v1.0.0', algorithm=alg, split=partition, batch_size=batch_size_inp)
    return ds

def insertion_sort(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="insertion_sort")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])) # outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def bubble_sort(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="bubble_sort")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def heap_sort(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="heapsort")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def quick_sort(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="quicksort")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def minimum(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="minimum")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def binary_search(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="binary_search")
    inputs = []
    outputs = []
    targets = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        targets.append(feedback.features.inputs[2].data[0]) # target
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(targets)
    return np.array(inputs), np.array(outputs), np.array(targets)

def quick_select(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="quickselect")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def maximum_subarray(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="find_maximum_subarray_kadane")
    inputs = []
    starts = []
    ends = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        starts.append(feedback.outputs[1].data[0]) 
        ends.append(feedback.outputs[0].data[0])
        if i >= num_samples-1:
            return np.array(inputs), np.array(starts), np.array(ends)
    return np.array(inputs), np.array(starts), np.array(ends)

def activity_selection(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="activity_selector")
    inputs_f = []
    inputs_s = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs_f.append(feedback.features.inputs[0].data[0]) # inputs
        inputs_s.append(feedback.features.inputs[2].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs_f), np.array(outputs), np.array(inputs_s)
    return np.array(inputs_f), np.array(outputs), np.array(inputs_s)

def task_scheduling(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="task_scheduling")
    inputs_d = []
    inputs_w = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs_d.append(feedback.features.inputs[0].data[0]) # inputs
        inputs_w.append(feedback.features.inputs[2].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs_d), np.array(outputs), np.array(inputs_w)
    return np.array(inputs_d), np.array(outputs), np.array(inputs_w)

def matrix_mul(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="matrix_chain_order")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0][1:,1:]) #outputs
        # if i == 0:
        #     print(feedback.outputs[0].data[0])
        #     print(feedback.outputs[0].data[0][1:,1:])
        #     exit()
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def longest_common_subseq(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="lcs_length")
    inputs = []
    input_strings = []
    outputs = []
    outputs_arrows = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        # we rearrange the outputs and take the bottom left corner to get a (4,8,8) matrix
        half_dimension = (feedback.outputs[0].data[0].shape[1])//2
        temp = np.swapaxes(feedback.outputs[0].data[0],0,2)[:,half_dimension:,:-half_dimension]#[:,8:,:-8]
        # We take the first three (8,8) matracies and combine them with the directional meanings
        temp2 = ((temp[0]*3)+temp[1]+(temp[2]*2)).T
        d = {3:"↖", 1:"↑", 2:"←"}
        temp2_arrow = np.vectorize(d.get)(temp2.astype(int))
        out = np.array([temp2,temp2_arrow]) # we return both the numbers and the arrow encodings for ease of use
        outputs.append(temp2)
        outputs_arrows.append(temp2_arrow)

        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        input_strings.append(feedback.features.inputs[2].data[0]) # inputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(input_strings), np.array(outputs_arrows)
    return np.array(inputs), np.array(outputs), np.array(input_strings), np.array(outputs_arrows)

def opt_binary_search_tree(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="optimal_bst")
    inputs_p = []
    inputs_q = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs_p.append(feedback.features.inputs[0].data[0]) # inputs
        inputs_q.append(feedback.features.inputs[2].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0][:-1,1:]) #outputs
        if i >= num_samples-1:
            return np.array(inputs_p), np.array(outputs), np.array(inputs_q)
    return np.array(inputs_p), np.array(outputs), np.array(inputs_q)

def bfs(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="bfs")
    inputs = []
    outputs = []
    starts = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        starts.append(feedback.features.inputs[3].data[0])
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(starts)
    return np.array(inputs), np.array(outputs), np.array(starts)


def dfs(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="dfs")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def topological_sort(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="topological_sort")
    inputs = []
    outputs = []
    output_heads = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(pointers_to_list(feedback.outputs[0].data[0])[::-1]) #outputs
        output_heads.append(feedback.outputs[1].data[0])
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(output_heads)
    return np.array(inputs), np.array(outputs), np.array(output_heads)

def articulation_points(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="articulation_points")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def bridges(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="bridges")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        out = feedback.outputs[0].data[0]
        out = np.where(out == -1, out + 1, out) # map -1's to 0's
        outputs.append(out) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def strongly_connected_comps(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="strongly_connected_components")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(strong_comp_pointers(feedback.outputs[0].data[0])) #outputs
        if i >= num_samples-1:
            return np.array(inputs), outputs
    return np.array(inputs), outputs

def kruskal(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="mst_kruskal")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs).astype(int)
    return np.array(inputs), np.array(outputs).astype(int)

def prim(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="mst_prim")
    inputs = []
    outputs = []
    starts = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        starts.append(feedback.features.inputs[3].data[0])
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(starts)
    return np.array(inputs), np.array(outputs), np.array(starts)

def bellmanford(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="bellman_ford")
    inputs = []
    outputs = []
    starts = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        starts.append(feedback.features.inputs[3].data[0])
        outputs.append(feedback.outputs[0].data[0])
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(starts)
    return np.array(inputs), np.array(outputs), np.array(starts)

def dijkstras(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="dijkstra")
    inputs = []
    outputs = []
    starts = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0].tolist()) # inputs
        starts.append(feedback.features.inputs[3].data[0])
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), starts
    return np.array(inputs), np.array(outputs), starts

def floydwarshall(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="floyd_warshall")
    inputs = []
    outputs = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        outputs.append(feedback.outputs[0].data[0].T) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs)
    return np.array(inputs), np.array(outputs)

def DAGsp(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="dag_shortest_paths")
    inputs = []
    outputs = []
    starts = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        starts.append(feedback.features.inputs[3].data[0])
        temp = feedback.outputs[0].data[0].copy()
        temp[temp == np.arange(temp.shape[0])] = -1
        temp[np.argmax(feedback.features.inputs[3].data[0])] = -2
        outputs.append(temp) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(starts)
    return np.array(inputs), np.array(outputs), np.array(starts)

def naive_strings(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="naive_string_matcher")
    inputs = []
    outputs = []
    strings = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        strings.append(feedback.features.inputs[2].data[0])
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(strings)
    return np.array(inputs), np.array(outputs), np.array(strings)

def kmp_strings(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="kmp_matcher")
    inputs = []
    outputs = []
    strings = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        inputs.append(feedback.features.inputs[0].data[0]) # inputs
        strings.append(feedback.features.inputs[2].data[0])
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(inputs), np.array(outputs), np.array(strings)
    return np.array(inputs), np.array(outputs), np.array(strings)

def segment_intersect(partition="train", num_samples=1):
    print("partition is ", partition)
    ds = get_dataset(partition=partition, alg="segments_intersect")
    xs = []
    outputs = []
    ys = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        xs.append(feedback.features.inputs[1].data[0]) # inputs
        ys.append(feedback.features.inputs[2].data[0])
        
        outputs.append(feedback.outputs[0].data) #outputs
        print(feedback.features.inputs[1].data[0])
        print(feedback.features.inputs[2].data[0])
        print(feedback.features.inputs[1].data[0].shape)
        print(feedback.features.inputs[2].data[0].shape)
        print(feedback.outputs[0].data)
        exit()
        if i >= num_samples-1:
            return np.array(xs), np.array(outputs), np.array(ys)
    return np.array(xs), np.array(outputs), np.array(ys)

def graham_scan(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="graham_scan")
    xs = []
    outputs = []
    ys = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        xs.append(feedback.features.inputs[1].data[0]) # inputs
        ys.append(feedback.features.inputs[2].data[0])
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(xs), np.array(outputs), np.array(ys)
    return np.array(xs), np.array(outputs), np.array(ys)

def jarvis_march(partition="train", num_samples=1):
    ds = get_dataset(partition=partition, alg="jarvis_march")
    xs = []
    outputs = []
    ys = []
    for i, feedback in enumerate(ds.as_numpy_iterator()):
        xs.append(feedback.features.inputs[1].data[0]) # inputs
        ys.append(feedback.features.inputs[2].data[0])
        
        outputs.append(feedback.outputs[0].data[0]) #outputs
        if i >= num_samples-1:
            return np.array(xs), np.array(outputs), np.array(ys)
    return np.array(xs), np.array(outputs), np.array(ys)

def main_caller(name, partition, num_samples):
    output_dict = {}
    if name == "insertion_sort":
        inp, out = insertion_sort(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "bubble_sort":
        inp, out = bubble_sort(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "heap_sort":
        inp, out = heap_sort(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "quick_sort":
        inp, out = quick_sort(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "minimum":
        inp, out = minimum(partition, num_samples) # one hot output
        output_dict["input"] = inp
        output_dict["output"] = np.argmax(out, 1)[:, np.newaxis]
    elif name == "binary_search":
        inp, out, tar = binary_search(partition, num_samples)# one hot output, single value target
        output_dict["input"] = inp
        output_dict["output"] = np.argmax(out, 1)[:, np.newaxis]
        output_dict["target"] = tar[:, np.newaxis]
    elif name == "quick_select":
        inp, out = quick_select(partition, num_samples) # always trying to find median, one hot output
        output_dict["input"] = inp
        output_dict["output"] = np.argmax(out, 1)[:, np.newaxis]
    elif name == "maximum_subarray":
        inp, out, tar = maximum_subarray(partition, num_samples) # returns inputs, starts, ends, starts, ends one hot
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(out, 1)[:, np.newaxis]
        output_dict["end"] = np.argmax(tar, 1)[:, np.newaxis]
    elif name == "activity_selection":
        inp, out, tar = activity_selection(partition, num_samples) # returns inputs_f, outputs, inputs_s, outputs one hot encoded
        output_dict["input_f"] = inp
        output_dict["input_s"] = tar
        indices_list = [np.where(row == 1)[0].tolist() for row in out]
        output_dict["output"] = indices_list
    elif name == "task_scheduling":
        inp, out, tar = task_scheduling(partition, num_samples) # returns inputs_d, outputs, inputs_w, outputs one hot
        output_dict["input_d"] = inp
        output_dict["input_w"] = tar
        indices_list = [np.where(row == 1)[0].tolist() for row in out]
        output_dict["output"] = indices_list
    elif name == "matrix_chain_mul":
        inp, out = matrix_mul(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "longest_common_subseq":
        inp, out, tar, out_arrows = longest_common_subseq(partition, num_samples) # returns inputs_d, outputs, input_strings
        output_dict["input_d"] = inp
        output_dict["input_string"] = tar
        output_dict["output"] = out
        output_dict["output_arrows"] = out_arrows
    elif name == "opt_bst": 
        inp, out, tar = opt_binary_search_tree(partition, num_samples) # returns inputs_p, outputs, inputs_q
        output_dict["input_p"] = inp
        output_dict["input_q"] = tar
        output_dict["output"] = out
    elif name == "bfs":
        inp, out, tar = bfs(partition, num_samples) # returns inputs, outputs, starting nodes (one hot encoded)
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "dfs":
        inp, out = dfs(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "topological_sort":
        inp, out, tar = topological_sort(partition, num_samples) # returns inputs, outputs, output heads (one hot)
        output_dict["input"] = inp
        output_dict["output_head"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "articulation_points":
        inp, out = articulation_points(partition, num_samples) # output is 1 for articulation nodes
        output_dict["input"] = inp
        indices_list = [np.where(row == 1)[0].tolist() for row in out]
        output_dict["output"] = indices_list # is list not numpy array
    elif name == "bridges":
        inp, out = bridges(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "strongly_connected_comps":
        inp, out = strongly_connected_comps(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "kruskal":
        inp, out = kruskal(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "prim":
        inp, out, tar = prim(partition, num_samples) # returns inputs, outputs, starting node (one hot)
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "bellman_ford":
        inp, out, tar = bellmanford(partition, num_samples) # returns inputs, outputs, starting node (one hot)
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "dijkstras":
        inp, out, tar = dijkstras(partition, num_samples) # returns inputs, outputs, starting node (one hot)
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "floyd_warshall":
        inp, out = floydwarshall(partition, num_samples)
        output_dict["input"] = inp
        output_dict["output"] = out
    elif name == "DAG_sp":
        inp, out, tar = DAGsp(partition, num_samples) # returns input, output, start node (one hot)
        output_dict["input"] = inp
        output_dict["start"] = np.argmax(tar, 1)[:, np.newaxis]
        output_dict["output"] = out
    elif name == "naive_strings":
        inp, out, tar = naive_strings(partition, num_samples) # returns inputs, outputs (one hot), strings
        output_dict["input"] = inp
        output_dict["string"] = tar 
        output_dict["output"] = np.argmax(out, 1)[:, np.newaxis]
    elif name == "kmp_strings":
        inp, out, tar = kmp_strings(partition, num_samples)
        output_dict["input"] = inp
        output_dict["string"] = tar 
        output_dict["output"] = np.argmax(out, 1)[:, np.newaxis]
    elif name == "segment_intersect":
        inp, out, tar = segment_intersect(partition, num_samples) # returns x, outputs, y
        output_dict["x"] = inp
        output_dict["y"] = tar
        output_dict["output"] = out
    elif name == "graham_scan":
        inp, out, tar = graham_scan(partition, num_samples) # returns x, outputs, y
        output_dict["x"] = inp
        output_dict["y"] = tar 
        indices_list = [np.where(row == 1)[0].tolist() for row in out]
        output_dict["output"] = indices_list
    elif name == "jarvis_march": 
        inp, out, tar = jarvis_march(partition, num_samples) # returns x, outputs, y
        output_dict["x"] = inp
        output_dict["y"] = tar 
        indices_list = [np.where(row == 1)[0].tolist() for row in out]
        output_dict["output"] = indices_list
    else:
        print("name not found")
        exit()
    for key in output_dict:
        if isinstance(output_dict[key], list):
            output_dict[key] = output_dict[key]
        else:   
            output_dict[key] = output_dict[key].tolist()
    return output_dict