import sys
from pathlib import Path

import re
import os
import numpy as np
from tabulate import tabulate
from collections import defaultdict


def extract_data_from_file(file_path):
    # Read the content of the file
    
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Use regex to find the WIKITEXT2 PPL float value
    ppl_match = re.search(r"WIKITEXT2 PPL: (\d+\.\d+)", content)
    if ppl_match:
        ppl_value = float(ppl_match.group(1))
    else:
        ppl_value = -1

    # Use regex to extract the dictionary string
    # dict_match = re.search(r"({.*?})", content)
    dict_match = re.findall(r"({.*?})", content)

    if dict_match:
        # Convert the dictionary string to a dictionary object
        # results_dict = eval(dict_match.group(1))
        results_dict = eval(dict_match[-1])
    else:
        results_dict = {}
    
    return ppl_value, results_dict

def extract_additional_tasks_performance(folder_path, tasks, entry_points):
    
    results = {}
    for task, entry_point in zip(tasks, entry_points):
        new_folder_path = folder_path.rstrip("/") + f"_{task}"
        file_path = os.path.join(new_folder_path, Path(new_folder_path).name + ".log")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()
                
                # Use regex to extract the dictionary string
                # dict_match = re.search(r"({.*?})", content)
                dict_match = re.findall(r"({.*?})", content)
                results_dict = eval(dict_match[-1])
                
                results[task] = results_dict[entry_point]
            else:
                results[task] = -1
        except:
            results[task] = -1
            
    return results


def extract_additional_wiki_perplexity(folder_path, wiki_seqlens):
    
    results = {}
    for seqlen in wiki_seqlens:
        new_folder_path = folder_path.rstrip("/") + f"_wiki{seqlen}"
        file_path = os.path.join(new_folder_path, Path(new_folder_path).name + ".log")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Use regex to find the WIKITEXT2 PPL float value
            ppl_match = re.search(r"WIKITEXT2 PPL: (\d+\.\d+)", content)
            if ppl_match:
                ppl_value = float(ppl_match.group(1))
            else:
                ppl_value = -1
        else:
            ppl_value = -1
        
        results[f"wiki{seqlen}"] = ppl_value
            
    return results


# Define the file path
max_seed = int(sys.argv[1])
file_prefix = sys.argv[2]

ppl_list = {}
dict_results_list = {}

seeds = range(max_seed)

for seed in range(max_seed):
    
    if "@" in file_prefix:
        folder_path = file_prefix.replace("@0", f"@{seed}")
    else:
        folder_path = file_prefix + f"@{seed}"
        
        if seed == 0 and not os.path.exists(folder_path):
            folder_path = file_prefix # the first batch experiment was not saved with the seed naming convention

    print(folder_path)
    file_path = os.path.join(folder_path, Path(folder_path).name + ".log")
        # Call the function
        
    if os.path.exists(file_path):
        ppl_value, results_dict = extract_data_from_file(file_path)

        # Print the extracted values
        print("seed: ", seed)
        print("WIKITEXT2 PPL:", ppl_value)
        print("Extracted Dictionary:", results_dict)
        
        ppl_list[seed] = ppl_value
        dict_results_list[seed] = results_dict
        
        additional_results = extract_additional_tasks_performance(
            folder_path, 
            ["truthfulqa", "mmlu", "gsm8k"], 
            ["truthfulqa_mc2", "mmlu", "gsm8k_cot_llama"],
        )
        
        additional_wiki_results = extract_additional_wiki_perplexity(
            folder_path, 
            [512, 8192]
        )

        dict_results_list[seed].update(additional_results)
        dict_results_list[seed].update(additional_wiki_results)
        dict_results_list[seed]["wiki2048"] = ppl_value
        
        # rename the keys
        
        removed_original_keys = ["lambada_openai", "lambada_standard", "arc_challenge", "arc_easy"]
        updated_original_keys = ["lambada_o", "lambada_s", "arc_c", "arc_e"]
        
        for i, key in enumerate(removed_original_keys):
            if key in dict_results_list[seed]:
                dict_results_list[seed][updated_original_keys[i]] = dict_results_list[seed][key]
                del dict_results_list[seed][key]
        
        if "acc_avg" in dict_results_list[seed]:
            del dict_results_list[seed]["acc_avg"] # accuracy only for a subset of tasks
        
        avg_acc = np.mean([dict_results_list[seed][k] for k in dict_results_list[seed] if "wiki" not in k])
        
        # avg_acc = round(avg_acc, 4)
        
        dict_results_list[seed]["real_acc_avg"] = avg_acc
        
        # convert to percentage
        for key in dict_results_list[seed].keys():
            if "wiki" not in key:
                dict_results_list[seed][key] = round(dict_results_list[seed][key] * 100, 1)
        
    else:
        dict_results_list[seed] = defaultdict(lambda: -1)
        print(f"seed {seed} exp is not found")
        
def return_available(input_list, key):
    ret = []
    
    for _, v in input_list.items():
        if key in v and (v[key] != -100 and v[key] != -1):
            ret.append(v[key])
    return ret

print("=" * 5 + "Avg" + "=" * 5)

tasks = [
    "wiki512",
    "wiki2048",
    "wiki8192",
    # "lambada",
    "lambada_o",
    "lambada_s",
    "winogrande",
    "arc_c",
    "arc_e",
    "hellaswag",
    "piqa",
    # "acc_avg", # not real avg
    "mmlu",
    "gsm8k",
    "truthfulqa",
    "real_acc_avg",
]

# for k in dict_results_list:
#     print(k)
#     print(dict_results_list[k])

print(f"WIKITEXT2 PPL: {np.mean(list(ppl_list.values())):.3f}")

def custom_round(number, task_name):
    if "wiki" in task_name:
        return round(number, 3)
    return round(number, 1)

avg_dict = {k: custom_round(np.mean(return_available(dict_results_list, k)), k) for k in dict_results_list[0]}
std_dict = {k: custom_round(np.std(return_available(dict_results_list, k)), k) for k in dict_results_list[0]}

print("Extracted Dictionary: ", avg_dict)
print("Standard Deviation: ", std_dict)

table = [tasks] 

for s in seeds:
    table.append([])
    for t in tasks:
        if t in dict_results_list[s]:
            table[-1].append(dict_results_list[s][t])
        else:
            table[-1].append(-1)

table.append([avg_dict[t] for t in tasks])
table.append([std_dict[t] for t in tasks])

# Print the table
print(tabulate(table, tablefmt="grid"))