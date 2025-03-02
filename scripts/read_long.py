import sys
from pathlib import Path

import re
import os
import numpy as np
from tabulate import tabulate
from collections import defaultdict


import re
import os
import sys
import json
from tabulate import tabulate
from collections import defaultdict


max_seed= int(sys.argv[1])
folder = sys.argv[2]


def extract_longeval_results(folder_path, lengths=[300, 460, 620]):
    results = {}
    
    stem = Path(folder_path).parts[-2:]
    
    if isinstance(stem, str):
        stem = [stem]
    
    new_folder_path = os.path.join(
        "qllm-eval/qllm_eval/evaluation/q_long/new_cases/lines/predictions/",
        *stem
    )
    
    new_folder_path = new_folder_path.rstrip("/") + f"_w_16_a_16_kv_16"
    
    # Meta-Llama-3-8B-Instruct/llama3-8b-instruct_quarot_3bit_w_n512_l2048_org_attn_none1_min0.01_expand8@2_w_16_a_16_kv_16/
    for length in lengths:
        complete_filename = os.path.join(new_folder_path, f"{length}_response.txt")
        
        try:
            with open(complete_filename, "r") as f:
                lines = f.readlines()
                
                result = lines[-1]
                
                match = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", result)

                # Check if a match is found and print it
                if match:
                    accuracy = float(match.group(1))
                    
                    results[f"le_{length}"] = round(accuracy * 100, 2)
                else:
                    results[f"le_{length}"] = -1
                    
        except:
            results[f"le_{length}"] = -1
                
    return results

def extract_leval_results(folder_path, tasks=["tpo", "quality", "coursera", "sci_fi", "gsm100", "codeU", "topic_retrieval_longchat"]):
    results = {}
    
    stem = Path(folder_path).parts[-1:]
    
    if isinstance(stem, str):
        stem = [stem]
    
    new_folder_path = os.path.join(
        "LEval/results/",
        *stem
    )
    
    new_folder_path = new_folder_path.rstrip("/") + f"-8k"
    
    for task_name in tasks:
        result_path = new_folder_path.removesuffix('/')
        log_file = os.path.join(result_path, task_name + ".log")

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()

            if task_name in ["topic_retrieval_longchat", "sci_fi"]:
                # print(task_name, lines[-1].split(":")[-1].strip())
                number = float(lines[-1].split(":")[-1].strip())
            else:
                metrics = eval(lines[0])
                # print(task_name, metrics["exact_match"])
                number = float(metrics["exact_match"])
                
            results[task_name] = round(number, 2)
        except:
            results[task_name] = -1
            
    return results


def compute_acc_banking(data):
    correct = 0
    
    for d in data:
        if d["pred"] == d["label"]:
            correct += 1
    return correct / len(data)


def compute_acc_tacred(data):
    selected_labels = ['org:founded_by', 'per:employee_of', 'org:alternate_names', 'per:cities_of_residence', 'per:children', 'per:title', 'per:siblings', 'per:religion', 'per:age', 'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees', 'per:countries_of_residence', 'org:city_of_headquarters', 'org:members', 'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters', 'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation', 'per:other_family', 'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders', 'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death']

    total_label = 0
    total_pred = 0
    total_correct = 0
    for d in data:
        pred, label = d["pred"], d["label"]
        if pred in selected_labels:
            total_pred += 1
        if label == pred:
            total_correct += 1
        total_label += 1

    precision = total_correct / (total_pred + 1e-8)
    recall = total_correct / (total_label + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return f1


def extract_banking77_results(folder_path, rounds=[2,3]):
    results = {}
    
    stem = Path(folder_path).parts[-1:]
    
    if isinstance(stem, str):
        stem = [stem]
    
    new_folder_path = os.path.join(
        "LongICLBench/bank77_round_result",
        *stem
    )
    
    for _round in rounds:
        file_name = new_folder_path.removesuffix('/') + f"_{_round}.json"
        
        try:
            with open(file_name, "r") as f:
                data = json.load(f)

            number = compute_acc_banking(data) * 100
            
            results[f"bank_{_round}"] = round(number, 2)
        except:
            results[f"bank_{_round}"] = -1
    
    return results


def extract_tacred_results(folder_path, rounds=[1,2]):
    results = {}
    
    stem = Path(folder_path).parts[-1:]
    
    if isinstance(stem, str):
        stem = [stem]
    
    new_folder_path = os.path.join(
        "LongICLBench/tacred_round_result",
        *stem
    )
    
    for _round in rounds:
        file_name = new_folder_path.removesuffix('/') + f"_{_round}.json"
        
        try:
            with open(file_name, "r") as f:
                data = json.load(f)

            number = compute_acc_tacred(data) * 100
            
            results[f"tacred_{_round}"] = round(number, 2)
        except:
            results[f"tacred_{_round}"] = -1
    
    return results


def extract_code_results(folder_path):
    results = {}
    
    stem = Path(folder_path).parts[-3:]
    
    if isinstance(stem, str):
        stem = [stem]
    
    new_folder_path = os.path.join(
        "lca-baselines/library_based_code_generation/results",
        *stem
    )
    
    new_folder_path = new_folder_path.removesuffix('/')
    file_name = os.path.join(new_folder_path, "metadata.json")
    
    try:
        with open(file_name, "r") as f:
            data = json.load(f)

        number = data["metrics"]["ChrF"]["mean"]
        
        results[f"long_code"] = round(number, 3)
    except:
        results[f"long_code"] = -1
    
    return results


# Define the file path
max_seed = int(sys.argv[1])
file_prefix = sys.argv[2].removesuffix("/")

ppl_list = {}
dict_results_list = defaultdict(dict)

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
        
    # if os.path.exists(file_path):
    # ppl_value, results_dict = extract_data_from_file(file_path)

    # # Print the extracted values
    # print("seed: ", seed)
    # print("WIKITEXT2 PPL:", ppl_value)
    # print("Extracted Dictionary:", results_dict)
    
    # ppl_list[seed] = ppl_value
    # dict_results_list[seed] = results_dict
    
    # additional_results = extract_additional_tasks_performance(
    #     folder_path, 
    #     ["truthfulqa", "mmlu", "gsm8k"], 
    #     ["truthfulqa_mc2", "mmlu", "gsm8k_cot_llama"],
    # )
    
    longeval_results = extract_longeval_results(folder_path, lengths=[300, 460, 620])
    
    Leval_tasks = ["tpo", "quality", "coursera", "sci_fi", "gsm100", "codeU", "topic_retrieval_longchat"]
    leval_results = extract_leval_results(
        folder_path, 
        tasks=Leval_tasks
    )
    
    # additional_wiki_results = extract_additional_wiki_perplexity(
    #     folder_path, 
    #     [512, 8192]
    # )
    
    # banking_results = extract_banking77_results(folder_path, rounds=[2,3])
    # tacred_results = extract_tacred_results(folder_path, rounds=[1,2])
    
    banking_results = extract_banking77_results(folder_path, rounds=[3])
    tacred_results = extract_tacred_results(folder_path, rounds=[2])
    
    long_code_results = extract_code_results(folder_path)
    
    dict_results_list[seed].update(longeval_results)
    dict_results_list[seed].update(leval_results)
    dict_results_list[seed].update(banking_results)
    dict_results_list[seed].update(tacred_results)
    dict_results_list[seed].update(long_code_results)
            
    # add avg results
    dict_results_list[seed]["LongEval"] = round(np.mean(
        [dict_results_list[seed][k] for k in ["le_300", "le_460", "le_620"]]
    ), 2)
    
    dict_results_list[seed]["LEval"] = round(np.mean(
        [dict_results_list[seed][k] for k in Leval_tasks]
    ), 2)
    
    dict_results_list[seed]["ICL"] = round(np.mean(
        [dict_results_list[seed][k] for k in ["bank_3", "tacred_2"]]
    ), 2)

    removed_original_keys = ["topic_retrieval_longchat", "long_code"]
    updated_original_keys = ["TopRet", "Lcode"]
    
    for i, key in enumerate(removed_original_keys):
        if key in dict_results_list[seed]:
            dict_results_list[seed][updated_original_keys[i]] = dict_results_list[seed][key]
            del dict_results_list[seed][key]
    
    # dict_results_list[seed].update(additional_wiki_results)
    # dict_results_list[seed]["wiki2048"] = ppl_value
    # else:
    #     dict_results_list[seed] = defaultdict(lambda: -1)
    #     print(f"seed {seed} exp is not found")
    
        
def return_available(input_list, key):
    ret = []
    
    for _, v in input_list.items():
        if v[key] != -1:
            ret.append(v[key])
    return ret

print("=" * 5 + "Avg" + "=" * 5)

tasks = [
    "le_300",
    "le_460",
    "le_620",
    "LongEval",
    "tpo", 
    "quality", 
    "coursera", 
    "sci_fi", 
    "gsm100", 
    "codeU", 
    "TopRet",
    "LEval",
    # "bank_2",
    "bank_3",
    # "tacred_1",
    "tacred_2",
    "ICL",
    "Lcode",
]

round_to_two_exceptions = ["Lcode"]

def custom_round(number, task_name):
    if "Lcode" in task_name:
        return round(number, 3)
    return round(number, 2)

avg_dict = {k: custom_round(np.mean(return_available(dict_results_list, k)), k) for k in dict_results_list[0]}
std_dict = {k: custom_round(np.std(return_available(dict_results_list, k)), k)  for k in dict_results_list[0]}

# for k in round_to_two_exceptions:
#     avg_dict[k] = float(f"{np.mean(return_available(dict_results_list, k)):.3f}")
#     std_dict[k] = float(f"{np.std(return_available(dict_results_list, k)):.3f}")

print("Extracted Dictionary: ", avg_dict)
print("Standard Deviation: ", std_dict)

table = [tasks] 

for s in seeds:
    table.append([])
    for t in tasks:
        table[-1].append(dict_results_list[s][t])

table.append([avg_dict[t] for t in tasks])
table.append([std_dict[t] for t in tasks])

# Print the table
print(tabulate(table, tablefmt="grid"))