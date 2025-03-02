import sys
import os
from tabulate import tabulate
from collections import defaultdict

max_seed = int(sys.argv[1])
pred_file = sys.argv[2]

pred_file = pred_file.rstrip("/")
result_path = "results/" + pred_file.split("/")[-1]


data_dict = defaultdict(dict)

for seed in range(max_seed):

    for task_name in [
        "tpo", "quality", "coursera", "sci_fi", "gsm100", "codeU", "topic_retrieval_longchat"
    ]:
        if "@" in result_path:
            if "@0" in result_path:
                log_file = os.path.join(result_path.replace("@0", f"@{seed}"), task_name + ".log")
            else:
                raise NotImplementedError("you should use @0 in the result path")
        else:
            result_path = result_path.removesuffix('/')
            log_file = os.path.join(result_path + f"@{seed}", task_name + ".log")
            if seed == 0 and not os.path.exists(log_file):
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
                
            number = round(number, 3)
            
            data_dict[seed][task_name] = number
        except:
            data_dict[seed][task_name] = -1
            

headers = list(data_dict[list(data_dict.keys())[0]].keys())

# Create the table
table = [headers]

avg_dict = defaultdict(float)
std_dict = defaultdict(float)

valid_exps_for_each_task = defaultdict(int)
for seed in data_dict:
    for task_name in data_dict[seed]:
        if data_dict[seed][task_name] != -1:
            valid_exps_for_each_task[task_name] += 1
            avg_dict[task_name] += data_dict[seed][task_name]

for task_name in avg_dict:
    avg_dict[task_name] /= valid_exps_for_each_task[task_name]
    avg_dict[task_name] = float(f"{avg_dict[task_name]:.3f}")
    
for seed in data_dict:
    for task_name in data_dict[seed]:
        if data_dict[seed][task_name] != -1:
            std_dict[task_name] += (data_dict[seed][task_name] - avg_dict[task_name]) ** 2
            
for task_name in avg_dict:
    std_dict[task_name] /= valid_exps_for_each_task[task_name]
    std_dict[task_name] = std_dict[task_name] ** 0.5
    std_dict[task_name] = float(f"{std_dict[task_name]:.3f}")

for seed in data_dict:
    table.append(data_dict[seed].values())

table.append(avg_dict.values())
table.append(std_dict.values())

# Print the table
print(tabulate(table, tablefmt="grid"))