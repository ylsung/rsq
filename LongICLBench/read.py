import re
import os
import sys
from tabulate import tabulate
from collections import defaultdict

import json



def compute_acc_banking(data):
    correct = 0
    
    for d in data:
        if d["pred"] == d["label"]:
            correct += 1
    print(len(data))
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


max_seed= int(sys.argv[1])
file_name = sys.argv[2]


tasks = ["bank77", "tacred"]
functions = [compute_acc_banking, compute_acc_tacred]

accuracies = defaultdict(list)
lengths = defaultdict(list)


seeds = range(max_seed)

start_round = 0

rounds = [1, 2, 3, 4]
for round in rounds:
    if f"_{round}_" in file_name:
        start_round = round
        
target_task = None
accuracy_function = None
for task, function in zip(tasks, functions):
    if task in file_name:
        target_task = task
        accuracy_function = function
        break
    
assert target_task is not None, f"Task not found in {file_name}"

for round in rounds:
    # file_name = file_name.replace(start_task, task)
    for seed in seeds:
        new_file_name = file_name.replace(f"_{start_round}_", f"_{round}_")
        if "@" in new_file_name:
            if "@0" in new_file_name:
                new_file_name = new_file_name.replace("@0", f"@{seed}") 
            else:
                raise NotImplementedError("This script only supports starting with seed 0")
        else:
            # 16 bit
            # for old naming convention
            new_file_name = new_file_name
            # if seed == 0 and not os.path.exists(new_file_name):
            #     new_file_name = new_file_name
            # else:
            #     raise NotImplementedError("This script only supports starting with seed 0")

        complete_filename = new_file_name
        
        try:
            with open(complete_filename, "r") as f:
                data = json.load(f)
                
            accuracies[seed].append(accuracy_function(data))
        except:
            accuracies[seed].append(-1)
            print(f"No such seed")
        # # print(completqe_filename)
        # try:
        #     with open(complete_filename, "r") as f:
        #         lines = f.readlines()
                
        #         result = lines[-1]
                
        #         match = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", result)

        #         # Check if a match is found and print it
        #         if match:
        #             accuracy = float(match.group(1))
                    
        #             accuracies[seed].append(accuracy)
        #         else:
        #             accuracies[seed].append(-1)
                    
        #         # Regular expression to find a floating point number following 'length'
        #         match = re.search(r"length\s*([0-9]*\.?[0-9]+)", result)

        #         # Check if a match is found and print it
        #         if match:
        #             length = float(match.group(1))
        #             length = f"{length:2f}"
        #             lengths[seed].append(length)
        #         else:
        #             lengths[seed].append(-1)
        # except:
        #     accuracies[seed].append(-1)
        #     lengths[seed].append(-1)


# names = ["_".join(f.split("_")[:-1]) for f in files]
names = [target_task] * len(rounds)
            
# Create the table
table = [names, rounds] 

for s in seeds:
    table.append(accuracies[s])
    
    
def get_valid_cases(accuracies, seeds, i):
    length = len([s for s in seeds if accuracies[s][i] != -1])
    
    if length == 0:
        return 1 # to make sure the division is not 0 and the average result is -1
    
    return length
avg_accuracies = [sum([accuracies[s][i] for s in seeds if accuracies[s][i] != -1]) / get_valid_cases(accuracies, seeds, i) for i in range(len(accuracies[0]))]
std_accuracies = [(sum([(accuracies[s][i] - avg_accuracies[i]) ** 2 for s in seeds if accuracies[s][i] != -1]) / get_valid_cases(accuracies, seeds, i)) ** 0.5 for i in range(len(accuracies[0]))]

# compute the standard deviation of accuracy for valid cases

avg_accuracies = [float(f"{acc:.3f}") for acc in avg_accuracies]
std_accuracies = [float(f"{acc:.3f}") for acc in std_accuracies]

table.append(avg_accuracies)
table.append(std_accuracies)

# Print the table
print(tabulate(table, tablefmt="grid"))
    