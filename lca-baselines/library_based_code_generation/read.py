import re
import os
import sys
from tabulate import tabulate
from collections import defaultdict

import json



max_seed= int(sys.argv[1])
folder = sys.argv[2]

accuracies = defaultdict(list)
lengths = defaultdict(list)


seeds = range(max_seed)

# file_name = file_name.replace(start_task, task)
for seed in seeds:
    file_name = os.path.join(folder, "metadata.json")
    if "@" in file_name:
        if "@0" in file_name:
            new_file_name = file_name.replace("@0", f"@{seed}") 
        else:
            raise NotImplementedError("This script only supports starting with seed 0")
    else:
        # 16 bit
        # for old naming convention
        new_file_name = file_name
        # if seed == 0 and not os.path.exists(new_file_name):
        #     new_file_name = new_file_name
        # else:
        #     raise NotImplementedError("This script only supports starting with seed 0")

    complete_filename = new_file_name
    try:
        with open(complete_filename, "r") as f:
            data = json.load(f)
            
        accuracies[seed].append(data["metrics"]["ChrF"]["mean"])
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
names = ["library_code_generation"]
            
# Create the table
table = [names] 

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
    