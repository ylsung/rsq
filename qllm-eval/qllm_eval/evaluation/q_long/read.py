import re
import os
import sys
from tabulate import tabulate
from collections import defaultdict


max_seed= int(sys.argv[1])
folder = sys.argv[2]


files = ["300_response.txt", "460_response.txt", "620_response.txt", # "770_response.txt", \
    "620_s0_e155_response.txt", "620_s156_e310_response.txt", "620_s311_e465_response.txt", "620_s466_e620_response.txt"]


accuracies = defaultdict(list)
lengths = defaultdict(list)


seeds = range(max_seed)

for seed in seeds:
    if "@" in folder:
        if "@0" in folder:
            new_folder = folder.replace("@0", f"@{seed}") 
        else:
            raise NotImplementedError("This script only supports starting with seed 0")
    else:
        # for old naming convention
        new_folder = folder.replace("_w_16_a_16_kv_16", f"@{seed}_w_16_a_16_kv_16") 

    if seed == 0 and not os.path.exists(new_folder):
        new_folder = folder

    for _file in files:
        complete_filename = os.path.join(new_folder, _file)
        # print(completqe_filename)
        try:
            with open(complete_filename, "r") as f:
                lines = f.readlines()
                
                result = lines[-1]
                
                match = re.search(r"accuracy:\s*([0-9]*\.?[0-9]+)", result)

                # Check if a match is found and print it
                if match:
                    accuracy = float(match.group(1))
                    
                    accuracies[seed].append(accuracy)
                else:
                    accuracies[seed].append(-1)
                    
                # Regular expression to find a floating point number following 'length'
                match = re.search(r"length\s*([0-9]*\.?[0-9]+)", result)

                # Check if a match is found and print it
                if match:
                    length = float(match.group(1))
                    length = f"{length:2f}"
                    lengths[seed].append(length)
                else:
                    lengths[seed].append(-1)
        except:
            accuracies[seed].append(-1)
            lengths[seed].append(-1)


names = ["_".join(f.split("_")[:-1]) for f in files]
            
# Create the table
table = [names, lengths[0]] 

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
    