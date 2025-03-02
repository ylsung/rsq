import re
import os
import sys
from tabulate import tabulate
from collections import defaultdict


folder = sys.argv[1]


files = [
    "30_total_documents_gold_at_0_score.txt",
    "30_total_documents_gold_at_14_score.txt",
    "30_total_documents_gold_at_29_score.txt",
]


accuracies = defaultdict(list)
lengths = defaultdict(list)


seeds = [0, 1, 2]

for seed in seeds:
    if "@" in folder:
        if "@0" in folder:
            new_folder = folder.replace("@0", f"@{seed}") 
        else:
            raise NotImplementedError("This script only supports starting with seed 0")
    else:
        # for old naming convention
        new_folder = folder + f"@{seed}"

    if seed == 0 and not os.path.exists(new_folder):
        new_folder = folder

    for _file in files:
        complete_filename = os.path.join(new_folder, _file)
        # print(completqe_filename)
        try:
            with open(complete_filename, "r") as f:
                lines = f.readlines()
                
                result = lines[-1]
                
                print(result)
                
                match = re.search(r"best_subspan_em:\s*([0-9]*\.?[0-9]+)", result)

                # Check if a match is found and print it
                if match:
                    accuracy = float(match.group(1))
                    
                    accuracy = float(f"{accuracy:.3f}")
                    
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
avg_accuracies = [float(f"{acc:.3f}") for acc in avg_accuracies]

table.append(avg_accuracies)

# Print the table
print(tabulate(table, tablefmt="grid"))
    