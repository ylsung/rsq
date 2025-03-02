from longeval.utils import maybe_monkey_patch, load_testcases
import sys
import numpy as np

test_file = sys.argv[1]
            
num_correct = 0
avg_length = 0

test_cases = load_testcases(test_file)

correct_line_indices = []
for idx, test_case in enumerate(test_cases):
    
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"].strip()
    expected_number = test_case["expected_number"]
    
    
    prompt_list = prompt.split("\n")
    
    for i, p in enumerate(prompt_list):
        if correct_line == p:
            correct_line_indices.append(i)
            break
        
print(np.min(correct_line_indices), np.max(correct_line_indices), np.mean(correct_line_indices))