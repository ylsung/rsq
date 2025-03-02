import re

import numpy as np

from collections import Counter

from longeval.utils import load_testcases

def fetch_number(input_string, target_start, target_end):
    match = re.search(rf'{target_start}\s*(.*?){target_end}', input_string)

    if match:
        label_number = match.group(1)
    else:
        print("Label number not found")
        label_number = "none"
    
    return label_number
    

def file_open(file):
    with open(file, "r") as f:
        return f.readlines()

file_16bit = "new_cases/lines/predictions/Meta-Llama-3-8B-Instruct/w_16_a_16_kv_16/620_s0_e155_response.txt"
file_4bit = "new_cases/lines/predictions/Meta-Llama-3-8B-Instruct/quarot_4bit_w_n1024_l1024_w_16_a_16_kv_16/620_s0_e155_response.txt"


lines_16bit = file_open(file_16bit)
lines_4bit = file_open(file_4bit)

label_list = []
pred_16bit_list = []
pred_4bit_list = []

for line_16bit, line_4bit in zip(lines_16bit, lines_4bit):
    if line_16bit.startswith("Label"):
        assert fetch_number(line_16bit, "Label:", ",") == fetch_number(line_4bit, "Label:", ",")
        
        label = fetch_number(line_16bit, "Label:", ",")
        
        pred_16bit = fetch_number(line_16bit, "Parsed:", ",")
        pred_4bit = fetch_number(line_4bit, "Parsed:", ",")
        
        print(label, pred_16bit, pred_4bit)
        
        label_list.append(label)
        pred_16bit_list.append(pred_16bit)
        pred_4bit_list.append(pred_4bit)

label_array = np.array(label_list)
pred_16bit_array = np.array(pred_16bit_list)
pred_4bit_array = np.array(pred_4bit_list)

pos_1 = np.where((pred_16bit_array == label_array) & (pred_4bit_array != label_array) & (pred_4bit_array != "-1"))[0]
pos_2 = np.where((pred_16bit_array == label_array) & (pred_4bit_array == "-1"))[0]
pos_3 = np.where((pred_16bit_array == label_array) & (pred_4bit_array == label_array))[0]


# print(Counter(label_list))
# print(Counter(pred_16bit_list))
# print(Counter(pred_4bit_list))
# # import pdb; pdb.set_trace()

print("both correct: ", ((label_array == pred_4bit_array) & (label_array == pred_16bit_array)).mean())
print("16 bit correct, 4 bit incorrect: ", ((label_array != pred_4bit_array) & (label_array == pred_16bit_array)).mean())
print("16 bit incorrect, 4 bit correct: ", ((label_array == pred_4bit_array) & (label_array != pred_16bit_array)).mean())
print("both incorrect: ", ((label_array != pred_4bit_array) & (label_array != pred_16bit_array)).mean())

print("16 bit fail to predict: ", (pred_16bit_array == "-1").mean())
print("4 bit fail to predict: ", (pred_4bit_array == "-1").mean())

test_file = "new_cases/lines/testcases/620_s0_e155_lines.jsonl"
            
test_cases = load_testcases(test_file)

# with open("sample_16correct_4wrong.txt", "w") as f:
#     f.write(test_cases[pos_1[-1]]["prompt"])
#     print(test_cases[pos_1[-1]]["correct_line"])
    
# with open("sample_16correct_4notfound.txt", "w") as f:
#     f.write(test_cases[pos_2[-1]]["prompt"])
#     print(test_cases[pos_2[-1]]["correct_line"])

print(pos_3)
with open("sample_bothcorrect.txt", "w") as f:
    f.write(test_cases[pos_3[-1]]["prompt"])
    print(test_cases[pos_3[-1]]["correct_line"])