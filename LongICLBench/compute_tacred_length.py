from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, TextStreamer
from huggingface_hub import hf_hub_download
import os
import torch
import argparse
import json

import numpy as np


selected_labels = ['org:founded_by', 'per:employee_of', 'org:alternate_names', 'per:cities_of_residence', 'per:children', 'per:title', 'per:siblings', 'per:religion', 'per:age', 'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 'org:top_members/employees', 'per:countries_of_residence', 'org:city_of_headquarters', 'org:members', 'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters', 'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation', 'per:other_family', 'per:stateorprovince_of_birth', 'org:dissolved', 'per:date_of_death', 'org:shareholders', 'per:alternate_names', 'per:parents', 'per:schools_attended', 'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded', 'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 'per:country_of_death']

def select_data(given_dataset, number_of_turns):
    turns = 0
    label_list = []
    selected_data_list = []
    for data in given_dataset:
        if data['relation'] not in label_list and data['relation'] in selected_labels:
            selected_data_list.append(data)
            label_list.append(data['relation'])
        if len(label_list) == len(selected_labels):
            turns += 1
            if turns == number_of_turns:
                break
            else:
                label_list = []
    return selected_data_list

def select_test(given_dataset, number_of_turns):
    selected_data_list = []
    count_dict = {rela: 0 for rela in selected_labels}
    print("==========")
    print(len(given_dataset))
    for data in given_dataset:
        if data['relation'] in selected_labels and count_dict[data['relation']] < number_of_turns:
            selected_data_list.append(data)
            count_dict[data['relation']] += 1
    return selected_data_list

def format_discovery_prompt(data_dict_list, with_instruction=False, round=0, context_token_number="2k", group=False):
    token_shot_map_dict = {"600": 5, "2k": 25, "5k": 67, "10k": 133, "15k": 204, "20k": 270, "25k": 362,
                           "32k": 421}
    prompt = 'Given a sentence and a pair of subject and object entities within the sentence, please predict the relation between the given entities.'
    if with_instruction:
        prompt = prompt + " The predicted relationship must come from these classes: "
        for i, word in enumerate(selected_labels):
            if i != len(selected_labels) - 1:
                prompt = prompt + word + ', '
            else:
                prompt = prompt + word + '.\n'
    prompt = prompt + ' The examples are as follows: \n'
    if round != 0:
        index = len(data_dict_list)
        print(f"======={round} round running========")
        print("number of instances: ", index)
    else:
        index = token_shot_map_dict[context_token_number]

    data_list = data_dict_list[:index]
    print("org data_list: ", data_list)
    if group:
        print("==============demo grouped==============")
        data_list = sorted(data_list, key=lambda d: d['relation'])
        print("after grouping data_list: ", data_list)

    position_number_record = {}
    pos = 0
    for data in data_list:
        pos += 1
        if data["relation"] not in position_number_record:
            position_number_record[data["relation"]] = {}
            position_number_record[data["relation"]]["number"] = 1
            position_number_record[data["relation"]]["pos"] = [pos]
        else:
            position_number_record[data["relation"]]["number"] += 1
            position_number_record[data["relation"]]["pos"].append(pos)
    print("position_number_record: ", position_number_record)
    for data in data_list:
        prompt = prompt + "sentence: " + data['sentence'] + '\n'
        prompt = prompt + "the subject is " + data["subject_entity"] + " and the object is " + data["object_entity"] + '\n'
        prompt = prompt + "the relation between the two entities is: " + data["relation"] + '\n'
    return prompt, position_number_record

def generate_text(project_id: str, location: str, prompt: str, model) -> str:
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Query the model
    responses = model.generate_content(prompt,
                                       generation_config=generation_config,
                                       stream=False)
    for response in responses:
        return response.text



parser = argparse.ArgumentParser(description="Long in-context Learning",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--context_length", type=str, default='2k', help="number of tokens the context have")
parser.add_argument("-m", "--model", type=str, help="model name to test")
parser.add_argument("-g", "--group", action="store_true", help="whether to group the type of demonstration")
parser.add_argument("-k", "--api_key", type=str, help="api key of open ai")
parser.add_argument("--test_number", type=int, help="number of examples to run for test")
parser.add_argument("--round", type=int, default=0, help="number of round for demonstration")
parser.add_argument("--instruct", action="store_true", help="whether to show all the labels as instruction")
parser.add_argument("--quantized_checkpoint", type=str, default=None, help="quantized checkpoint to load")
parser.add_argument("--rotated_quantized_checkpoint", type=str, default=None, help="rotated quantized checkpoint to load")
parser.add_argument("--postfix", type=str, default="", help="postfix for the output file")
args = parser.parse_args()

test_file = open('processed_data/test_tacred.json')
test_data = json.load(test_file)
train_file = open('processed_data/train_tacred.json')
train_data = json.load(train_file)
demo_data = select_data(given_dataset=train_data, number_of_turns=args.round)
eva_data = select_test(given_dataset=test_data, number_of_turns=20)



if args.model == "llama3-8b-instruct":
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

demo_prompt, position_number_record = format_discovery_prompt(demo_data, with_instruction=args.instruct,
                                                              context_token_number=args.context_length,
                                                              group=args.group, round=args.round)

total_label = 0
total_pred = 0
total_correct = 0

if args.round != 0:
    if args.instruct:
        output_file = f'tacred_round_instruct_result/{args.model}_{args.round}{args.postfix}.json'
    elif args.group:
        output_file = f'tacred_round_group_result/{args.model}_{args.round}{args.postfix}.json'
    else:
        output_file = f'tacred_round_result/{args.model}_{args.round}{args.postfix}.json'
else:
    if args.instruct:
        output_file = f'tacred_instruct_result/{args.model}_{args.round}{args.postfix}.json'
    elif args.group:
        output_file = f'tacred_group_result/{args.model}_{args.context_length}{args.postfix}.json'
    else:
        output_file = f'tacred_result/{args.model}_{args.context_length}{args.postfix}.json'
if not os.path.exists(output_file.split('/')[0]):
    os.makedirs(output_file.split('/')[0])

with open(output_file, mode='w', encoding='utf-8') as f:
    feeds = []
    f.write(json.dumps(feeds, indent=2))


lengths = []
print(f"==========Evluation for {args.model}; Round {args.round}==============")
for example in eva_data[:args.test_number]:
    cur_prompt = demo_prompt + "sentence: " + example['sentence'] + '\n'
    cur_prompt = cur_prompt + "the subject is " + example["subject_entity"] + " and the object is " + example["object_entity"] + '\n'
    cur_prompt = cur_prompt + "the relation between the two entities is: "
    
    processed_chat = demo_prompt.split("sentence:")
    
    system_prompt = processed_chat.pop(0).strip()
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    for chat in processed_chat:
        chat = chat.split("the relation between the two entities is:")
        messages.append(
            {"role": "user", "content": "sentence: " + chat[0].strip()}
        )
        messages.append(
            {"role": "assistant", "content": "the relation between the two entities is: " + chat[1].strip()}
        )
        
    messages.append(
        {"role": "user", 
         "content": "sentence: " + example['sentence'] + "\n" + "the subject is " + example["subject_entity"] + " and the object is " + example["object_entity"]
        }
    )
    
    message = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
    
    message += "the relation between the two entities is: "
    
    # directly use the inputs
    inputs = tokenizer(message, return_tensors="pt").to('cuda')
    # input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
    #                                             return_tensors='pt')
    prompt_length = inputs.input_ids.size()[-1]
    
    
    lengths.append(prompt_length)
    
print(np.mean(lengths))
