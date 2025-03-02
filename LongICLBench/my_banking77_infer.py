from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch
import argparse
import os
import json


import sys
import os
print(os.environ['CODEPATH'])
sys.path.append(f"{os.environ['CODEPATH']}/fake_quant")

from fake_quant.api import load_quantized_checkpoint


all_labels = [
            "activate_my_card",
            "age_limit",
            "apple_pay_or_google_pay",
            "atm_support",
            "automatic_top_up",
            "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit",
            "beneficiary_not_allowed",
            "cancel_transfer",
            "card_about_to_expire",
            "card_acceptance",
            "card_arrival",
            "card_delivery_estimate",
            "card_linking",
            "card_not_working",
            "card_payment_fee_charged",
            "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate",
            "card_swallowed",
            "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised",
            "change_pin",
            "compromised_card",
            "contactless_not_working",
            "country_support",
            "declined_card_payment",
            "declined_cash_withdrawal",
            "declined_transfer",
            "direct_debit_payment_not_recognised",
            "disposable_card_limits",
            "edit_personal_details",
            "exchange_charge",
            "exchange_rate",
            "exchange_via_app",
            "extra_charge_on_statement",
            "failed_transfer",
            "fiat_currency_support",
            "get_disposable_virtual_card",
            "get_physical_card",
            "getting_spare_card",
            "getting_virtual_card",
            "lost_or_stolen_card",
            "lost_or_stolen_phone",
            "order_physical_card",
            "passcode_forgotten",
            "pending_card_payment",
            "pending_cash_withdrawal",
            "pending_top_up",
            "pending_transfer",
            "pin_blocked",
            "receiving_money",
            "Refund_not_showing_up",
            "request_refund",
            "reverted_card_payment?",
            "supported_cards_and_currencies",
            "terminate_account",
            "top_up_by_bank_transfer_charge",
            "top_up_by_card_charge",
            "top_up_by_cash_or_cheque",
            "top_up_failed",
            "top_up_limits",
            "top_up_reverted",
            "topping_up_by_card",
            "transaction_charged_twice",
            "transfer_fee_charged",
            "transfer_into_account",
            "transfer_not_received_by_recipient",
            "transfer_timing",
            "unable_to_verify_identity",
            "verify_my_identity",
            "verify_source_of_funds",
            "verify_top_up",
            "virtual_card_not_working",
            "visa_or_mastercard",
            "why_verify_identity",
            "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal"
        ]

def select_data(given_dataset, number_of_turns):
    selected_data_list = []
    label_to_data_dict = {}
    for data in given_dataset:
        if data['label'] in label_to_data_dict:
            label_to_data_dict[data['label']].append(data)
        else:
            label_to_data_dict[data['label']] = [data]
    data_label_list = list(label_to_data_dict.keys())
    selected_label_to_count = {key:0 for key in data_label_list}
    for turn in range(number_of_turns):
        for i, key in enumerate(data_label_list):
            if len(label_to_data_dict[key]) > selected_label_to_count[key]:
                selected_data_list.append(label_to_data_dict[key][selected_label_to_count[key]])
                selected_label_to_count[key] += 1
            else:
                for other in range(i+1, len(data_label_list)):
                    other_key = data_label_list[other]
                    if len(label_to_data_dict[other_key]) > selected_label_to_count[other_key]:
                        selected_data_list.append(label_to_data_dict[other_key][selected_label_to_count[other_key]])
                        selected_label_to_count[other_key] += 1
                        break
    print("selected data list length: ", len(selected_data_list))
    return selected_data_list

def format_discovery_prompt(data_dict_list, round=0, with_instruction=False, context_token_number="2k"):
    token_shot_map_dict = {"2k": 77, "5k": 175, "10k": 346, "15k": 515, "20k": 685, "25k": 856,
                           "32k": 1101}
    prompt = 'Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format.'
    if with_instruction:
        prompt = prompt + 'You can only make prediction from the following categories: '
        for i, word in enumerate(all_labels):
            if i != len(all_labels) - 1:
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
    for data in data_dict_list[:index]:
        prompt = prompt + "service query: " + data['text'] + "\nintent category: " + all_labels[data['label']] + '\n'
    return prompt

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
parser.add_argument("-k", "--api_key", type=str, help="api key of open ai")
parser.add_argument("--instruct", action="store_true", help="whether to show all the labels as instruction")
parser.add_argument("--round", type=int, default=0, help="number of round for demonstration")
parser.add_argument("--test_number", type=int, help="number of examples to run for test")
parser.add_argument("--quantized_checkpoint", type=str, default=None, help="quantized checkpoint to load")
parser.add_argument("--rotated_quantized_checkpoint", type=str, default=None, help="rotated quantized checkpoint to load")
parser.add_argument("--postfix", type=str, default="", help="postfix for the output file")
args = parser.parse_args()

dataset = load_dataset("banking77")
train_data = dataset['train']
test_data = dataset['test']
demo_data = select_data(given_dataset=train_data, number_of_turns=args.round)
eva_data = select_data(given_dataset=test_data, number_of_turns=7)
total = 0
correct = 0

if args.model == "llama3-8b-instruct":
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

if args.rotated_quantized_checkpoint is not None:
    print(f"load rotated: {args.rotated_quantized_checkpoint}")
    model = load_quantized_checkpoint(
        model, args.rotated_quantized_checkpoint, rotate=True
    )
elif args.quantized_checkpoint is not None:
    print(f"load rotated: {args.quantized_checkpoint}")
    model = load_quantized_checkpoint(
        model, args.quantized_checkpoint, rotate=False
    )

model.to("cuda")

demo_prompt = format_discovery_prompt(demo_data, round=args.round, with_instruction=args.instruct, context_token_number=args.context_length)

if args.round != 0:
    if args.instruct:
        output_file = f'bank77_round_instruct_result/{args.model}_{args.round}{args.postfix}.json'
    else:
        output_file = f'bank77_round_result/{args.model}_{args.round}{args.postfix}.json'
else:
    if args.instruct:
        output_file = f'bank77_instruct_result/{args.model}_{args.context_length}{args.postfix}.json'
    else:
        output_file = f'bank77_result/{args.model}_{args.context_length}{args.postfix}.json'
if not os.path.exists(output_file.split('/')[0]):
    os.makedirs(output_file.split('/')[0])
with open(output_file, mode='w', encoding='utf-8') as f:
    feeds = []
    f.write(json.dumps(feeds, indent=2))

print(f"==========Evluation for {args.model}; Round {args.round}==============")

for example in eva_data[:args.test_number]:
    cur_prompt = demo_prompt + "service query: " + example['text'] + "\nintent category: "

    messages = [
        {"role": "user", "content": cur_prompt}
    ]
    
    message = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
    
    # directly use the inputs
    inputs = tokenizer(cur_prompt, return_tensors="pt").to('cuda')
    # input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
    #                                             return_tensors='pt')
    prompt_length = inputs.input_ids.size()[-1]
    output_ids = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)

    print("org response: ", response)
    temp_prompt = "intent category:"
    if example['text'] not in response:
        response = response.split("service query")[0].strip()
    else:
        response = list(response.split(example['text']))[-1].strip().split(temp_prompt)
        if len(response) > 1:
            response = response[1].split("service query")[0].strip()
        else:
            response = response[0].strip()
    response = response.strip().split("\n")[0]

    response = response.lower().strip()
    
    print("pred: ", response)
    label = all_labels[example['label']]
    label = label.lower()
    print("label: ", label)
    if response == label:
        correct += 1
    total += 1
    print("accuracy: ", correct/total)
    print("correct: ", correct)
    print("all: ", total)
    output_dict = {}
    output_dict['text'] = example['text']
    output_dict['label'] = label
    output_dict['pred'] = response
    feeds.append(output_dict)
    with open(output_file, mode='w', encoding='utf-8') as feedsjson:
        feedsjson.write(json.dumps(feeds, indent=2))
