from .utils import *
from .simple_generation import simple_generation

def test_lines_one_sample_two_models(model, guide_model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]
        
    if "Llama-3" in args.model_name_or_path and "Instruct" in args.model_name_or_path:
        # sys_prompt = "Below is a record of our previous conversation on many different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say 'I would like to discuss the topic of <TOPIC>'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first/second/third topic names. Now the record start."
        sys_prompt = ""

        text_inputs = ""
        text_inputs += "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        text_inputs += "\n" + sys_prompt
        text_inputs += "<|eot_id|><|start_header_id|>user<|end_header_id|>"
        text_inputs += "\n" + prompt
        text_inputs += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        device = getattr(model, "device", "cpu")
        
        if args.generate_with_prefix:
            extended_text_inputs = text_inputs + "According to the record, the <REGISTER_CONTENT> in"
        else:
            extended_text_inputs = text_inputs
        
        inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.size()[-1]
        extented_inputs = tokenizer(extended_text_inputs, return_tensors="pt").to(device)
        # sample = model.generate(**extented_inputs, do_sample=False, max_new_tokens=100)
        # output = tokenizer.decode(sample[0][prompt_length:])
        
        my_sample = simple_generation(
            model, guide_model, guide_type=args.guide_type, 
            apply_module=args.apply_module, top_k=args.top_k, 
            using_distorted_guide_layer=args.using_distorted_guide_layer,
            magnitude_dim=args.magnitude_dim,
            **extented_inputs, max_new_tokens=100
        )
        my_output = tokenizer.decode(my_sample[0][prompt_length:])
        
        output = my_output

    elif "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        from transformers import pipeline
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
        # Use next word prediction to get storywriter answer
        prompt += f'Line <{test_case["random_idx"][0]}>: <REGISTER_CONTENT> is'
        prompt_length = len(tokenizer(prompt).input_ids)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=15, do_sample=True, use_cache=True)[0]['generated_text'][len(prompt):]
    elif "THUDM/chatglm" in args.model_name_or_path:
        prompt_length = len(tokenizer(prompt).input_ids)
        output, _ = model.chat(tokenizer, prompt, history=[], max_length=16384)
    elif "gpt-" in args.model_name_or_path:
        prompt_length, output = retrieve_from_openai(prompt, args.model_name_or_path)
    elif "claude" in args.model_name_or_path:
        prompt_length, output = retrieve_from_anthropic(prompt, args.model_name_or_path)
    else:
        if "longchat" in args.model_name_or_path:
            conv = get_conversation_template("vicuna")
        else:
            conv = get_conversation_template(args.model_name_or_path)
        print(f"Using conversation template: {conv.name}")

        if "mpt-30b-chat" in args.model_name_or_path or "mpt-7b-8k-chat" in args.model_name_or_path:
            prompt += f'Answer in the format <{test_case["random_idx"][0]}> <REGISTER_CONTENT>.'
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.shape[-1]
        
        # Disable use_cache if using longchat models with flash attention
        use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)

        device = getattr(model, "device", "cpu")
        
        output = model.generate(input.input_ids.to(device), max_new_tokens=100, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)
    if idx ==0:
        with open(output_file, "w") as f:
            try:
                f.write(summary)
                f.write("\n")
            except:
                f.write(f"Label: {expected_number}, Predict: -1, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' '))
                f.write("\n")
    else:
        with open(output_file, "a+") as f:
            try:
                f.write(summary)
                f.write("\n")
            except:
                f.write(f"Label: {expected_number}, Predict: -1, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' '))
                f.write("\n")
    
    return expected_number == response_number, prompt_length, summary
