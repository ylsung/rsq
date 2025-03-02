import os
import torch

from .example_generation_model import ExampleGenerationModel

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


import sys
import os
print(os.environ['CODEPATH'])
sys.path.append(f"{os.environ['CODEPATH']}/fake_quant")

from fake_quant.api import load_quantized_checkpoint



class HFChatModel(ExampleGenerationModel):

    def __init__(self, model_name: str, use_bm25: bool = False, args=None):
        self.model_name = model_name

        
        config = AutoConfig.from_pretrained(model_name)
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"
        
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16)
        
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
        
        model = model.eval()
        model.cuda()
        
        self.model = model
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.use_bm25 = use_bm25

    def generate(self, task_description: str, project_apis: list[str] = None) -> str:
        instruction = (
            self.get_prompt(task_description)
            if not self.use_bm25
            else self.get_bm25_prompt(task_description, project_apis)
        )
        
        prompt = [
            {"role": "user", "content": instruction},
        ]
        
        message = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True
                )
    
        # directly use the inputs
        inputs = self.tokenizer(message, return_tensors="pt").to('cuda')
        # input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
        #                                             return_tensors='pt')
        prompt_length = inputs.input_ids.size()[-1]
        output_ids = self.model.generate(**inputs, max_new_tokens=2048)
        response = self.tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)

        return response
        

    def name(self):
        if not self.use_bm25:
            return self.model_name
        else:
            return f"bm25/{self.model_name}"