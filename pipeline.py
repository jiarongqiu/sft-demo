import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, PeftModel

class SFTPipeline:

    MODEL_NAME="distilgpt2"
    PROMPT_TEMPLATE = (
        "Instruction: {instruction}\n"
        "{input_line}"
        "Response:{response}"
    )

    def __init__(self):
        pass
    
    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        tokenizer.add_special_tokens({
            'pad_token': tokenizer.eos_token,
            'eos_token': tokenizer.eos_token
        })
        self.tokenizer = tokenizer
    
    def build_model(self,use_lora=True):
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME, torch_dtype=torch.float32)
        model.resize_token_embeddings(len(self.tokenizer))
        if use_lora:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["c_attn"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(model, lora_config)
        else:
            self.model = model
    
    def format_prompt(self, instruction, input_text=None, response=""):   
        input_line = f"Input: {input_text}\n" if input_text and input_text.strip() else ""
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_line=input_line,
            response=response
        )
    
    def predict(self, instruction, input_text="", max_new_tokens=100, temperature=0.7, top_p=0.9, top_k=50, no_repeat_ngram_size=3):
        prompt = self.format_prompt(instruction, input_text, response="")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cpu")
        attention_mask = encoded["attention_mask"].to("cpu")
        prompt_length = input_ids.shape[-1]
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True
        )
        outputs = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        return outputs
    
    def build_dataset(self,path):
        def preprocess_function(example):
            instruction = example["instruction"]
            output = example["output"]
            input_field = example.get("input", "")
            prompt = self.format_prompt(instruction, input_field, response=output)
            tokenized = self.tokenizer(prompt, truncation=True, max_length=256)
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        dataset = load_dataset("json", data_files=path)
        dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
        return dataset
    
    def train(self, datasets,num_train_epochs=3, batch_size=4):
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir='./ckpts',
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="steps",
            eval_steps=50,
            logging_steps=10,
            save_steps=50,
            fp16=False
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        self.trainer.train()
    
    