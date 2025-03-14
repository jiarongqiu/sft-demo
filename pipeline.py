import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import evaluate
bleu = evaluate.load("bleu")

class SFTPipeline:

    MODEL_NAME = "distilgpt2"
    PROMPT_TEMPLATE = """
        ### Instruction:
        {instruction}
        ### Input:
        {input_line}
        ### Response:
        {response}
    """

    def __init__(self):
        pass
    
    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
    
    def build_model(self, quantized=False, use_lora=False):
        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float32
        )
        model.resize_token_embeddings(len(self.tokenizer))
        
        # if quantized:
        #     model.to("cpu")
        #     model = torch.quantization.quantize_dynamic(
        #         model,
        #         {torch.nn.Linear},
        #         dtype=torch.qint8
        #     )
        
        if use_lora:
            lora_config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(model, lora_config)
        else:
            self.model = model
    
    def format_prompt(self, instruction, input_text=None, response=""):
        input_line = f"{input_text}\n" if input_text and input_text.strip() else ""
        return self.PROMPT_TEMPLATE.format(
            instruction=instruction,
            input_line=input_line,
            response=response
        )
    
    def predict(self, instruction, input_text=""):
        prompt = self.format_prompt(instruction, input_text, response="")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return outputs
    
    def build_dataset(self, paths):
        def preprocess_function(examples):
            instructions = examples["instruction"]
            if isinstance(instructions[0], list):
                instructions = [" ".join(instr) for instr in instructions]
            
            outputs = examples["output"]
            if isinstance(outputs[0], list):
                outputs = [" ".join(output) for output in outputs]
            
            input_fields = examples.get("input", [""] * len(instructions))
            if isinstance(input_fields[0], list):
                input_fields = [" ".join(inp) for inp in input_fields]
            
            prompts = [
                self.format_prompt(instr, inp, response=out)
                for instr, inp, out in zip(instructions, input_fields, outputs)
            ]
            
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors=None
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        dataset = load_dataset("json", data_files=paths)
        
        for split in dataset.keys():
            dataset[split] = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=["instruction", "output", "input"]
            )
        
        self.dataset = dataset
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        if predictions.ndim > 2:
            predictions = predictions.argmax(axis=-1)
        labels[labels == -100] = self.tokenizer.pad_token_id
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu_score = bleu.compute(predictions=predictions, references=[[label] for label in labels])
        return {"bleu": bleu_score["bleu"]}
    
    def train(self, num_epochs=1, batch_size=4):
        training_args = TrainingArguments(
            output_dir='./ckpts',
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=50,
            save_steps=100,
            fp16=False,
            save_total_limit=2,
            evaluation_strategy="steps",
            eval_steps=50  
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"], 
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()