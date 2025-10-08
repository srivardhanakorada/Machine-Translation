## Imports
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import bitsandbytes as bnb

## Parameters
checkpoint = "google-t5/t5-base"
batch_size = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 5e-5
NUM_EPOCHS= 10
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
NUM_BEAMS = 1
SRC_LANG = "English"
TGT_LANG = "Telugu"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_compute_dtype=torch.bfloat16
)

config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.1,
    bias='none'
)

## Dataset
raw_dataset = load_dataset('ai4bharat/BPCC', 'daily')
raw_dataset = raw_dataset['tel_Telu']
splits = raw_dataset.train_test_split(test_size=0.1, seed = 42)
train_raw, val_raw = splits['train'], splits['test']
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=True)
def tokenize_function(x):
    pref = f"translate {SRC_LANG} to {TGT_LANG}: "
    return tokenizer([pref + s for s in x["src"]], 
                    text_target = x["tgt"],
                    truncation=True,max_length=MAX_SRC_LEN) 
tokenized_train_dataset = train_raw.map(tokenize_function,batched=True)
tokenized_val_dataset = val_raw.map(tokenize_function,batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['src_lang', 'tgt_lang', 'src', 'tgt'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['src_lang', 'tgt_lang', 'src', 'tgt'])
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

## Model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, quantization_config = bnb_config, device_map='auto')
model = prepare_model_for_kbit_training(model=model,use_gradient_checkpointing=True)
model = get_peft_model(model, config)

total_parms, trainable_params = 0,0
for n,p in model.named_parameters():
    total_parms += p.numel()
    if p.requires_grad:
        trainable_params += p.numel()
print(f"Trainable params: {trainable_params/1e6:.2f}M / {total_parms/1e6:.2f}M " , f"({100*trainable_params/total_parms:.2f}%)")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model)
train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=batch_size, 
                            collate_fn=data_collator,num_workers=2,pin_memory=True)
val_dataloader = DataLoader(tokenized_val_dataset, shuffle=False, batch_size=batch_size, 
                            collate_fn=data_collator,num_workers=2,pin_memory=True)
model.to(DEVICE)

## Training
# QLoRA typically uses bitsandbytes paged optimizers to save memory:
# optimizer = bnb.optim.PagedAdamW32bit(
#     filter(lambda p: p.requires_grad, model.parameters()),  # only adapters
#     lr=LR
# )
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, 
                        num_training_steps=NUM_EPOCHS*len(train_dataloader))
progress_bar = tqdm(range(NUM_EPOCHS*len(train_dataloader)))

## Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in train_dataloader:
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()        
        scheduler.step()
        optimizer.zero_grad()   
        progress_bar.update(1)
## Evaluation
model.eval()
metric = evaluate.load('sacrebleu')
with torch.no_grad():
    for batch in val_dataloader:
        labels = batch["labels"].clone()
        batch = {k:v.to(DEVICE) for k,v in batch.items() if k!="labels"}
        gen_ids = model.generate(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
                                max_new_tokens=MAX_TGT_LEN,num_beams=NUM_BEAMS)
        predictions = tokenizer.batch_decode(gen_ids,skip_special_tokens=True)
        labels[labels== -100] = tokenizer.pad_token_id
        refs = tokenizer.batch_decode(labels,skip_special_tokens=True)
        metric.add_batch(predictions=predictions, references=[[r] for r in refs])
    bleu = metric.compute()
print(f" BLEU-Score: {bleu['score']:.2f}")