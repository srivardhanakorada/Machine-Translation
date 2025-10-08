## Imports
import torch
from datasets import load_dataset,Features,Value
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

## Parameters
checkpoint = "google-t5/t5-base"
batch_size = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 5e-5
NUM_EPOCHS= 5
MAX_SRC_LEN = 64
MAX_TGT_LEN = 64
NUM_BEAMS = 4
SRC_LANG = "English"
TGT_LANG = "French"
DATA_PATH = "data/eng-fra.txt"

config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8, 
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.1,
    bias='none',
)

## Dataset
features = Features({"src":Value("string"),"tgt":Value("string")})
raw_dataset = load_dataset("csv",data_files={"full":DATA_PATH},delimiter="\t",column_names=["src", "tgt"],features=features)
raw_dataset = raw_dataset['full']
splits = raw_dataset.train_test_split(test_size=0.3, seed = 42)
train_raw, val_raw = splits['train'], splits['test']
tokenizer = AutoTokenizer.from_pretrained(checkpoint,use_fast=True)
def tokenize_function(x):
    pref = f"translate {SRC_LANG} to {TGT_LANG}: "
    return tokenizer([pref + s for s in x["src"]], 
                    text_target = x["tgt"],
                    truncation=True,max_length=MAX_SRC_LEN) 
tokenized_train_dataset = train_raw.map(tokenize_function,batched=True)
tokenized_val_dataset = val_raw.map(tokenize_function,batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['src', 'tgt'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['src', 'tgt'])
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

## Model
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model)
val_dataloader = DataLoader(tokenized_val_dataset, shuffle=False, batch_size=batch_size,collate_fn=data_collator,num_workers=2,pin_memory=True)
model.to(DEVICE)

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