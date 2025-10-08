## Imports
import torch
import os
from datasets import Features,Value, load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate

## Parameters
CHECKPOINT = "google-t5/t5-base"
DATA_PATH = "data/eng-fra.txt"
BATCH_SIZE = 16
DEVICE = 'cuda'
LR = 5e-5
NUM_EPOCHS= 10
MAX_TGT_LEN = 64
MAX_LEN = 64
NUM_BEAMS = 1
SRC_LANG = "English"
TGT_LANG = "French"
MODEL_SAVE_PATH = 'cktps/custom_t5.pt'

## Dataset
features = Features({"src":Value("string"),"tgt":Value("string")})
raw_dataset = load_dataset("csv",data_files={"full":DATA_PATH},delimiter="\t",column_names=["src", "tgt"],features=features)
raw_dataset = raw_dataset['full']
splits = raw_dataset.train_test_split(test_size=0.1, seed = 42)
train_raw, val_raw = splits['train'], splits['test']
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
def tokenize_function(x):
    x["src"] = [f"translate {SRC_LANG} to {TGT_LANG}: " + x["src"][i] for i in range(0,len(x["src"]))]
    return tokenizer(x["src"], text_target = x["tgt"],truncation=True,max_length=MAX_LEN) 
tokenized_train_dataset = train_raw.map(tokenize_function,batched=True)
tokenized_val_dataset = val_raw.map(tokenize_function,batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['src', 'tgt'])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(['src', 'tgt'])
tokenized_train_dataset.set_format("torch")
tokenized_val_dataset.set_format("torch")

## Training
start_epoch = 0
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)  
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model)
train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_val_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator)
optimizer = AdamW(model.parameters(), lr = LR)
scheduler = get_scheduler('linear', optimizer = optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS*len(train_dataloader))
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Resuming from checkpoint: {MODEL_SAVE_PATH}")
    state = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    start_epoch = state['epoch'] + 1
    print(f"Resumed at epoch {start_epoch}")
model.to(DEVICE)
progress_bar = tqdm(range((start_epoch-1)*len(train_dataloader),NUM_EPOCHS*len(train_dataloader)))

## Loop
for epoch in range(start_epoch,NUM_EPOCHS):
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
            labels = batch["labels"]
            batch = {k:v.to(DEVICE) for k,v in batch.items() if k!="labels"}
            gen_ids = model.generate(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],max_new_tokens=MAX_TGT_LEN,num_beams=NUM_BEAMS)
            predictions = tokenizer.batch_decode(gen_ids,skip_special_tokens=True)
            labels[labels== -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels,skip_special_tokens=True)
            metric.add_batch(predictions=predictions, references=[[r] for r in refs])
        bleu = metric.compute()
    print(f"epoch: {epoch} | sacrebleu: {bleu}") 
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch, 
    }     
    torch.save(state, MODEL_SAVE_PATH)