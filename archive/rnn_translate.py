## imports
import torch #type:ignore
import re
import unicodedata
import random
from torch.utils.data import Dataset, random_split, DataLoader #type:ignore
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

## Parameters
BOS = '<BOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'
BATCH_SIZE=128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
DEVICE = "cuda:0"
NUM_LAYERS = 4
MAX_LEN=20
generator = torch.Generator().manual_seed(50)
LR = 5e-4
## Data

# helper functions
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

# read file
def read_file(file_path="data/eng-fra.txt"):
    lang_one,lang_two = [],[]
    with open(file_path,encoding="utf-8") as f:
        for line in f:
            sent_one,sent_two = line.split("\t")
            sent_one = normalizeString(sent_one)
            sent_two = normalizeString(sent_two)
            lang_one.append(sent_one); lang_two.append(sent_two)
    return lang_one, lang_two
lang_one,lang_two = read_file()

# Dataset
class Translation_Dataset(Dataset):
    def __init__(self,lang_one,lang_two):
        self.src_lang = lang_one
        self.tgt_lang = lang_two
        self.src_vocab,self.rev_src_vocab = self.get_vocabulary(self.src_lang)
        self.tgt_vocab,self.rev_tgt_vocab = self.get_vocabulary(self.tgt_lang)

    def get_vocabulary(self,lang):
        vocab = {0:PAD, 1:BOS, 2:EOS, 3:UNK}
        rev_vocab = {PAD:0,BOS:1,EOS:2,UNK:3}
        key = 4
        for sent in lang:
            sent_ls = sent.split(' ')
            for tok in sent_ls:
                if tok not in rev_vocab:
                    vocab[key] = tok
                    rev_vocab[tok] = key
                    key += 1
        return vocab, rev_vocab
    
    def __len__(self): return len(self.src_lang)

    def __getitem__(self,idx):
        src_sent, tgt_sent = self.src_lang[idx], self.tgt_lang[idx]
        src_sent_tensor,tgt_sent_tensor = self.sent_to_tensor(src_sent,mode='src'), self.sent_to_tensor(tgt_sent,mode='tgt')
        return src_sent_tensor,tgt_sent_tensor
    
    def sent_to_tensor(self,sent,mode):
        if mode == 'src': rev_vocab = self.rev_src_vocab
        else: rev_vocab = self.rev_tgt_vocab
        result = [rev_vocab[BOS]]
        for word in sent.split(' '): result.append(rev_vocab.get(word,rev_vocab[UNK]))
        result.append(rev_vocab[EOS])
        return torch.tensor(result)

def collate_fn(batch):
    src_batch,tgt_batch = zip(*batch)
    padded_src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    padded_tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return padded_src_batch,padded_tgt_batch

full_dataset = Translation_Dataset(lang_one=lang_one, lang_two=lang_two)
src_vocab_size = len(full_dataset.src_vocab)
tgt_vocab_size = len(full_dataset.tgt_vocab)
train_dataset, test_dataset = random_split(full_dataset, [0.75, 0.25], generator= generator)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, collate_fn= collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, collate_fn= collate_fn)

## Model
class RNNEncoder(nn.Module):
    def __init__(self,input_size,embedding_dim,hidden_size,num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(self.input_size,embedding_dim,padding_idx=0)
        self.net = nn.GRU(embedding_dim,hidden_size,num_layers,batch_first=True)
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        embeddings = self.embedding_layer(x)
        _,h = self.net(embeddings,h_0)
        return h

class RNNDecoder(nn.Module):
    def __init__(self,input_size,embedding_dim,hidden_size,num_layers, max_len):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.init_hidden = nn.Linear(hidden_size, hidden_size)
        self.embedding_layer = nn.Embedding(self.input_size,embedding_dim,padding_idx=0)
        self.net = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.input_size)

    def forward(self,context_vector,targets=None):
        outputs = []
        context_vector = torch.tanh(self.init_hidden(context_vector))
        decoder_input = torch.empty(context_vector.size(1), 1,dtype=torch.long).fill_(1).to(device=context_vector.device)
        loop_len = targets.size(1)-1 if targets is not None else self.max_len
        for i in range(loop_len):
            decoder_output,context_vector = self.forward_step(decoder_input,context_vector)
            outputs.append(decoder_output)
            if targets is not None: ## Forced
                decoder_input = targets[:,i].unsqueeze(1)
            else: ##Unforced
                _, top1 = decoder_output.topk(1)
                decoder_input = top1.view(-1,1)
        outputs = torch.cat(outputs,dim=1)
        outputs = F.log_softmax(outputs,dim=2)
        return outputs
    
    def forward_step(self, decoder_input, context_vector):
        embed = self.embedding_layer(decoder_input)  # B, E
        decoder_output,context_vector = self.net(embed,context_vector)  # B,1,H  :: # L,B,H
        decoder_output = self.fc(decoder_output) # B,V
        return decoder_output, context_vector

encoder = RNNEncoder(input_size=src_vocab_size, hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM, num_layers=NUM_LAYERS).to(device=DEVICE)
decoder = RNNDecoder(input_size = tgt_vocab_size ,embedding_dim=EMBEDDING_DIM ,hidden_size = HIDDEN_SIZE,num_layers = NUM_LAYERS, max_len = MAX_LEN).to(device=DEVICE)

# train loop
criterion = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = LR)
for epoch in range(1):
    print(f'Epoch {epoch+1}')
    for i, (s,t) in enumerate(train_dataloader):
        s,t = s.to(DEVICE), t.to(DEVICE)
        h  = encoder(s)
        if i < len(train_dataloader)//2: out = decoder(context_vector = h, targets = t)
        else : out = decoder(context_vector = h)
        optimizer.zero_grad()
        t = t[:,1:]
        min_size = min(t.size(1),out.size(1))
        out = out[:,:min_size,:]
        t = t[:,:min_size]
        out,t = out.reshape(-1,tgt_vocab_size),t.reshape(-1)
        loss = criterion(out, t)
        loss.backward()
        optimizer.step()
        if i%100 == 0: print(f"{i}:{loss.item()}")

def tensor_to_sentence(tensor, rev_vocab):
    sentence = []
    for idx in tensor:
        word = rev_vocab.get(idx.item(), UNK)
        if word == EOS:
            break
        if word not in [BOS, PAD]:
            sentence.append(word)
    return ' '.join(sentence)

def output_to_sentence(out, rev_vocab):
    _, top_tokens = out.topk(1, dim=2)  # [B, T, 1]
    top_tokens = top_tokens.squeeze(2)  # [B, T]
    decoded_sentences = []
    for seq in top_tokens:
        decoded_sentences.append(tensor_to_sentence(seq, rev_vocab))
    return decoded_sentences

def evaluate_bleu(encoder, decoder, dataloader, dataset, device=DEVICE):
    encoder.eval()
    decoder.eval()
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    with torch.no_grad():
        for s, t in dataloader:
            s, t = s.to(device), t.to(device)
            h = encoder(s)
            out = decoder(context_vector=h)
            pred_sentences = output_to_sentence(out, dataset.tgt_vocab)
            true_sentences = []
            for seq in t: true_sentences.append(tensor_to_sentence(seq, dataset.tgt_vocab))
            for pred, ref in zip(pred_sentences, true_sentences):
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                if len(pred_tokens) == 0 or len(ref_tokens) == 0: continue
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(bleu)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if len(bleu_scores) > 0 else 0
    print(f"Average BLEU score on test set: {avg_bleu:.4f}")
    return avg_bleu
evaluate_bleu(encoder, decoder, test_dataloader, full_dataset, device=DEVICE)