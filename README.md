## English → French Neural Machine Translation using T5

### Overview

Implemented a **Machine Translation model** that translates English sentences into French using **Google T5 model** fine-tuned on a bilingual dataset.

We used  **Hugging Face** in built library **Transformers**  for **PEFT** enabled finetuning.

---

To build you own model you could follow below steps after cloning our repository.


1. Install all dependencies using pip:

```bash
pip 
```

---

### 📁 Dataset

The dataset used is a **tab-separated text file** (`eng-fra.txt`) containing English–French sentence pairs:

```
I love apples.    J'aime les pommes.
How are you?      Comment ça va ?
This is a book.   C'est un livre.
```

Each line has:

```
<English sentence> \t <French sentence>
```
---

### ⚙️ Model Configuration

| Parameter    | Description               | Example               |
| ------------ | ------------------------- | --------------------- |
| `CHECKPOINT` | Pretrained model name     | `"google-t5/t5-base"` |
| `SRC_LANG`   | Source language           | `"English"`           |
| `TGT_LANG`   | Target language           | `"French"`            |
| `BATCH_SIZE` | Batch size                | `16`                  |
| `LR`         | Learning rate             | `5e-5`                |
| `NUM_EPOCHS` | Number of training epochs | `10`                  |
| `MAX_LEN`    | Max sequence length       | `64`                  |

---

### 🧠 Training Details

1. **Data Loading & Tokenization**
   The text pairs are tokenized using the T5 tokenizer with the prefix:

   ```
   translate English to French:
   ```

   which activates T5’s translation mode.

2. **Model Fine-Tuning**
   The `AutoModelForSeq2SeqLM` class is used to fine-tune the base T5 model with AdamW optimization and a linear learning-rate scheduler.

---

### 🔍 Evaluation Example

After training, you can test translation interactively:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
state = torch.load("cktps/custom_t5.pt", map_location="cpu")
model.load_state_dict(state['model'])

text = "translate English to French: How are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### 📈 Results

The model achieves a **sacreBLEU score of 46.2** after fine-tuning on 30k sentence pairs for 10 epochs.

---

