## English → French Neural Machine Translation using T5

### Overview

This project implements a **Neural Machine Translation (NMT)** system that translates English sentences into French using the **Google T5 model**, fine-tuned on a bilingual dataset.

The implementation utilizes the **Hugging Face Transformers** library with **Parameter-Efficient Fine-Tuning (PEFT)** techniques. The English–French sentence pairs used for training are obtained from the [Tatoeba Project](http://tatoeba.org/home).

---

To build your own translation model, follow the steps below after cloning the repository.

1. **Download the Dataset**  
   Download the data from [here](https://www.manythings.org/anki/) and extract it to `data/eng-fra.txt`. The file is a **tab-separated text file** containing English–French sentence pairs:

```
I love apples.    J'aime les pommes.
How are you?      Comment ça va ?
This is a book.   C'est un livre.
```
Each line follows the structure:
```
<English sentence> \t <French sentence>
```
2. **Install Dependencies**  
Install all required dependencies using:
```bash
pip install -r requirements.txt
```
3. Train the LoRA Adapters
Train the T5 model with LoRA adapters using:
```bash
python -W ignore src/machine_translation.py
```
This process fine-tunes the T5 model with LoRA adapters attached to the Query and Value projection matrices for 10 epochs, using a learning rate of 5e-5 and a linear scheduler with no warmup steps.
The trained adapters are saved in the artifacts directory.
4. Assemble the Full Model
Integrate the trained adapters with the base T5 model using:
```bash
python -W ignore utils/form_artifacts.py
```
You can generate a sample translation from the trained model using the --sample argument:
```bash
python -W ignore utils/form_artifacts.py --sample "Hello, I am Vardhan!"
```
5. Deploy the Model via FastAPI
To serve the fine-tuned model using FastAPI, run:
```bash
cd hosting_dir
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

From limited training on 30,000 English–French sentence pairs, the model achieved a sacreBLEU score of 46.2 after fine-tuning, indicating strong translation performance for a lightweight setup.


## Planned Future Extensions
1. Extend to support to multiple languages
2. Include quantization for smaller memroy foot-print
