import os
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---- env: silence tokenizer fork warning (optional but nice)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ====== Paths (ABSOLUTE) ======
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# ====== Initialize App (mount static with a NAME) ======
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ====== Load Model (use merged dir if you chose Mode B) ======
MODEL_DIR = Path("/home/teja/three/vardhan/placement_projects/machine_translation/artifacts/merged_t5_fr")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at startup so uvicorn doesn't reload on import failures
tokenizer = None
model = None

@app.on_event("startup")
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    model.eval()

# ====== Translation Function ======
def translate_text(text: str) -> str:
    # Make sure the prefix matches what you trained with
    prompt = f"translate English to French: {text}".strip()
    batch = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=128,
            num_beams=4,                # nicer test-time quality
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ====== Routes ======
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "translated_text": None, "english_text": "", "use_beam": False, "latency_ms": None},
    )

@app.post("/translate", response_class=HTMLResponse)
async def translate(request: Request, english_text: str = Form(...)):
    translated_text = translate_text(english_text)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "english_text": english_text, "translated_text": translated_text, "use_beam": True},
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}
