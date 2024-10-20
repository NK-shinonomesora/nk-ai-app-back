from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from torch.nn.functional import cosine_similarity

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chat(BaseModel):
    message: str

class InputMessage(BaseModel):
    text: str

class TwoInputMessage(BaseModel):
    firstText: str
    secondText: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/chat/")
async def create_chat(message: Chat):
    return message

@app.post("/document_classification")
async def sentimentanalysis(input: InputMessage):
    text_classification_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-marc_ja")
    # textの極性を予測
    result = text_classification_pipeline(input.text)[0]
    return { "text": input.text, "label": result["label"], "score": '{:.2%}'.format(result["score"]) }

@app.post("/natural_language_inference")
async def natural_language_inference(inputs: TwoInputMessage):
    nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
    result = nli_pipeline({"text": inputs.firstText, "text_pair": inputs.secondText})
    label = result["label"]
    jaLabel = ""

    if label == "entailment":
        jaLabel = "含意"
    elif label == "contradiction":
        jaLabel = "矛盾"
    else:
        jaLabel = "中立"

    return {
        "text": "1.  " + inputs.firstText + " / 2. " + inputs.secondText,
        "label": jaLabel,
        "score": '{:.2%}'.format(result["score"])
    }

@app.post("/semantic_textual_similarity")
async def semantic_textual_similarity(inputs: TwoInputMessage):
    sim_enc_pipeline = pipeline(
        model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",
        task="feature-extraction"
    )

    text_emb = sim_enc_pipeline(inputs.firstText, return_tensors=True)[0][0]
    sim_emb = sim_enc_pipeline(inputs.secondText, return_tensors=True)[0][0]

    # textとsim_textの類似度を計算
    sim_pair_score = cosine_similarity(text_emb, sim_emb, dim=0)

    score = (sim_pair_score.item() + 1.0) / 2.0
    
    return {
        "text": "1.  " + inputs.firstText + " / 2. " + inputs.secondText,
        "label": "-",
        "score": '{:.2%}'.format(score)
    }
