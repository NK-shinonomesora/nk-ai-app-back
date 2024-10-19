from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

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