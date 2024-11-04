from fastapi import FastAPI, Request, status, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator
from transformers import pipeline
from torch.nn.functional import cosine_similarity
from typing import Dict, List, Annotated, Optional
from pydantic_core import ErrorDetails, PydanticCustomError
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sqlmodel import Field, Session, SQLModel, create_engine, select
import spacy
import uuid

mysql_url = "mysql+pymysql://root:Salto0916_@localhost/nk_ai"

engine = create_engine(mysql_url)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

# カスタムエラーメッセージ
CUSTOM_MESSAGES = {
    "string_too_short": "{input}は必須項目です。",
    "string_too_long": "{input}は{max_length}文字以下で入力してください。",
}

def convert_errors(
    e: ValidationError, custom_messages: Dict[str, str]
) -> List[ErrorDetails]:
    new_errors: List[ErrorDetails] = []
    for error in e.errors():
        custom_message = custom_messages.get(error['type'])
        if custom_message:
            ctx = error.get('ctx')
            input = error.get("loc")
            error['msg'] = (
                custom_message.format(input=input[1], **ctx) if ctx else custom_message
            )
        new_errors.append(error)
    return new_errors

app = FastAPI()

# 例外ハンドラをオーバーライド
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # ここでエラーメッセージを日本語に置換
    exc = convert_errors(e=exc, custom_messages=CUSTOM_MESSAGES)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc}),
    )

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
    text: str = Field(min_length=1, max_length=10)

    @field_validator('text')
    def text_must_not_equal_hello(cls, v):
        if v == 'hello':
            raise PydanticCustomError(
                'not_a_hello',
                '"{wrong_value}"は使用禁止です！',
                dict(wrong_value=v),
            )
        return v

class TwoInputMessage(BaseModel):
    firstText: str
    secondText: str

class Memo(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str = Field(index=True)
    content: str = Field(min_length=1, max_length=1000)

    @field_validator('title')
    def title_word_count_check(cls, v):
        if len(v) < 1 or len(v) > 11:
            raise PydanticCustomError(
                'not_a_title',
                'タイトルは1~10文字以下にしてください。',
                dict(wrong_value=v),
            )
        return v
    
class Memo_Annotation(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    memo_id: int
    annotation_id: int

class Annotation_Master(SQLModel, table=True):
    id: Optional[int] = Field(primary_key=True)
    label: str
    word: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

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

def ner_pred(doc):
    result = {}
    for ent in doc.ents:
        newData = {}
        newData[ent.text] = ent.label_
        result = {**result, **newData}
    return result

def named_entity_recognition(texts: List[str]):
    lastResult = {}
    nlp = spacy.load("./models/model-last")
    for text in texts:
        doc = nlp(text)
        result = ner_pred(doc)
        lastResult = {**lastResult, **result}
    return lastResult

@app.post("/memo/")
async def create_memo(memo: Memo, session: SessionDep) -> Memo:
    try:
        memo_id = uuid.uuid4()
        memo.id = memo_id
        session.add(memo)

        extractedWords = named_entity_recognition([memo.title, memo.content])
        annotationMasterIds = []
        for word, label in extractedWords.items():
            statement = select(Annotation_Master.id).where(
                Annotation_Master.label == label,
                Annotation_Master.word == word
                )
            id = session.exec(statement).all()
            annotationMasterIds.append(id)
        
        for annotation_id in annotationMasterIds:
            model = Memo_Annotation()
            model.memo_id = memo_id
            model.annotation_id = annotation_id
            session.add(model)
            
        session.commit()
        session.refresh(memo)
        session.refresh(model)

        return memo
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail="Failed to create memo") from e


@app.get("/memos/")
def read_memos(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Memo]:
    memos = session.exec(select(Memo).offset(offset).limit(limit)).all()
    return memos

@app.get("/memo/search/")
def search_memo(keyword: str, session: SessionDep,):
    extractedWords = named_entity_recognition([keyword])
    annotationMasters = []
    for word, label in extractedWords.items():
        statement = select(Annotation_Master.id, Annotation_Master.label).where(
            Annotation_Master.label == label,
            Annotation_Master.word == word
            )
        data = session.exec(statement).one()
        annotationMasters.append(data)
    
    memoAnnotations = session.exec(select(Memo_Annotation.memo_id, Memo_Annotation.annotation_id)).all()
    scores = {}
    for memoAnnotation in memoAnnotations:
        memo_id = memoAnnotation[0]
        annotation_id = memoAnnotation[1]
        for annotationMaster in annotationMasters:
            annotation_master_id = annotationMaster[0]
            annotation_label = annotationMaster[1]
            score = 0
            if annotation_id == annotation_master_id:
                score = 2
            else:
                statement = select(Annotation_Master.label).where(Annotation_Master.id == annotation_id)
                targetLabel = session.exec(statement).one()
                if targetLabel == annotation_label:
                    score = 1
            if memo_id in scores:
                scores[memo_id] = scores[memo_id] + score
            else:
                scores[memo_id] = score

    sortedScores = sorted(scores.items(), reverse=True, key=lambda x : x[1])
    topTwoScores = sortedScores[:3]
    
    memos = []

    for memoInfo in topTwoScores:
        memo_id = memoInfo[0]
        statement = select(Memo).where(Memo.id == memo_id)
        memo = session.exec(statement).one()
        memos.append(memo)

    return {"memos": memos}