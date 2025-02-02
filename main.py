from fastapi import FastAPI, Request, status, Query, HTTPException, Cookie
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from torch.nn.functional import cosine_similarity
from typing import List, Annotated, Dict, Union
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
import spacy
from db.db_setting import SessionDep
from validation.input_validation import convert_errors
from db.classes import *
import time
import uuid
from utils.password_hash import get_hash_string
from utils.random_str import get_random_string

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 例外ハンドラをオーバーライド
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # ここでエラーメッセージを日本語に置換
    exc = convert_errors(e=exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc}),
    )

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
async def create_memo(memo: InputMemo, session: SessionDep):
    try:
        memo_id = uuid.uuid4()
        memo_data = {
            "id": memo_id,
            "title": memo.title,
            "content": memo.content,
            "created_at": int(time.time())
        }
        memo = Memo(**memo_data)
        session.add(memo)

        extractedWords = named_entity_recognition([memo.title, memo.content])

        if not extractedWords:
            session.commit()
            session.refresh(memo)
            return

        annotationMasterIds = []
        for word, label in extractedWords.items():
            statement = select(Annotation_Master.id).where(
                Annotation_Master.label == label,
                Annotation_Master.word == word
            )
            id = session.exec(statement).all()
            if id:
                annotationMasterIds.append(id)

        if not annotationMasterIds:
            session.commit()
            session.refresh(memo)
            return
        
        for annotation_id in annotationMasterIds:
            model = Memo_Annotation()
            model.memo_id = memo_id
            model.annotation_id = annotation_id
            session.add(model)
            
        session.commit()
        session.refresh(memo)
        session.refresh(model)
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=400, detail="Failed to create memo") from e


@app.get("/memo/detail/")
def read_memo(
    memo_id: str,
    session: SessionDep,
) -> Dict[str, Memo]:
    memo = session.get(Memo, memo_id)
    if memo is None:
        raise HTTPException(status_code=404, detail="存在しないページにアクセスしました。")
    return {"memo": memo}

@app.get("/memos/")
def read_memos(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Memo]:
    memos = session.exec(select(Memo).offset(offset).limit(limit)).all()
    return memos

@app.get("/memo/search/")
def search_memo(keyword: str, session: SessionDep):
    extractedWords = named_entity_recognition([keyword])

    if not extractedWords:
        return

    annotationMasters = []
    for word, label in extractedWords.items():
        statement = select(Annotation_Master.id, Annotation_Master.label).where(
            Annotation_Master.label == label,
            Annotation_Master.word == word
        )
        try:
            data = session.exec(statement).one()
        except Exception as e:
            return
        
        annotationMasters.append(data)
            
    if not annotationMasters:
        return
    
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
    topTwoScores = sortedScores[:5]
    
    memos = []

    for memoInfo in topTwoScores:
        memo_id = memoInfo[0]
        statement = select(Memo).where(Memo.id == memo_id)
        memo = session.exec(statement).one()
        memos.append(memo)

    return {"memos": memos}

@app.delete("/memo/{memo_id}")
async def delete_memo(
    memo_id: str,
    session: SessionDep,
):
    try:
        memo = session.get(Memo, memo_id)
        statement = select(Memo_Annotation).where(Memo_Annotation.memo_id == memo_id)
        memo_annotations = session.exec(statement).all()

        for memo_annotation in memo_annotations:
            session.delete(memo_annotation)

        session.delete(memo)
        session.commit()
    except Exception as e:
        print(e)
        session.rollback()
        raise HTTPException(status_code=400, detail="Failed to delete memo") from e

# @app.post("/login/")
# def login(user: InputUser, session: SessionDep) -> str:
#     hash_password = get_hash_string(user.password)
#     statement = select(User).where(User.id == user.id, User.password == hash_password)
#     user_model = session.exec(statement).first()
#     if user_model is None:
#         raise HTTPException(status_code=401, detail="ログイン認証エラー")
#     user_model.session_id = get_hash_string(get_random_string(30))
#     user_model.session_id_created_at = int(time.time())
#     session.add(user_model)
#     session.commit()
#     return user_model.session_id


# @app.get("/auth/{user_id}")
# def auth(user_id: str, session: SessionDep) -> Dict[str, str]:
#     statement = select(User).where(User.id == user_id)
#     user = session.exec(statement).first()
#     if user is None:
#         return { "session_id": "" }
#     if user.session_id is None:
#         return { "session_id": "" }
#     return {"session_id": user.session_id }


# @app.get("/session/{session_id}")
# def auth(session_id: str, session: SessionDep) -> Dict[str, str]:
#     statement = select(User).where(User.session_id == session_id)
#     user = session.exec(statement).first()
#     if user is None:
#         return { "user_id": "" }
#     return {"user_id": user.id }