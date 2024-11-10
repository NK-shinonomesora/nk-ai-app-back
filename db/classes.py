from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError
from sqlmodel import SQLModel
from sqlmodel import Field as ModelField
from typing import Optional

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
    id: str = ModelField(primary_key=True)
    title: str = ModelField(index=True)
    content: str = ModelField(min_length=1, max_length=1000)
    created_at: int

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
    id: Optional[int] = ModelField(primary_key=True)
    memo_id: int
    annotation_id: int

class Annotation_Master(SQLModel, table=True):
    id: Optional[int] = ModelField(primary_key=True)
    label: str
    word: str