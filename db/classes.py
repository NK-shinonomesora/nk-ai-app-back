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

class InputMemo(BaseModel):
    title: str = Field(min_length=1, max_length=10)
    content: str = Field(min_length=1, max_length=10)
    @field_validator('title')
    def title_word_count_check(cls, v):
        if v == 'hello':
            raise PydanticCustomError(
                'not_a_hello',
                '"{wrong_value}"は使用禁止です！',
                dict(wrong_value=v),
            )
        return v

class Memo(SQLModel, table=True):
    id: str = ModelField(primary_key=True)
    title: str = ModelField(index=True)
    content: str
    created_at: int
    
class Memo_Annotation(SQLModel, table=True):
    id: Optional[int] = ModelField(primary_key=True)
    memo_id: int
    annotation_id: int

class Annotation_Master(SQLModel, table=True):
    id: Optional[int] = ModelField(primary_key=True)
    label: str
    word: str