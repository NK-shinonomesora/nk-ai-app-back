from typing import List
from pydantic_core import ErrorDetails, PydanticCustomError
from pydantic import ValidationError, field_validator

# カスタムエラーメッセージ
CUSTOM_MESSAGES = {
    "string_too_short": "{input}は必須項目です。",
    "string_too_long": "{input}は{max_length}文字以下で入力してください。",
}

JAPANESE_MESSAGE = {
    "title": "タイトル",
    "content": "内容",
}

def convert_errors(e: ValidationError) -> List[ErrorDetails]:
    new_errors: List[ErrorDetails] = []
    for error in e.errors():
        custom_message = CUSTOM_MESSAGES.get(error['type'])
        if custom_message:
            ctx = error.get('ctx')
            input = error.get("loc")

            error['msg'] = (
                custom_message.format(input=JAPANESE_MESSAGE.get(input[1]), **ctx) if ctx else custom_message
            )
        new_errors.append(error)
    return new_errors

