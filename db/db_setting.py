from sqlmodel import Session, create_engine
from fastapi import Depends
from typing import Annotated

mysql_url = "mysql+pymysql://root:Salto0916_@localhost/nk_ai"

engine = create_engine(mysql_url)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]