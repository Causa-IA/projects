from pydantic import BaseModel

class RagQuery(BaseModel):
    query: str