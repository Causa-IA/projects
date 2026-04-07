from fastapi import APIRouter
from model.rag_model import RagQuery
from services.rag_service import medical_rag_query

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query")
async def rag_query(datos: RagQuery):
    resultado = medical_rag_query(datos.query)
    return resultado