from fastapi import APIRouter, Query, HTTPException
from backend.services.retrieval_service import rag_answer_hybrid_service

router = APIRouter()


@router.get("/query")
def query_rag(q: str = Query(..., description="Your nutrition question"),
              top_k: int = Query(5, ge=1, le=20)):
    try:
        answer, context = rag_answer_hybrid_service(q, top_k=top_k)
        return {"answer": answer, "context": context}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status():
    return {"status": "ok"}
