from fastapi import APIRouter, Query, HTTPException, Request
from backend.services.retrieval_service import rag_answer_hybrid_service

router = APIRouter()


@router.get("/query")
def query_rag(
    request: Request,
    q: str = Query(..., description="Your nutrition question"),
    top_k: int = Query(5, ge=1, le=20)
):
    try:
        
        if not request.app.state.ready:
            raise HTTPException(
                status_code=503,
                detail="System is still initializing"
            )

        answer, context = rag_answer_hybrid_service(q, top_k=top_k)

        return {
            "answer": answer,
            "context": context,
            "query": q
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing query"
        )


@router.get("/status")
def status(request: Request):
    return {
        "status": "ready" if getattr(request.app.state, "ready", False) else "starting",
        "service": "nutrition-rag-api"
    }


@router.get("/")
def root():
    return {
        "message": "Nutrition RAG API is running",
        "endpoints": ["/query", "/status"]
    }
