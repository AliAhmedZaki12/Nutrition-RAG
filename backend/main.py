from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import pandas as pd

from backend.routes import router
from backend.services.retrieval_service import init_retrievers
from backend.vectorstore.pinecone_client import get_index

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Nutrition RAG API...")

    try:
        index = get_index()

        # ✅ FIX #2: unified source → CSV only (same file used during Pinecone upsert)
        # Old code used parquet here but CSV in upsert → silent index/text mismatch
        df = pd.read_csv("data/meta/chunks_meta.csv")
        chunks = df.to_dict(orient="records")

        init_retrievers(index, chunks)
        print("✅ Retrievers initialized successfully")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        init_retrievers(None, [])

    yield
    print("🛑 Shutting down...")


app = FastAPI(
    title="Nutrition RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
