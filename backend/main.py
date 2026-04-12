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
        #  1. Connect to Pinecone
        index = get_index()

        #  2. Load metadata (FIXED PATH)
        df = pd.read_csv("backend/data/meta/chunks_meta.csv")
        chunks = df.to_dict(orient="records")

        if len(chunks) == 0:
            raise ValueError("No data found in CSV")

        #  3. Init retrievers
        init_retrievers(index, chunks)

        #  4. Debug Pinecone
        stats = index.describe_index_stats()
        print("📊 Pinecone stats:", stats)

        print(" Retrievers initialized successfully")

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        # ❌ IMPORTANT: لا تعمل init بـ None
        raise RuntimeError("Startup failed — check logs")

    yield

    print("🛑 Shutting down...")


app = FastAPI(
    title="Nutrition RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
