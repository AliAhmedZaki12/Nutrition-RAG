from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import pandas as pd

from backend.routes import router
from backend.services.retrieval_service import init_retrievers
from backend.vectorstore.pinecone_client import get_index

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Nutrition RAG API...")

    app.state.ready = False

    try:
        # 1. Connect to Pinecone
        index = get_index()

        # 2. Load metadata safely (portable path)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "backend/data/meta/chunks_meta.csv")

        df = pd.read_csv(csv_path)
        chunks = df.to_dict(orient="records")

        if not chunks:
            print(" Warning: No chunks found in CSV")

        # 3. Init retrievers
        init_retrievers(index, chunks)

        # 4. Debug Pinecone
        stats = index.describe_index_stats()
        print(" Pinecone stats:", stats)

        print(" Retrievers initialized successfully")

        app.state.ready = True

    except Exception as e:
        print(f" Startup error: {e}")
        print(" Running in degraded mode (API still alive)")
        app.state.ready = False

    yield

    print("🛑 Shutting down...")


app = FastAPI(
    title="Nutrition RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
