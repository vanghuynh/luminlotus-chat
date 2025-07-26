from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from src.apis.routers.rag_agent_template import router as router_rag_agent_template
from src.apis.routers.file_processing_router import router as router_file_processing

api_router = APIRouter()
api_router.include_router(router_rag_agent_template)
api_router.include_router(router_file_processing)

def create_app():
    app = FastAPI(
        docs_url="/docs",
        title="AI Service ",
    )

    @app.get("/")
    def root():
        return {
            "message": "Backend is running. ",

        }

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
