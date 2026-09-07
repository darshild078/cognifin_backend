import gradio as gr
from app.main import app as fastapi_app
from app.core.config import settings
from app.core.database import ensure_indexes
from app.services.rag_service import rag_service

# Secure status interface for browser visitors
# 1. Explicitly initialize Database and RAG Vector Engine
ensure_indexes()
try:
    rag_service.initialize()
except Exception as e:
    pass

# 2. Secure status interface for browser visitors
docs_line = "- 📖 **API Docs:** [/docs](/docs)\n" if settings.ENABLE_DOCS else ""

with gr.Blocks(title="CogniFin AI Gateway") as demo:
    gr.Markdown("# 🛡️ CogniFin AI Production Gateway")
    gr.Markdown(
        f"""
        The **CogniFin Enterprise Financial API** is operational.
        
        {docs_line}- 🩺 **System Health:** [/health](/health)
        """
    )

# Mount FastAPI app onto Gradio's internal FastAPI application
demo.app.mount("", fastapi_app)
# 3. Attach all FastAPI routes directly into demo.app
demo.app.include_router(fastapi_app.router)

if __name__ == "__main__":
    demo.launch()
