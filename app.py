import gradio as gr
from app.main import app as fastapi_app
from app.core.config import settings

# Secure status interface for browser visitors
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

if __name__ == "__main__":
    demo.launch()
