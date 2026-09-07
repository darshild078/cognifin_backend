import gradio as gr
from app.main import app as fastapi_app

# Gradio interface for browser visitors
with gr.Blocks(title="CogniFin AI Backend") as demo:
    gr.Markdown("# 🚀 CogniFin AI Enterprise Backend is Live")
    gr.Markdown(
        """
        The **CogniFin Financial RAG Backend** is active and serving requests.
        
        - 📖 **API Docs:** [/docs](/docs)
        - 🩺 **System Health:** [/health](/health)
        """
    )

# Mount FastAPI app onto Gradio's internal FastAPI application
demo.app.mount("", fastapi_app)

if __name__ == "__main__":
    demo.launch()
