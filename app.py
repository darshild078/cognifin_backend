import os
import uvicorn
import gradio as gr
from app.main import app as fastapi_app

# Status page for browser visitors on Hugging Face Spaces
with gr.Blocks(title="CogniFin AI Backend") as demo:
    gr.Markdown("# 🚀 CogniFin AI Enterprise Backend is Live")
    gr.Markdown(
        "The **CogniFin Financial RAG Backend** is active and serving requests."
    )

# Mount Gradio app onto FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
