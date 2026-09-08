try:
    import spaces
except ImportError:
    spaces = None

import os

# Disable Gradio SSR Node.js proxy to serve pure Python FastAPI
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr

from app.api.router import api_router
from app.core.config import settings
from app.core.database import ensure_indexes
from app.core.logging import logger, setup_logging
from app.services.rag_service import rag_service

# 1. Structured Logging and System Initialization
setup_logging()
logger.info("module=app action=startup status=initializing")

ensure_indexes()
try:
    rag_service.initialize()
except Exception as e:
    logger.warning(f"module=app action=rag_init status=degraded error='{e}'")

# 2. ZeroGPU function registered into Gradio dependency graph
if spaces is not None:
    @spaces.GPU
    def gpu_worker(text: str) -> str:
        return text
else:
    def gpu_worker(text: str) -> str:
        return text

# 3. Status interface for browser visitors
docs_line = "- 📖 **API Docs:** [/docs](/docs)\n" if settings.ENABLE_DOCS else ""

with gr.Blocks(title="CogniFin AI Gateway") as demo:
    gr.Markdown("# 🛡️ CogniFin AI Production Gateway")
    gr.Markdown(
        f"""
        The **CogniFin Enterprise Financial API** is operational.
        
        {docs_line}- 🩺 **System Health:** [/health](/health)
        - 🌐 **Status Endpoint:** [/status](/status)
        """
    )
    dummy_box = gr.Textbox(visible=False)
    dummy_btn = gr.Button(visible=False)
    dummy_btn.click(fn=gpu_worker, inputs=dummy_box, outputs=dummy_box)

if __name__ == "__main__":
    # 4. Launch Gradio server non-blocking to satisfy ZeroGPU startup probe
    port = int(os.getenv("PORT", 7860))
    logger.info(f"module=app action=launch_gateway port={port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        prevent_thread_lock=True,
        ssr_mode=False,
    )

    # 5. Mount Complete API Router onto the active server instance
    demo.app.include_router(api_router)
    logger.info("module=app action=routes_mounted status=ready")

    # 6. Block main thread to keep server alive
    demo.block_thread()


