import gradio as gr
from src.rag_pipeline import retriever_qa

def create_ui():
    """Creates and returns the Gradio UI application."""
    
    # Create Gradio interface
    rag_application = gr.Interface(
        fn=retriever_qa,
        allow_flagging="never",
        inputs=[
            gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
            gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
        ],
        outputs=gr.Textbox(label="Answer"),
        title="WatsonX Document Q&A Bot",
        description="Upload a PDF document and ask any question. The chatbot will answer based on the document's content."
    )
    
    return rag_application