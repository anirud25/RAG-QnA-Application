from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM
from langchain.chains import RetrievalQA

from src.ingest import document_loader, text_splitter, vector_database

## LLM
def get_llm():
    # Note: A project_id is required for this model
    project_id = "skills-network"
    
    model_id = 'ibm/granite-3-2-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }

    watsonx_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id=project_id,
        params=parameters,
    )
    return watsonx_llm

## Retriever
def retriever(file):
    # This function now correctly uses the file path passed from Gradio
    splits = document_loader(file.name) 
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False)
    
    response = qa.invoke(query)
    return response['result']