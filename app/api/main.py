from fastapi import FastAPI, Query
from app.rag.engine import RAGEngine
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine=RAGEngine()

@app.get("/query")
def query(question:str = Query(...,description="User Question")):
    """
    Return an LLM generated answer grounded using the PDF ANswer
    """
    try:
        answer=rag_engine.generate_answer(question)
        return{
            "question":question,
            "answer":answer
        }
    except Exception as e:
        return{
            "question":question,
            "answer":None,
            "error":str(e)
        }