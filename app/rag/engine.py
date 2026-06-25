from app.core.config import PDF_PATH, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, TOP_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

class RAGEngine:
    """
    Singleton-style RAG ENgine.
    Initialized once and serve all queries
    """

    def __init__(self):
        self.vector_store=None
        self._initialize()

    def _initialize(self):
        load_dotenv()

        text=PDFLoader(PDF_PATH).load()

        chunks=LangchainTextChunker(CHUNK_SIZE,CHUNK_OVERLAP).chunk(text)

        embeddings=EmbeddingModel(EMBEDDING_MODEL_NAME)

        self.vector_store=VectorStore(embeddings)
        self.vector_store.build(chunks)
        self.llm=ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=1024
        )

    def generate_answer(self, question:str):
        """
        generate an answer using the vector store with a grounded prompt.
        retrieve top-k chunks and pass tem to llm with strict prompt
        """

        contexts=self.vector_store.search(query=question,k=TOP_K)
        combined_text="\n\n".join(contexts)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use only the information provided in the context below to answer the question. If the answer is not present in the context, reply 'I don't know' politely and humbly."),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])
        
        chain = prompt | self.llm
        
        result = chain.invoke({
            "context": combined_text,
            "question": question
        })

        return result.content