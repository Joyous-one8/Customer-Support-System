from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    """
    Wrapper around sentence transformer embeddings
    """
    def __init__(self, model_name):
        self.model=HuggingFaceEmbeddings(
            model_name=model_name
        )
    
    def embed_document(self,texts):
        return self.model.embed_documents(texts)
    
    def embed_query(self,query:str):
        return self.model.embed_query(query)