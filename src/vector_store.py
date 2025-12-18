import os
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, collection_name="rag_collection"):
        # Persistent storage
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
        
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, documents):
        """
        documents: list of dicts {'content': str, 'metadata': dict}
        """
        if not documents:
            return
            
        # Create IDs
        existing_count = self.collection.count()
        ids = [str(existing_count + i) for i in range(len(documents))]
        
        documents_text = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        # Ensure metadata values are strings/int/float (Chroma restriction)
        # Convert any complex objects to string if needed
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        embeddings = self.embedding_model.encode(documents_text).tolist()
        
        self.collection.add(
            ids=ids,
            documents=documents_text,
            embeddings=embeddings,
            metadatas=clean_metadatas
        )

    def search(self, query, k=5):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        return results
