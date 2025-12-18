import os
from groq import Groq
from src.vector_store import VectorStore

class RAGSystem:
    def __init__(self):
        self.vector_store = VectorStore()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile" 

    def query(self, user_query):
        # 1. Retrieve
        results = self.vector_store.search(user_query)
        
        if not results['documents'][0]:
            return "No relevant context found in the database.", results
            
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # 2. Construct Context
        context_str = ""
        for i, doc in enumerate(documents):
            source = metadatas[i].get('source', 'Unknown')
            page = metadatas[i].get('page', 'Unknown')
            img_index = metadatas[i].get('image_index', '')
            
            ref = f"Page {page}"
            if img_index:
                ref += f" (Image {img_index})"
                
            context_str += f"Source: {source} [{ref}]\nContent: {doc}\n\n"
            
        # 3. Generate Answer
        system_prompt = f"""You are an advanced Multi-Modal RAG assistant. 
You answer questions based on the provided context, which includes text and detailed descriptions of images, charts, and tables from documents.
The user cannot see the original images, only your extracted descriptions.
Always cite your sources using the [Page X] format provided in the context.
If the context doesn't contain the answer, say "I cannot find the answer in the provided documents."

Context:
{context_str}
"""
        
        completion = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model=self.model,
            temperature=0.1
        )
        
        return completion.choices[0].message.content, results
