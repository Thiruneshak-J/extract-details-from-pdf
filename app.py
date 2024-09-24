import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Step 2: Store data in a vector store
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
    
    def add_documents(self, docs):
        self.documents.extend(docs)
        self.vectorizer.fit(self.documents)
    
    def retrieve(self, query, top_n=5):
        query_vector = self.vectorizer.transform([query])
        document_vectors = self.vectorizer.transform(self.documents)
        similarities = (document_vectors * query_vector.T).toarray()
        top_indices = np.argsort(similarities, axis=0)[-top_n:].flatten()[::-1]
        return [self.documents[i] for i in top_indices]

# Step 3: Set up RAG with a language model
def answer_question(question, context):
    nlp = pipeline("question-answering")
    answer = nlp(question=question, context=context)
    return answer['answer']

# Main execution
if __name__ == "__main__":
    # Extract text from PDF
    pdf_path = "thiru3535.pdf"  # Specify your PDF file
    pdf_text = extract_text_from_pdf(pdf_path)

    # Create and populate vector store
    vector_store = SimpleVectorStore()
    vector_store.add_documents([pdf_text])

    # Ask a question
    user_question = "What is the main topic of the document?"
    retrieved_docs = vector_store.retrieve(user_question)
    
    # Combine retrieved documents for context
    context = " ".join(retrieved_docs)

    # Generate the answer
    answer = answer_question(user_question, context)
    print(f"Q: {user_question}\nA: {answer}")

