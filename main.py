import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


with open("data/cleaned_documents.pkl", "rb") as f:
    documents = pickle.load(f)

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into chunks with overlap.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunked_documents = []

for page in documents:

    chunks = chunk_text(page)

    chunked_documents.extend(chunks)


documents = chunked_documents

print("=" * 50)
print(f"Number of Chunks: {len(documents)}")
print("=" * 50)

print(documents[0])


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


if os.path.exists("embeddings/embeddings.npy") and os.path.exists("embeddings/faiss_index.index"):

    doc_embedding = np.load("embeddings/embeddings.npy")

    index = faiss.read_index("embeddings/faiss_index.index")

    print("Embeddings and FAISS index loaded.")

else:

    doc_embedding = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(doc_embedding)

    index = faiss.IndexFlatIP(doc_embedding.shape[1])

    index.add(doc_embedding)

    np.save("embeddings.npy", doc_embedding)

    faiss.write_index(index, "faiss_index.index")

    print("Embeddings created and saved.")


def embed_q(question):

    return model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")


def retrieve_top_k(question_embedding, k=3):

    faiss.normalize_L2(question_embedding)

    distances, indices = index.search(question_embedding, k)

    return [documents[i] for i in indices[0]]


def build_prompt(question, retrieved_docs):

    context = "\n\n".join(retrieved_docs)

    return f"""
You are an automotive repair assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, reply exactly:

"I don't know based on the provided manual."

Context:
{context}

Question:
{question}

Answer:
"""


def ask_llm(prompt):

    response = client.chat.completions.create(

        model=os.getenv("GROQ_MODEL_NAME"),

        max_tokens=300,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


test_questions = [

    "How do I replace the fuel pump?",

    "How do I remove the steering wheel?",

    "Where is the instrument panel fuse block?",

    "What is the capital of Saudi Arabia?"
]


def test_rag(test_questions):

    for question in test_questions:

        print("=" * 80)
        print("Question:")
        print(question)

        question_embedding = embed_q(question)

        retrieved_docs = retrieve_top_k(question_embedding)

        prompt = build_prompt(question, retrieved_docs)

        answer = ask_llm(prompt)

        print("\nRetrieved Chunks:\n")

        for i, chunk in enumerate(retrieved_docs, 1):

            print(f"Chunk {i}")
            print("-" * 40)
            print(chunk[:500])
            print()

        print("Answer:\n")
        print(answer)
        print("=" * 80)


if __name__ == "__main__":

    test_rag(test_questions)