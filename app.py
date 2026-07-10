import gradio as gr

from main import (
    embed_q,
    retrieve_top_k,
    build_prompt,
    ask_llm
)


def chat(question):

    question_embedding = embed_q(question)

    retrieved_docs = retrieve_top_k(question_embedding)

    prompt = build_prompt(question, retrieved_docs)

    answer = ask_llm(prompt)

    retrieved_chunks = ""

    for i, chunk in enumerate(retrieved_docs, 1):

        retrieved_chunks += f"Chunk {i}\n"
        retrieved_chunks += "-" * 40 + "\n"
        retrieved_chunks += chunk
        retrieved_chunks += "\n\n"

    return answer, retrieved_chunks


app = gr.Interface(
    fn=chat,

    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask a question about the vehicle manual..."
    ),

    outputs=[
        gr.Textbox(
            label="Answer",
            lines=8
        ),

        gr.Textbox(
            label="Retrieved Chunks",
            lines=20
        )
    ],

    title="Automotive RAG Assistant",

    description="Ask questions about the automotive service manual.",

    examples=[
        ["Where is the instrument panel fuse block located?"],
        ["How do I remove the steering wheel?"],
        ["What is the procedure for replacing the front drive axle?"],
        ["How do I test a circuit using a multimeter?"],
        ["What are the steps for transfer case removal?"],
        ["What is the recommended engine oil viscosity?"],  
        ["How can I calibrate the radar sensor after replacing the front bumper?"] 
    ]
)

app.launch()