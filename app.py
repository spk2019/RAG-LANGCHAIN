import gradio as gr
from train import rag_chain_gemini,rag_chain_gpt,rag_chain_mistral 

def ask(question):
    answer1 = rag_chain_gpt.invoke(question)
    answer2 = rag_chain_gemini.invoke(question) 
    answer3 = rag_chain_mistral.invoke(question) 
    return answer1,answer2,answer3    # Ask anything about cricket worldcup final 2023.


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ICC CRICKET WORLDCUP FINAL 2023")
    question = gr.Textbox(label="Question",placeholder="Type your question")
    answer1 = gr.Textbox(label="openAI",placeholder="Answer")
    answer2 = gr.Textbox(label="GEMINI",placeholder="Answer")
    answer3 = gr.Textbox(label="TogetherAI",placeholder="Answer")
    ask_button = gr.Button("Submit")
    ask_button.click(fn=ask,inputs=[question],outputs=[answer1,answer2,answer3])
   
    
if __name__ == "__main__":
    demo.launch(share = True)

