from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
import getpass
import os

####### get your api keys ###############

os.environ["OPENAI_API_KEY"] = "your open-ai key"
os.environ["GOOGLE_API_KEY"]= "your google-api-key"
os.environ["TOGETHER_API_KEY"] = "your togther-api key"

prompt = hub.pull("rlm/rag-prompt")

############# Indexing: Load ############
loader = TextLoader("./sample.txt")
docs = loader.load()

###############  Indexing: Split #################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

###############  Indexing: Store ###############

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

################ Retriever ###########################
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

###########################################################
################ USING GPT3.5-TURBO #######################
###########################################################
gpt_llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain_gpt = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | gpt_llm
    | StrOutputParser()
)




###########################################################
################ USING GEMINI #######################
###########################################################


gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro")


rag_chain_gemini = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | gemini_llm
    | StrOutputParser()
)



###########################################################
################ USING TogetherAI #######################
###########################################################


mistral_llm= ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",)



rag_chain_mistral = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | mistral_llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print(rag_chain_gemini.invoke("Which won cricket world cup 2023 ?"))