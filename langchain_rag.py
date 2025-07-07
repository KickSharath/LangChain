from langchain_ollama import ChatOllama

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA


llm = ChatOllama(model = "mistral", temperature = 1)

# Load and embed a document
loader = TextLoader('demo.txt')
document = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
chuck = splitter.split_documents(document)

embeddings = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.from_documents(chuck, embedding = embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

while True:
    query = input("\n Kick: ")

    if query.lower() in ['exit']:
        print("Exting...")
        break
    response = qa_chain.invoke({"query": query})
    print("📘 Answer:", response["result"])

