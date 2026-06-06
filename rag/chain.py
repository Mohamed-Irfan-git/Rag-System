from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

def create_chain(retriever):

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa