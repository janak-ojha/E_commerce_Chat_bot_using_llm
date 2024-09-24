from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ecommbot.ingest import ingestdata

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    # Replace ChatOpenAI with HuggingFacePipeline
    pipeline = HuggingFacePipeline(
        task="text-generation", model="text-generation-finetuned-t5-xl"  # Pre-trained T5 model
    )

    # Create HuggingFaceEmbeddings instance with your API key
    embedding = HuggingFaceEmbeddings(api_key="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")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | pipeline
        | StrOutputParser()
    )

    return chain

if __name__ == '__main__':
    vstore = ingestdata("done", embedding)  # Pass the embedding instance to ingestdata
    chain = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))