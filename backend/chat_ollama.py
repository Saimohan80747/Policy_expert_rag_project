# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate

# model = ChatOllama(
#     max_tokens=100,
#     model="gemma3:1b"
# )

# template = """
#     You are an exeprt in answering questions about a mouri tech policies. You have access to the following content policy documents: {context}
#     You should use the information from these documents to answer the question provided. 
#     If the answer is not found in the documents, you should respond with "I don't know". 
#     Here is the question to answer: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)
# chat = prompt | model

# def chat_with_ollama(data):
#     answer = "Here is your Answer :\\n"
    
#     context = data.get("retrieved_documents", [])
#     question = data.get("query", "")
#     result = chat.invoke({"context": context, "question": question})
#     answer += f"{result}\n\n"
#     return answer



