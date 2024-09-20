import os
import sys
import mimetypes
import traceback

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class CompletionError(Exception):
    """Custom exception for completion errors"""
    pass
    
@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception), before_sleep=before_sleep_log(logger, logging.WARNING))
def generate_feedback():
    def get_review():
        ### Construct retriever ###
        api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is the correct environment variable
        if not api_key:
            raise CompletionError("API Key not found. Please set OPENAI_API_KEY in your environment variables.")
        
        llm_model = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("MY_URL"))
    
        # Load the entire app's content for context
        loader_all_files_content = TextLoader("all_files_content.txt")
        documents = loader_all_files_content.load()
    
        # Split the app's context into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        context_chunks = text_splitter.split_documents(documents)
    
        # Create a vector store for retrieving context using OpenAI embeddings
        vectorstore = Chroma.from_documents(documents=context_chunks, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        system_prompt = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm_model, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are given a chat history that contains \
        an entire app with all the files and modules and the user will give you some diff \
        files. Please analyze these diffs in the context of the entire app, what are the \
        benefits, the drawbacks, if there was a mistake in the logic, how will this \
        change impact the modules that are dependent on this file. Point out any bugs \
        or potential issues that might appear 🐛. Suggest improvements related to \ 
        coding best practices. Ensure the code changes align with the descriptions in \
        the commit messages. Highlight any security vulnerabilities or concerns.🪪 \
        Focus on major issues and avoid minor stylistic preferences. \
        Use bullet points for multiple comments. \
        Be specific in your feedback and provide examples if possible. \
        When analyzing the diff, always consider that definitions or initializations \
        might exist elsewhere in the codebase. Avoid assuming variables or functions \
        are undefined without checking the retrieved context\n\n
        {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chat_history = []
        MAX_TOKENS = 8192
        
        def count_tokens(message_content, model):
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(message_content))
        
        # Process each diff file and provide feedback
        for root, dirs, files in os.walk("diffs"):
            for filename in files:
                if filename.endswith(".diff"):
                    filepath = os.path.join(root, filename)
                    print("Processing diff file:", filepath)
                    
                    chat_history.clear()
                    loader_diff = TextLoader(filepath)
                    diff_documents = loader_diff.load()
            
                    # Split the diff into manageable chunks
                    diff_chunks = text_splitter.split_documents(diff_documents)
                    
                    token_count = 0
                    chat_history.append(HumanMessage(content = "Here starts the diff file:\n"))
                    token_count += count_tokens("Here starts the diff file:\n", model="gpt-4")
                    token_count += count_tokens("Please analyse the last diff file that was given to you in the context of the entire app", model="gpt-4")
                    for i, chunk in enumerate(diff_chunks):
                        chunk_content = chunk.page_content
                        chunk_tokens = count_tokens(chunk_content, model="gpt-4")
                        if token_count + chunk_tokens >= 7500:
                            break
                        else:
                            chat_history.append(HumanMessage(content = chunk_content))
                            token_count += chunk_tokens
    
                    result = rag_chain.invoke({"input": "Please analyse the last diff file that was given to you in the context of the entire app", "chat_history": chat_history})
                    
                    print(f"diffs/{filename}\n")
                    print("answer: ", result["answer"])
            
                    # Append the review to the reviews.txt file
                    with open("reviews.txt", "a") as output_file:
                        with open(filepath, "r") as file:
                            file_relative_path = os.path.relpath(os.path.splitext(filepath)[0], start="diffs")
                            output_file.write(f"FILE: {file_relative_path}\nDIFF: {file.read()}\nENDDIFF\nREVIEW: \n{result['answer']}\nENDREVIEW\n")
                    
                    
    try:
        get_review()
    except Exception as e:
        traceback.print_exc()
        raise CompletionError(f"Failed to generate feedback after 3 retries: {str(e)}")

def get_all_files_and_content():
    """Get contents of all files from the 'all_files' directory."""
    with open("all_files_content.txt", "w") as total_files:
        for root, _, files in os.walk("all_files"):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                # Guess the MIME type of the file
                mime_type, _ = mimetypes.guess_type(file_path)
                
                try:
                    if mime_type and mime_type.startswith('text'):
                        # If it's a text file, read and write its content
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                            total_files.write(f"File name: {file_name}\nContent:\n{content}\n\n")
                    else:
                        total_files.write(f"File name: {file_name}\nContent: <binary file>\n\n")
                except Exception as e:
                    total_files.write(f"File name: {file_name}\nError reading file: {str(e)}\n\n")

        
if __name__ == "__main__":
    get_all_files_and_content()
    generate_feedback()  
