import os
import sys
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class CompletionError(Exception):
    """Custom exception for completion errors"""
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception), before_sleep_log=before_sleep_log(logger, logging.WARNING))
def generate_feedback():
    ### Construct retriever ###
    api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is the correct environment variable
    if not api_key:
        raise CompletionError("API Key not found. Please set OPENAI_API_KEY in your environment variables.")
    
    llm_model = OpenAI(model="gpt-4")

    # Load the entire app's content for context
    loader_all_files_content = TextLoader("all_files_content.txt")
    documents = loader_all_files_content.load()

    # Split the app's context into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    context_chunks = text_splitter.split_documents(documents)

    # Create a vector store for retrieving context using OpenAI embeddings
    vectorstore = Chroma.from_documents(documents=context_chunks, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    ### Contextualize question ###
    system_prompt = """You are given a chat history that contains \
    an entire app with all the files and modules and the user will give you a diff \
    file. Please analyze this diff in the context of the entire app, what are the \
    benefits, the drawbacks, if there was a mistake in the logic, how will this \
    change impact the modules that are dependent on this file. Point out any bugs \
    or potential issues that might appear 🐛. Suggest improvements related to \ 
    coding best practices. Ensure the code changes align with the descriptions in \
    the commit messages. Highlight any security vulnerabilities or concerns.🪪 \
    Focus on major issues and avoid minor stylistic preferences. \
    Use bullet points for multiple comments. \
    Be specific in your feedback and provide examples if possible."""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # Create the RetrievalQA chain for answering questions using the app context
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_model, retriever=retriever, chain_type="stuff", prompt=qa_prompt
    )
    
    # Process each diff file and provide feedback
    for filename in os.listdir("diffs"):
        loader_diff = TextLoader(f"diffs/{filename}")
        diff_documents = loader_diff.load()

        # Split the diff into manageable chunks
        diff_chunks = text_splitter.split_documents(diff_documents)

        review_for_file = ""
        for i, chunk in enumerate(diff_chunks):
            query = f"Analyze the following diff in the context of the whole app:\n{chunk.page_content}"
            response = retrieval_chain(query)
            review_for_file += response["result"]

        # Append the review to the reviews.txt file
        with open("reviews.txt", "a") as output_file:
            with open(f"diffs/{filename}") as file:
                output_file.write(f"FILE: {filename}\nDIFF: {file.read()}\nENDDIFF\nREVIEW: \n{review_for_file}\nENDREVIEW")

def get_file_diffs(file_list):
    """Generate diff files from the provided file list."""
    if not os.path.isdir("diffs"):
        os.mkdir("diffs")
    for file_name in file_list.split():
        diff_file = f"diffs/{file_name}.diff"
        if os.path.exists(diff_file):
            with open(f"diffs/{file_name}.txt", "w") as original_file:
                with open(diff_file, 'r') as file:
                    diff = file.read()
                original_file.write(diff)

def get_all_files_and_content():
    """Get contents of all files from the 'all_files' directory."""
    with open("all_files_content.txt", "w") as total_files:
        for root, _, files in os.walk("all_files"):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    content = file.read()
                    total_files.write(f"File name: {file_name}\nContent:\n{content}\n\n")

        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <file_names>")
        sys.exit(1)

    files = sys.argv[1]
    get_file_diffs(files)
    get_all_files_and_content()
    generate_feedback()  # Corrected function call
