import os
import sys
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

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
        
        llm_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("MY_URL"))
    
        # Load the entire app's content for context
        loader_all_files_content = TextLoader("all_files_content.txt")
        documents = loader_all_files_content.load()
    
        # Split the app's context into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        context_chunks = text_splitter.split_documents(documents)
    
        # Create a vector store for retrieving context using OpenAI embeddings
        vectorstore = Chroma.from_documents(documents=context_chunks, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()
    
        ### Contextualize question ###
        # system_prompt = """You are given a chat history that contains \
        # an entire app with all the files and modules and the user will give you a diff \
        # file. Please analyze this diff in the context of the entire app, what are the \
        # benefits, the drawbacks, if there was a mistake in the logic, how will this \
        # change impact the modules that are dependent on this file. Point out any bugs \
        # or potential issues that might appear 🐛. Suggest improvements related to \ 
        # coding best practices. Ensure the code changes align with the descriptions in \
        # the commit messages. Highlight any security vulnerabilities or concerns.🪪 \
        # Focus on major issues and avoid minor stylistic preferences. \
        # Use bullet points for multiple comments. \
        # Be specific in your feedback and provide examples if possible."""

        # PROJECT_ID = "code-review-329e5"
        # SESSION_ID = "user_session"
        # COLLECTION_NAME = "chat_history"

        # client = firestore.client(project = PROJECT_ID)
        # chat_history = FirestoreChatMessageHistory(
        #     session_id = SESSION_ID,
        #     collection = COLLECTION_NAME,
        #     client = client,
        # )
        # chat_history.add_ai_message(system_prompt)

        # for filename in os.listdir("diffs"):
        #     loader_diff = TextLoader(f"diffs/{filename}")
        #     diff_documents = loader_diff.load()
    
        #     # Split the diff into manageable chunks
        #     diff_chunks = text_splitter.split_documents(diff_documents)
    
        #     review_for_file = ""
        #     chat_history.add_user_message(f"Please analyze the following diff in the context of the whole app:\n")
        #     for i, chunk in enumerate(diff_chunks):
        #         chat_history.add_user_message(chunk.page_content)
        #     response = chat_history.add_user_message("Can you please review the last diff added")
        #     chat_history.add_ai_message(response.message)

        
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
        Be specific in your feedback and provide examples if possible.\n\n
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
        
        # Process each diff file and provide feedback
        for filename in os.listdir("differs"):
            loader_diff = TextLoader(f"differs/{filename}")
            diff_documents = loader_diff.load()
    
            # Split the diff into manageable chunks
            diff_chunks = text_splitter.split_documents(diff_documents)
    
            review_for_file = ""
            chat_history.append(HumanMessage(content = "Here starts the diff file:\n"))
            for i, chunk in enumerate(diff_chunks):
                chat_history.append(HumanMessage(content = chunk.page_content))
            result = rag_chain.invoke({"input": "Please analyse the last diff file that was given to you in the context of the entire app", "chat_history": chat_history})

            chat_history.append(HumanMessage(content = "Please analyse the last diff file that was given to you in the context of the entire app"))
            chat_history.append(SystemMessage(content = result["answer"]))
    
            # Append the review to the reviews.txt file
            with open("reviews.txt", "a") as output_file:
                with open(f"differs/{filename}", "r") as file:
                    output_file.write(f"FILE: {os.path.splitext(filename)[0]}\nDIFF: {file.read()}\nENDDIFF\nREVIEW: \n{result['answer']}\nENDREVIEW")
                    
    try:
        get_review()
    except Exception as e:
        raise CompletionError(f"Failed to generate feedback after 3 retries: {str(e)}")

def get_file_diffs(file_list):
    """Generate diff files from the provided file list."""
    if not os.path.isdir("differs"):
        os.mkdir("differs")
    for file_name in file_list.split():
        # Replace slashes with underscores
        sanitized_file_name = file_name.replace("/", "_")
        diff_file = f"diffs/{sanitized_file_name}.diff"
        
        if os.path.exists(diff_file):
            with open(f"differs/{sanitized_file_name}.txt", "w") as original_file:
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
