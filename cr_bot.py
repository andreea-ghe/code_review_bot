import os
import sys
from litellm import completion
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
class CompletionError(Exception):
    """Custom exception for completion errors"""
    pass
@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception), before_sleep=before_sleep_log(logger, logging.WARNING))

def generate_feedback(file_name, diff, code_content, all_files_content):
    """Generate feedback using OpenAI GPT model."""
    system_message = f"""\
    Hello! 👋
    
    Your task is to review the recent code changes and provide detailed feedback.😃 Here’s what you need to do:
    Review Changes: Examine the code differences and the final code provided. Please focus on differences and its impact on the whole project.
    🐛Identify Bugs: Point out any bugs or potential issues.
    Best Practices: Suggest improvements related to coding best practices.
    Commit Message Alignment: Ensure the code changes align with the descriptions in the commit messages.
    🪪Security Concerns: Highlight any security vulnerabilities or concerns.
    
    Instructions:
    Focus on major issues and avoid minor stylistic preferences.
    Use bullet points for multiple comments.
    Be specific in your feedback and provide examples if possible.
    
    This is the file {file_name} which changes I would like you to comment on.
    Its Code Changes:
    {diff}
    Its entire code:
    {code_content}

    Next, here is the list of all files and their content to help you understand the overall impact: {all_files_content}
        
    Thank you for your attention to detail and expertise! 🚀
    P.S. Please write in a github markdown format. Thank you <3
    Your review:
    """
    # This line is optional, if wanting to traverse the entire code, if the code is too large, it won't work as it doesn't have enough tokens
    # Next, here is the list of all files and their content to help you understand the overall impact: {all_files_content}

    def get_completion():
        response = completion(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
            ],
            api_key=os.getenv('MY_URL')
            # api_base=os.getenv('MY_URL') for running local models
        )
        return response

    try:
        response = get_completion()
        return response['choices'][0]['message']['content']
    except Exception as e:
        raise CompletionError(f"Failed to generate feedback after 3 retries: {str(e)}")

def review_code_diffs(diffs, file_contents, all_files_content):
    review_results = []
    for file_name, diff in diffs.items():
        print("The differences are:\n", diff)
        if diff:
            code_content = file_contents.get(file_name, "")
            answer = generate_feedback(file_name, diff, code_content, all_files_content)
            review_results.append(f"FILE: {file_name}\nDIFF: {diff}\nENDDIFF\nREVIEW: \n{answer}\nENDREVIEW")

    return "\n".join(review_results)


def get_file_contents(file_list):
    contents = {}
    for file_name in file_list.split():
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                content = file.read()
            contents[file_name] = content
    return contents
 

def get_file_diffs(file_list):
    diffs = {}
    for file_name in file_list.split():
        diff_file = f"diffs/{file_name}.diff"
        if os.path.exists(diff_file):
            with open(diff_file, 'r') as file:
                diff = file.read()
            diffs[file_name] = diff
    return diffs

def get_all_files_and_content():
    """Get contents of all files from the 'all_files' directory."""
    all_files_dir = "all_files"  # Directory where all files are stored
    all_files_content = ""

    for root, _, files in os.walk(all_files_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as file:
                content = file.read()
                all_files_content += f"File name: {file_name}\nContent:\n{content}\n\n"
    
    return all_files_content
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <file_names>")
        sys.exit(1)

    files = sys.argv[1]
    file_diffs = get_file_diffs(files)
    file_contents = get_file_contents(files)
    all_files_content = get_all_files_and_content()  # Get contents of all files
    result = review_code_diffs(file_diffs, file_contents, all_files_content)
    with open('reviews.txt', 'w') as output_file:
        output_file.write(result)
