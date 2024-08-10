import os
from github import Github
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Get environment variables
GITHUB_TOKEN = os.getenv("MY_GITHUB_TOKEN")
REPO_NAME = os.getenv("GITHUB_REPOSITORY")
print(REPO_NAME)
if not GITHUB_TOKEN or not REPO_NAME:
    raise ValueError("Missing GITHUB_TOKEN or GITHUB_REPOSITORY environment variable")

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)

class GitHubError(Exception):
    """Custom exception for GitHub errors"""
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception), before_sleep=before_sleep_log(logger, logging.WARNING))
def get_latest_commit_branch():
    latest_commit_date = None
    latest_commit_branch = None
    
    branches = repo.get_branches()
    for branch in branches:
        commits = repo.get_commits(sha=branch.name)
        if commits:
            latest_commit = commits[0]
            if latest_commit_date is None or latest_commit.commit.author.date > latest_commit_date:
                latest_commit_date = latest_commit.commit.author.date
                latest_commit_branch = branch.name

    return latest_commit_branch

@retry(stop=stop_after_attempt(3), wait=wait_exponential(), retry=retry_if_exception_type(Exception), before_sleep=before_sleep_log(logger, logging.WARNING))
def create_and_merge_pull_request():
    try:
        base = repo.default_branch
        head = get_latest_commit_branch()

        if head is None:
            raise GitHubError("No branch found for the latest commit.")

        print(f"Latest commit branch: {head}")
        print(f"Base branch: {base}")

        if head == base:
            print(f"No new changes to merge. Both head and base are {base}.")
            return

        # Check for existing pull requests
        pulls = repo.get_pulls(state='open', head=f"{repo.owner.login}:{head}", base=base)
        if pulls.totalCount > 0:
            print(f"A pull request already exists for {head} into {base}.")
            pr = pulls[0]
        else:
            # Create a pull request
            pr_title = f"Automated PR from {head} to {base}"
            pr_body = "This pull request is automatically created."
            pr = repo.create_pull(title=pr_title, body=pr_body, head=head, base=base)
            print(f"Pull request created: {pr.html_url}")
    except Exception as e:
        raise GitHubError(f"Failed to create and merge pull request after 3 retries: {str(e)}")

if __name__ == "__main__":
    create_and_merge_pull_request()
