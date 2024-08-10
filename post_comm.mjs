import { Octokit } from '@octokit/core';

async function postCommentToGitHub(comment_body, commit_id, file_path, start_line, line, start_side, side) {
  try {
    const { default: fetch } = await import('node-fetch');
    const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN, request: { fetch } });

    comment_body = comment_body.replace(/\\n/g, '\n').replace(/\\t\+/g, '    ');

    console.log('Arguments:', { comment_body, commit_id, file_path, start_line, line, start_side, side });

    const [owner, repo] = process.env.GITHUB_REPOSITORY.split('/');
    const pull_number = process.env.GITHUB_EVENT_NUMBER;

    console.log(comment_body);

    // Fetch existing comments
    const { data: existingComments } = await octokit.request('GET /repos/{owner}/{repo}/pulls/{pull_number}/comments', {
      owner,
      repo,
      pull_number
    });

    // Check if the comment already exists
    const isCommentExisting = existingComments.some(comment => 
      comment.body === comment_body && 
      comment.path === file_path && 
      comment.position === parseInt(line) && 
      comment.commit_id === commit_id
    );

    if (isCommentExisting) {
      console.log('Comment already exists, skipping...');
      return;
    }

    // Post new comment
    let response;
    if (start_line == line) {
      response = await octokit.request('POST /repos/{owner}/{repo}/pulls/{pull_number}/comments', {
        owner,
        repo,
        pull_number,
        commit_id,
        path: file_path,
        body: comment_body,
        line: parseInt(line),
        side
      });
    } else {
      response = await octokit.request('POST /repos/{owner}/{repo}/pulls/{pull_number}/comments', {
        owner,
        repo,
        pull_number,
        commit_id,
        path: file_path,
        body: comment_body,
        start_line: parseInt(start_line),
        line: parseInt(line),
        start_side,
        side
      });
    }

    if (response.status !== 201) {
      throw new Error(`GitHub API responded with ${response.status}: ${response.statusText}`);
    }

    console.log('Comment posted successfully:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error posting comment:', error.message);
    console.error(error);
  }
}

const args = process.argv.slice(2);

if (args.length < 7) {
  console.error('Insufficient arguments. Expected: comment_body, commit_id, file_path, start_line, line, start_side, side');
  process.exit(1);
}

const [comment_body, commit_id, file_path, start_line, line, start_side, side] = args;

const formatted_comment_body = comment_body.replace(/\\n/g, '\n').replace(/\\t\+/g, '    ');

postCommentToGitHub(formatted_comment_body, commit_id, file_path, start_line, line, start_side, side)
  .then(response => console.log('Comment posted successfully:', response))
  .catch(error => console.error('Error posting comment:', error));
