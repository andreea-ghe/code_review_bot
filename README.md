## Configuration

1. go to the repo homepage which you want integrate this bot
2. click settings
3. click actions under Secrets and Variables
4. create a new secret `MY_GITHUB_TOKEN` with the value of your github token (remember to give it all the rights)
5. create another secret `MY_SECRET_URL` which will have the api key of your model
   
![image that shows how to use secrets](https://github.com/andreea-ghe/code_review_bot/blob/main/secrets_photo.png?raw=true)

6. create `.github/workflows/trigger_code_review.yml` add bellow content for an automatic code review:
```
name: Trigger GPT Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]
  workflow_dispatch:

jobs:
  call_gpt_code_review:
    uses: andreea-ghe/code_review_bot/.github/workflows/main.yml@main
    with:
      base_branch: 'main'
    secrets:
      token: ${{ secrets.MY_GITHUB_TOKEN }}
      secret_url: ${{ secrets.MY_SECRET_URL }}
```
7. create `.github/workflows/trigger_pull_request.yml` with the following content (this step is optionally, only if you would
   like an automatic pull request to be started when a commit is made):
```
name: Trigger Super-Linter

on: push

jobs:
  call_superlinter:
    uses: andreea-ghe/code_review_bot/.github/workflows/superlinter.yml@main
    with:
      base_branch: 'main'
      github_repository: ${{ github.repository }}
    secrets:
      token: ${{ secrets.MY_GITHUB_TOKEN }}
      secret_url: ${{ secrets.MY_SECRET_URL }}
```

## Start using:

The robot will automatically do the code review when a new Pull request is created, the review information will show in the file 
changes part. Every time a commit is made, the review will be updated.

### Disclaimer:

❗️⚠️ This tool can help you spot bugs, but as with anything, use your judgement. Sometimes it hallucinates things that sound plausible but are false — in this case, re-run the review.
