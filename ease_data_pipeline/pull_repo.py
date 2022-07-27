from git import Repo
from datetime import date
from datetime import datetime

PATH_OF_GIT_REPO = '/c/Users/user/dags/analytics_app/.git'  # make sure .git folder is properly configured
repo = Repo(PATH_OF_GIT_REPO)
my_branch = 'ease_pipeline' # my branch name, where I'm going to push file
repo.git.pull('origin', my_branch) # Trying to update local repository, if there are modification on remote Repo 
print('git pull')     
