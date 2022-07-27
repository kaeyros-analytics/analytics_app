from git import Repo
from datetime import date
from datetime import datetime

file_name = datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
PATH_OF_GIT_REPO = r'/c/Users/user/dags/analytics_app/.git'  # make sure .git folder is properly configured
COMMIT_MESSAGE = 'push at ' + file_name # Commit messge


repo = Repo(PATH_OF_GIT_REPO)
my_branch = 'ease_pipeline' # my branch name, where I'm going to push file
repo.git.pull('origin', my_branch) # Trying to update local repository, if there are modification on remote Repo 
        
        
#if repo.untracked_files: # Here we firstly check if there are local modifications
repo.git.add(A=True)
repo.git.commit(m=COMMIT_MESSAGE)
repo.git.push('origin', my_branch)
print('git push')
#else:
#print('Noting to commit')
