import os

PROJECT_ROOT = {'orrbarkat': '/Users/orrbarkat/repos/tracking',
                'Amos':'/path/to/your_project'} # replace amos with you username

def project_root():
    return PROJECT_ROOT[os.getlogin()]

