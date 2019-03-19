import os, sys, subprocess, glob
import os.path as path
from shutil import copyfile


def copy_code(sources_path, backup_path, exclude_dirs=None):
    """
    Creates a copy of the code currently used to run in the provided directory.
    Folders specified in exclude_dirs are not copied.
    """
    # Never copy cache.
    if not exclude_dirs:
        exclude_dirs = []
    exclude_dirs.append('__pycache__')

    for root, dirs, files in os.walk(sources_path, topdown=True):

        # Remove excluded folders in_place within the walk iterator.        
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # Strip leading local directory and construct destination directory.
        cropped_root = root[2:] if (root[:2] == './') else root 
        dest_dir = os.path.join(backup_path, cropped_root)     

        # Create target folder.
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy files.
        for filename in filter(lambda x: not '.pyc' in x, files): #lambda x: '.py' in x and not '.pyc' in x, files
            source_file_path = os.path.join(root, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            copyfile(source_file_path, dest_file_path)

def retrieve_git_hash():
    """
    Gets and returns the current gith hash.
    """
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return git_hash
    except subprocess.CalledProcessError as e:
        print(e.output)
    return False

def save_run_params_in_file(folder_path, filename, run_config):
    with open(path.join(folder_path, "run_params.conf"), 'w') as run_param_file:
        for attr, value in sorted(run_config.__dict__.items()):
                run_param_file.write(attr + ': ' +  str(value) + '\n')