import os, sys, subprocess, glob
import os.path as path
from shutil import copyfile


def copy_code(source_dir, dest_dir, exclude_dirs=['__pycache__', '.git', 'experiments'], exclude_files=['.pyc']):
    """
    Copies code from source_dir to dest_dir. Excludes specified folders and files by substring-matching.
    """
    source_basename = path.basename(source_dir)
    for root, dirs, files in os.walk(source_dir, topdown=True):

        # skip ignored dirs
        if any(ex_subdir in root for ex_subdir in exclude_dirs):
            continue

        # construct destination dir
        cropped_root = root[2:] if (root[:2] == './') else root
        subdir_basename = path.basename(cropped_root)

        # do not treat the root as a subdir
        if subdir_basename == source_basename:
            subdir_basename = ""
        dest_subdir = os.path.join(dest_dir, subdir_basename)

        # create destination folder
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        # copy files
        for filename in filter(lambda x: not any(substr in x for substr in exclude_files), files):
            source_file_path = os.path.join(root, filename)
            dest_file_path = os.path.join(dest_subdir, filename)
            copyfile(source_file_path, dest_file_path)

def retrieve_git_hash():
    """
    Retrieves and returns the current gith hash.
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