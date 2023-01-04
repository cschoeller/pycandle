import os
import sys
import subprocess
import glob
import os.path as path
from shutil import copyfile

import torch


def copy_code(source_dir, dest_dir, exclude_dirs=[], exclude_files=[]):
    """
    Copies code from source_dir to dest_dir. Excludes specified folders and files by substring-matching.

    Parameters:
        source_dir (string): location of the code to copy
        dest_dir (string): location where the code should be copied to
        exclude_dirs (list of strings): folders containing strings specified in this list will be ignored
        exclude_files (list of strings): files containing strings specified in this list will be ignored
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
    Retrieves and returns the current gith hash if execution location is a git repo.
    """
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']).strip()
        return str(git_hash)[2:-1]
    except subprocess.CalledProcessError as e:
        print(e.output)
    return False


def save_run_params_in_file(folder_path, filename, run_config):
    """
    Receives a config class, fetches all member variables and saves them
    in a config file for logging purposes.

    Parameters:
        folder_path - output folder
        filename - output filename
        run_config - shallow class with parameter members
    """
    with open(path.join(folder_path, "run_params.conf"), 'w') as run_param_file:
        for attr, value in sorted(run_config.__dict__.items()):
            run_param_file.write(attr + ': ' + str(value) + '\n')


def load_model(model, checkpoint_path):
    """
    Loads the model with the state dict in the checkpoint. The loading is performed
    partial, i.e., only those variables that comply with the provided model are loaded,
    others are ignored.
    """
    training_mode = model.training
    model.eval()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if training_mode:
        model.train()


def save_model(model, output_path):
    """
    Saves the provided model's state dict in the output path location. Before
    saving the model is set to evaluation mode.
    """
    training_mode = model.training
    model.eval()
    model_state_dict = model.state_dict()
    torch.save({'arch': model.__class__.__name__,
                'model_state_dict': model_state_dict,
                }, output_path)
    if training_mode:
        model.train()
