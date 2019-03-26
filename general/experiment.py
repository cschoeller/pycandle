import os
import os.path as path
import sys
from time import time
import datetime

from tee import Tee
from utils import *


class Experiment:
    """
    Generates a folder where all experiments will be stored an then a named experiment with current
    timestamp and provided name. Automatically starts logging the console output and creates a copy
    of the currently executed code in the experiment folder. The experiment's subfolder paths are provided
    to the outside as member variables. It also allows adding of more subfolders conveniently.

    Args:
        experiment_name - name of the exerpiment to be created
        experiments_path - location where all experiments will be stored, default is './experiments'

    Example:
        >>> experiment = Experiment('mnist_classification')
        >>> print(experiment.plots) # path to experiment plots
    """

    def __init__(self, experiment_name, experiments_path=None):
        self.experiments_path = self._set_experiments_dir(experiments_path)
        self.name = self._set_experiment_name(experiment_name)
        self.path = path.join(self.experiments_path, self.name) # path to current experiment
        self._sub_directories = ['plots', 'logs', 'code'] # default sub-directories
        self._init_directories()
        self._tee = Tee(path.join(self.logs, 'console_output.log'), 'w') # start to log console
        self._copy_sourcecode()

    def _set_experiments_dir(self, experiments_path):
        if experiments_path != None:
            return experiments_path
        local_path = os.path.dirname(sys.argv[0])
        local_path = local_path if local_path != '' else './'
        return path.join(local_path, "experiments")

    def _set_experiment_name(self, experiment_name):
        date_time = datetime.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H:%M:%S')
        return date_time + "_" + experiment_name

    def _init_directories(self):
        """ Create all basic directories. """
        self._create_directory(self.experiments_path)
        self._create_directory(path.join(self.experiments_path, self.name))
        for sub_dir_name in self._sub_directories:
            self.add_directory(sub_dir_name)

    def _create_directory(self, dir_path):
        if not path.exists(dir_path):
            os.makedirs(dir_path)

    def _add_member(self, key, value):
        """ Add a member variable named 'key' with value 'value' to the experiment instance. """
        self.__dict__[key] = value

    def _copy_sourcecode(self):
        """ Copy code from execution directory in experiment code directory. """
        sources_path = os.path.dirname(sys.argv[0])
        sources_path = sources_path if sources_path != '' else './'
        copy_code(sources_path, self.code, exclude_dirs=[path.basename(self.experiments_path), '.vscode', '.git'])

    def add_directory(self, dir_name):
        """
        Add a sub-directory to the experiment. The directory will be automatically
        created and provided to the outside as a member variable.
        """
        # store in sub-dir list
        if not dir_name in self._sub_directories:
            self._sub_directories.append(dir_name)
        # add as member
        dir_path = path.join(self.experiments_path, self.name, dir_name)
        self._add_member(dir_name, dir_path)
        # create directory
        self._create_directory(dir_path)

    def add_file(self, folder_path, filename, content):
        """ Adds a file with provided content to folder. Convenience function. """
        with open(path.join(folder_path, filename), 'w') as textfile:
            textfile.write(content)
