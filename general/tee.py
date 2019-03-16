import sys

class Tee(object):
    """
    Forks the stdout stream in a file while still displaying it
    on the terminal.
    """

    def __init__(self, filename, mode):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        
    def flush(self):
        self.file.flush()