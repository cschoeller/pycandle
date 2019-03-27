import sys

class Tee(object):
    r"""
    This object works like the linux tee. It forks the stdout stream, this means it writes all command line
    output into specified file but it will still be displayed as console output.

    Args:
        filename (string): name of the file write the stdout to
        mode (string): should be either 'w' for write or 'a' for append (accepts all open() modes)

    Example:
        >>> tee = Tee('console_output.log', 'w')
    """

    def __init__(self, filename, mode):
        """ Open file and redirect system stdout. """
        self._file = open(filename, mode)
        self._stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        """ Reset streams to initial state on destruction. """
        sys.stdout = self._stdout
        self._file.close()

    def write(self, data):
        """ Write to file and stdout. Is called automatically. """
        self._file.write(data)
        self._stdout.write(data)
        
    def flush(self):
        """ Force a buffer flush. """
        self._file.flush()