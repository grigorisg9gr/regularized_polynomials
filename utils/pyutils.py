import os
from os import system, listdir, walk
from os.path import isdir, isfile, sep, join, getmtime
import shutil
import errno
from pathlib import Path
import pickle


def mkdir_p(path, mode=500):
    """ 'mkdir -p' in Python. """
    try:  # http://stackoverflow.com/a/11860637/1716869
        os.makedirs(path, mode=mode)
        return path
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and isdir(path):
            return path
        else:
            raise


def export_pickle(obj, filename):
    """It exports an object to a pickle.    """
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

