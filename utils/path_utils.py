from os import makedirs, getcwd
from os.path import basename, exists, join, isfile, dirname, isdir
import numpy as np
import shutil
from pathlib import Path
from functools import reduce
import operator


def get_by_nested_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def get_config_value(config, path, default=None):
    """
    Fetches the value of a nested key or returns default value.
    Wrapper around get_by_nested_path().
    """
    error = False
    try:
        value = get_by_nested_path(config, path)
    except KeyError:
        value, error = default, True
    return value, error


def create_result_dir(result_dir, config_path, config):
    if not exists(result_dir):
        makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        shutil.copy(fn, '{}/{}'.format(result_dir, basename(fn)))

    copy_to_result_dir(config_path, result_dir)
    # # define the paths we want to copy from. Some of those might
    # # not exist; copy only the existing ones.
    pool = [['model', 'fn']]
    for path1 in pool:
        val, err = get_config_value(config, path1)
        if not err:
            copy_to_result_dir(val, result_dir)


def ensure_config_paths(config, pb=None, verbose=True):
    """
    Parses the config (i.e. a dict) and ensures the
    paths with label 'fn' exist, or tries to replace
    it with local paths.
    """
    def _get_key(config, key):
        try:
            config[key]
            return True
        except (AttributeError, KeyError):
            return False
    # # set the base name (if the modules should change dir). If pb is provided,
    # # use that, otherwise use the current working directory.
    pbase = pb if pb is not None and isdir(pb) else getcwd()
    # # boolean to understand if something is changed.
    changed = False
    if _get_key(config, 'model'):
        updp = Path(config['model']['fn'])
        if not updp.exists():
            config['model']['fn'] = join(pbase, updp.parts[-2], updp.parts[-1])
            changed = True
    if verbose and changed:
        print('Changed the paths in config to base: {}.'.format(pbase))
    return config

