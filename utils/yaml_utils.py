from os.path import splitext, basename, dirname, isfile, join
import sys
import torch


def load_module(fn, name):
    mod_name = splitext(basename(fn))[0]
    mod_path = dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    model1 = model(**args) if args else model()
    return model1
