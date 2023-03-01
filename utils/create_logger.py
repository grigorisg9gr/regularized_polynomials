import os
import logging
import time
def create_logger(root_out_path):
    #set up logger
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)
    assert os.path.exists(root_out_path), '{} does not exits'.format(root_out_path)
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(root_out_path,log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger

class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

        def __getattr__(self, attr):
            return self.get(attr)

        def __setattr__(self, key, value):
            self.__setitem__(key, value)

        def __setitem__(self, key, value):
            super(Map, self).__setitem__(key, value)
            self.__dict__.update({key: value})

        def __delattr__(self, item):
            self.__delitem__(item)

        def __delitem__(self, key):
            super(Map, self).__delitem__(key)
            del self.__dict__[key]


