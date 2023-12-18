import os
from fastda.utils.train_api import train
from fastda.utils.arg_parser import fastda_arg_parser
from robustkd.models import *
from robustkd.loaders import *
from robustkd.trainers import *

if __name__ == '__main__':
    project_root = os.getcwd()
    package_name = 'robustkd'
    arg = fastda_arg_parser(project_root, package_name)
    train(arg)
    print('Done!')

