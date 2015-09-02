# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np
import json

from utils import *
from importer import *


if __name__ != '__main__':
    os.sys.exit(1)

parser = argparse.ArgumentParser(\
    description="Import dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--clear', metavar="C", nargs=1,
                    default=[0], type=int,
                    help='Clear previous dataset (0-1)')
parser.add_argument('--language', metavar="lang", nargs='?',
                    default=['EN', 'DU', 'GR', 'SP'],
                    help='Given a folder, imports all its subfolders that \
                            agrees with the indicated language')
parser.add_argument('-i', metavar="path", nargs='?',
                    default=[''], help='Input path.')
parser.add_argument('-o', metavar="path", nargs='?',
                    default=[''], help='Output path.')
parser.add_argument('--config', metavar="conf", nargs='?',
                    default="conf/config.json", help='Configuration file')
args = parser.parse_args()

config = get_configuration(args.config)

if type(args.language) == str:
    args.language = [args.language]

clear(args.language, args.o, bool(args.clear[0]))
import_languages(config, args.language, args.i, args.o)
