# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np

from utils import *
from db_layer import *
from feature_extractor import *

if __name__ != '__main__':
    os.sys.exit(1)

parser = argparse.ArgumentParser(\
    description="Train and compute the authors' features.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--clear', metavar="C", nargs=1,
                    default=[0], type=int,
                    help='Clear previous features (0-1)')
parser.add_argument('--train', metavar="T", nargs=1,
                    default=[1], type=int,
                    help='Train features (0-1)')
parser.add_argument('--compute', metavar="C", nargs=1,
                    default=[1], type=int,
                    help='Compute features (0-1)')
parser.add_argument('--language', metavar="lang", nargs='?',
                    default=["DU", "EN", "GR", "SP"],
                    help='Language')
parser.add_argument('--config', metavar="conf", nargs='?',
                    default="conf/config.json", help='Configuration file')
args = parser.parse_args()

config = get_configuration(args.config)
db = db_layer(args.config)

fe = concat_fe(args.config,
               [
                   clear_fe(args.config),
                   num_tokens_fe(args.config),
                   stop_words_fe(args.config),
                   punctuation_fe(args.config),
                   structure_fe(args.config),
                   char_distribution_fe(args.config)
               ])

if type(args.language) == str:
    args.language = [args.language]

for ln in args.language:
    print "Language:", ln

    authors = db.get_authors(ln)

    if args.clear[0]:
        print "Clearing features..."
        for id_author, author in enumerate(authors):
            db.clear_features(author, commit=True)

            if id_author % 10 == 0:
                print "%0.2f%%\r" % (id_author * 100.0 / len(authors)),
                os.sys.stdout.flush()

    if args.train[0]:
        print "Training features..."
        fe.train(authors)

    if args.compute[0]:
        print "Computing features..."
        for id_author, author in enumerate(authors):
            author = fe.compute(author)
            if id_author % 10 == 0:
                print "%0.2f%%\r" % (id_author * 100.0 / len(authors)),
                os.sys.stdout.flush()
        print
    print
