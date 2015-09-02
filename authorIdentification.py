#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np
import json
import commands as cmd

from src.utils import *
from src.importer import *
from src.db_layer import *
from src.feature_extractor import *
from src.classifier import *

if __name__ != '__main__':
    os.sys.exit(1)


parser = argparse.ArgumentParser(\
    description="Imports dataset and trains and computes authors' features.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cleardataset', metavar="C", nargs=1,
                    default=[0], type=int,
                    help='Clear imported dataset')
parser.add_argument('--clearfeatures', metavar="C", nargs=1,
                    default=[0], type=int,
                    help='Clear previous features (0-1)')
parser.add_argument('--language', metavar="lang", nargs='?',
                    default=['EN', 'DU', 'GR', 'SP'],
                    help='Only handles the given languages')
parser.add_argument('-i', metavar="path", nargs='?',
                    default=[''], help='Importer input path.')
parser.add_argument('-o', metavar="path", nargs='?',
                    default=['dataset/'], help='Output path.')
parser.add_argument('--train', metavar="T", nargs=1,
                    default=[1], type=int,
                    help='Train features (0-1)')
parser.add_argument('--compute', metavar="C", nargs=1,
                    default=[1], type=int,
                    help='Compute features (0-1)')
parser.add_argument('--train-model', metavar="TM", nargs=1,
                    default=[1], type=int,
                    help='Train the model (0-1)')
parser.add_argument('--config', metavar="conf", nargs='?',
                    default="conf/config.json", help='Configuration file')
args = parser.parse_args()

config = get_configuration(args.config)

if type(args.i) != str:
    args.i = args.i[0]

if type(args.o) != str:
    args.o = args.o[0]

if type(args.language) == str:
    args.language = [args.language]

print args.language

if args.i != '':
    clear(args.language, args.o, bool(args.cleardataset[0]))
    import_languages(config, args.language, args.i, args.o)

db = db_layer(args.config)


for ln in args.language:
    fe = concat_fe(args.config,
                   [
                       clear_fe(args.config),
                       pos_fe(args.config),
                       hapax_fe(args.config),
                       word_distribution_fe(args.config),
                       num_tokens_fe(args.config),
                       stop_words_fe(args.config),
                       punctuation_fe(args.config),
                       structure_fe(args.config),
                       char_distribution_fe(args.config),
                       spacing_fe(args.config),
                       punctuation_ngrams_fe(args.config),
                       stopword_topics_fe(args.config),
                       word_topics_fe(args.config)
                   ])
    fe.set_db(db)
    print "Language:", ln

    authors = db.get_authors(ln)

    if args.clearfeatures[0]:
        print "Clearing features..."
        for id_author, author in enumerate(authors):
            db.clear_features(author, commit=True)

            if id_author % 10 == 0:
                print "%0.2f%%\r" % (id_author * 100.0 / len(authors)),
                os.sys.stdout.flush()

    if args.train[0]:
        print "Training features..."
        fe.train(authors)
        db.store_feature_extractor(fe, ln)
    else:
        fe = db.get_feature_extractor(ln)
        pass

    if args.compute[0]:
        print "Computing features..."
        for id_author, author in enumerate(authors):

            author = fe.compute(author, known=True)
            author = fe.compute(author, known=False)

            if (id_author + 1) % 10 == 0:
                print "%0.2f%%\r" % ((id_author + 1) * 100.0 / len(authors)),
                os.sys.stdout.flush()
        print
    print

    if args.train_model[0]:
        #random.shuffle(authors)
        gt = db.get_ground_truth(ln)
        pos = [a for a in authors if gt[a] == 1.0]
        neg = [a for a in authors if gt[a] == 0.0]

        rate = 0.7
        tr = pos[: int(rate * len(pos))] + neg[: int(rate * len(neg))]
        ts = pos[int(rate * len(pos)):] + neg[int(rate * len(neg)):]

        models = [
                  ("Weights", weighted_distance_classifier(args.config, ln)),
                  ("reject-RF", reject_classifier(args.config, ln,
                                                  rf_classifier(args.config,
                                                                ln))),
                  ("adj-RF",  adjustment_classifier(args.config, ln,
                                                    rf_classifier(args.config,
                                                    ln))),
                  ("rej-adj-RF",
                   adjustment_classifier(args.config, ln,
                                         reject_classifier(args.config, ln,
                                                   rf_classifier(args.config,
                                                   ln)))),
                  ("RF", rf_classifier(args.config, ln)),
                  ("UBM", ubm(args.config, ln, fe,  n_pca=5, \
                                     n_gaussians=2, r=8, normals_type='diag')),
                 ]

        model = model_selector(args.config, ln, [x[1] for x in models])
        model.set_db(db)
        model.train(tr)
        metrics = model.metrics(ts)
        print "Acc: %0.4f" % metrics[0]
        print "AUC: %0.4f" % metrics[1]
        print "c@1: %0.4f" % metrics[2]
        print "Ranking: %0.4f" % (metrics[1] * metrics[2])
        print
