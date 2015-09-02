import sys
import os
import re
import optparse
import numpy as np
import json
import commands as cmd

from src.utils import *
from src.importer import *
from src.db_layer_simple import *
from src.feature_extractor import *
from src.classifier import *


parser = optparse.OptionParser()
parser.add_option('-i', help='path to training corpus', dest='documents_path', type='string')
parser.add_option('-o', help='path to output directory', dest='model_path', type='string')
(opts, args) = parser.parse_args()

mandatories = ['documents_path', 'model_path']
for m in mandatories:
    if not opts.__dict__[m]:
        print "mandatory option is missing"
        parser.print_help()
        exit(-1)

documents_path = opts.documents_path
model_path = opts.model_path
contents_path = os.path.join(opts.documents_path, 'contents.json')

with open(contents_path) as contents_file:
    contents_dict = json.load(contents_file)

language = contents_dict["language"]
authors = contents_dict["problems"]
config_file = "conf/config_run.json"
config = get_configuration(config_file)

db = db_layer(language, authors, config_file, documents_path)

print "Language:", language
print "Number of examples in training set:", len(authors)

ln = db.get_language()

## clean authors folder
remove_dirs(os.path.join("features", ln, "Train"))

## features
fe = concat_fe(config_file,
               [
                   clear_fe(config_file),
                   pos_fe(config_file),
                   hapax_fe(config_file),
                   word_distribution_fe(config_file),
                   num_tokens_fe(config_file),
                   stop_words_fe(config_file),
                   punctuation_fe(config_file),
                   structure_fe(config_file),
                   char_distribution_fe(config_file),
                   spacing_fe(config_file),
                   punctuation_ngrams_fe(config_file),
                   stopword_topics_fe(config_file),
                   word_topics_fe(config_file)
               ])
fe.set_db(db)

print "Clearing features..."
for id_author, author in enumerate(authors):
    db.clear_features(author, commit=True)

    if id_author % 10 == 0:
        print "%0.2f%%\r" % (id_author * 100.0 / len(authors)),
        os.sys.stdout.flush()

print "Training features..."
fe.train(authors)
db.store_feature_extractor(fe, ln)

print "Computing features..."
for id_author, author in enumerate(authors):
    author = fe.compute(author, known=True)
    author = fe.compute(author, known=False)

    if (id_author + 1) % 10 == 0:
        print "%0.2f%%\r" % ((id_author + 1) * 100.0 / len(authors)),
        os.sys.stdout.flush()
print

print "Training model..."

models = [
          ("Weights", weighted_distance_classifier(config_file, ln)),
          ("reject-RF", reject_classifier(config_file, ln,
                                          rf_classifier(config_file,
                                                        ln))),
          ("adj-RF",  adjustment_classifier(config_file, ln,
                                            rf_classifier(config_file,
                                            ln))),
          ("rej-adj-RF",
           adjustment_classifier(config_file, ln,
                                 reject_classifier(config_file, ln,
                                           rf_classifier(config_file,
                                           ln)))),
          ("RF", rf_classifier(config_file, ln)),
          ("UBM", ubm(config_file, ln, fe,  n_pca=5, \
                             n_gaussians=2, r=8, normals_type='diag')),
         ]

model = model_selector(config_file, ln, [x[1] for x in models])
model.set_db(db)
model.train(authors)

print "Storing model on", opts.model_path, "..."
db.store_model(opts.model_path, model, fe)
