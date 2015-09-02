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
parser.add_option('-i', help='path to testing corpus', dest='documents_path', type='string')
parser.add_option('-m', help='path to classification model', dest='model_path', type='string')
parser.add_option('-o', help='path to output directory', dest='out_path', type='string')
(opts, args) = parser.parse_args()

mandatories = ['documents_path', 'model_path', 'out_path']
for m in mandatories:
    if not opts.__dict__[m]:
        print "mandatory option is missing"
        parser.print_help()
        exit(-1)


print "Performing test on input dataset", opts.documents_path

out_path = os.path.join(opts.out_path, 'answers.txt')
contents_path = os.path.join(opts.documents_path, 'contents.json')

with open(contents_path) as contents_file:
    contents_dict = json.load(contents_file)

language = contents_dict["language"]
authors = contents_dict["problems"]
config_file = "conf/config_run.json"
config = get_configuration(config_file)

db = db_layer(language, authors, config_file, opts.documents_path, False)

print "Language:", language
print "Number of examples in testing set:", len(authors)

ln = db.get_language()

## clean authors dir
remove_dirs(os.path.join("features", ln, "Test"))

print "Loading trained objects on", opts.model_path, "..."

try:	
    trained_object = db.get_model(opts.model_path)
except Exception as e:
    print str(e)
    print "Trained object could not be found."
    exit(-1)

fe = trained_object['fe']
fe.set_db(db)

print "Computing features..."
for id_author, author in enumerate(authors):
    author = fe.compute(author, known=True)
    author = fe.compute(author, known=False)

    if (id_author + 1) % 10 == 0:
        print "%0.2f%%\r" % ((id_author + 1) * 100.0 / len(authors)),
        os.sys.stdout.flush()
print

model = trained_object['model']
model.set_db(db)

if not os.path.exists(opts.out_path):
    os.makedirs(opts.out_path)

print "Predicting and writing to file..."
with open(out_path, 'w') as output_file:
    for id_author, author in enumerate(authors):
        pred_val = model.predict(author)
        output_file.write(author + " " + str(pred_val) + "\n")


