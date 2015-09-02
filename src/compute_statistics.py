# -*- coding: utf-8 -*-

import sys
import os
import re
import argparse
import numpy as np

from utils import *
from db_layer import *
from feature_extractor import *


def global_statistics(args, config):
    print "Global statistics..."
    args.language.sort()
    f = open(os.path.join(config["results"], "global_statistics.tex"), 'w')

    for ln in args.language:
        authors = db.get_authors(ln)
        num_documents = [len(db.get_author(a)["corpus"]) for a in authors]
        f.write("\\textbf{%2s} & %3d & %4.2f & %4.2f & %d & %d & %4d\\\\\n" %\
                    (ln, len(authors),\
                     np.mean(num_documents), np.std(num_documents),\
                     np.min(num_documents), np.max(num_documents),
                     len(db.get_author(authors[0])["features"])
                    )
               )


def feature_curves(args, config):
    print "Curves..."
    auc_per_language = {}
    max_num_features = 0
    features_names = []

    for ln in args.language:
        print "Language:", ln

        authors = db.get_authors(ln)
        authors = [db.get_author(a, reduced=True) for a in authors]

        features = [a["features"].keys() for a in authors]
        features = list(set(utils.flatten(features)))
        features.sort()
        features_names += features

        areas = []
        for id_feature, feature in enumerate(features):
            area = utils.get_feature_auc(feature, authors, True, config, ln)
            areas.append((area, feature))

            if id_feature % 2 == 0:
                print "%0.2f%%\r" % (id_feature * 100.0 / len(features)),
                os.sys.stdout.flush()

        areas.sort()
        areas.reverse()
        auc_per_language[ln] = areas
        max_num_features = max(max_num_features, len(areas))

    features_names = list(set(features_names))
    max_ft = max(map(len, features_names))

    args.language.sort()

    f = open(os.path.join(config["results"], "features_auc", "results.tex"),
             'w')

    f.write("\\hline\n")
    for ln_id, ln in enumerate(args.language):
        if ln_id > 0:
            f.write(" & ")
        f.write("\multicolumn{2}{|c|}{\\textbf{%s}}" % ln)
    f.write("\\\\\n")
    f.write("\\hline\n")

    for f_id in range(max_num_features):
        for ln_id, ln in enumerate(args.language):
            if f_id < len(auc_per_language[ln]):
                if ln_id > 0:
                    f.write(" &"),
                f.write(u"%0.4f & " % auc_per_language[ln][f_id][0])
                ft = unicode(auc_per_language[ln][f_id][1])
                while len(ft) < max_ft:
                    ft = u' ' + ft
                f.write(ft.encode('utf8'))

            else:
                if ln_id > 0:
                    f.write(" &")
                f.write(" &")
        f.write("\\\\\n")
    f.write("\\hline\n")
    f.close()


def feature_histograms(args, config):
    print "Histograms..."

    for ln in args.language:
        print "Language:", ln

        authors = db.get_authors(ln)
        authors = [db.get_author(a, reduced=True) for a in authors]

        features = [a["features"].keys() for a in authors]
        features = list(set(utils.flatten(features)))
        features.sort()

        areas = []
        for id_feature, feature in enumerate(features):
            values = [a["features"][feature] for a in authors]
            xmax, xmin = max(values), min(values)
            y, x = np.histogram(values, bins=np.linspace(xmin - 1, xmax + 1,
                                                         (xmax - xmin)))
            nbins = y.size
            plt.hist(values, bins=nbins)
            plt.title(feature)
            save_image_or_show(os.path.join(config["results"],
                                            "histograms", ln),
                               feature + ".png")

            if id_feature % 2 == 0:
                print "%0.2f%%\r" % (id_feature * 100.0 / len(features)),
                os.sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(\
        description="Train and compute the authors' features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--language', metavar="lang", nargs='?',
                        default=["DU", "EN", "GR", "SP"],
                        help='Language')
    parser.add_argument('--config', metavar="conf", nargs='?',
                        default="conf/config.json", help='Configuration file')
    args = parser.parse_args()

    config = get_configuration(args.config)
    db = db_layer(args.config)

    if type(args.language) == str:
        args.language = [args.language]

    global_statistics(args, config)
    feature_curves(args, config)
    feature_histograms(args, config)
