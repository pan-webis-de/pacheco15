#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils
from db_layer import db_layer
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(\
        description="Compute the dataset statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', metavar="conf", nargs='?',
                        default="conf/config.json", help='Configuration file')
    args = parser.parse_args()

    db = db_layer(args.config)
    languages = db.get_languages()
    authors = db.get_authors()
    print "total", len(authors)
    print "Number of languages:", len(languages)
    print "Languages:", ', '.join(languages)

    ln_sts = {}
    for ln in languages:
        ln_sts[ln] = {}

        authors_ids = db.get_authors(ln)
        authors = [db.get_author(id_) for id_ in authors_ids]
        ln_sts[ln]["num_authors"] = len(authors)

        n_docs_per_author = [len(a["documents"]) for a in authors]
        chars_per_author = [np.mean([len(c) for c in a["corpus"]])\
                               for a in authors]

        ln_sts[ln]["num_documents"] = sum(n_docs_per_author)
        ln_sts[ln]["avg_docs_per_author"] = np.mean(n_docs_per_author)
        ln_sts[ln]["std_docs_per_author"] = np.std(n_docs_per_author)
        ln_sts[ln]["min_docs_per_author"] = np.min(n_docs_per_author)
        ln_sts[ln]["max_docs_per_author"] = np.max(n_docs_per_author)
        ln_sts[ln]["avg_avg_chars_per_author"] = np.mean(chars_per_author)
        ln_sts[ln]["std_avg_chars_per_author"] = np.std(chars_per_author)
        ln_sts[ln]["min_avg_chars_per_author"] = np.min(chars_per_author)
        ln_sts[ln]["max_avg_chars_per_author"] = np.max(chars_per_author)

    print
    print "Documents per author"
    print "Lang & Num &   Avg &   Std & Min & Max\\\\"
    for ln in ln_sts:
        print "%4s & %3d & %5.2f & %5.2f & %3d & %3d\\\\" % \
            (ln,
             ln_sts[ln]["num_documents"],
             ln_sts[ln]["avg_docs_per_author"],
             ln_sts[ln]["std_docs_per_author"],
             ln_sts[ln]["min_docs_per_author"],
             ln_sts[ln]["max_docs_per_author"],
            )
    print
    print "Average document length (chars) per author"
    print "Lang &     Avg &     Std &  Min &  Max\\\\"
    for ln in ln_sts:
        print "%4s & %7.2f & %7.2f & %4d & %4d\\\\" % \
            (ln,
             ln_sts[ln]["avg_avg_chars_per_author"],
             ln_sts[ln]["std_avg_chars_per_author"],
             ln_sts[ln]["min_avg_chars_per_author"],
             ln_sts[ln]["max_avg_chars_per_author"],
            )
    print
    print "Bye!"


if __name__ == "__main__":
    main()
