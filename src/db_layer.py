# -*- coding: utf-8 -*-

from datetime import datetime
import pickle
import numpy as np

import tempfile
import json
import os

import utils


class db_layer:
    def __init__(self, config_filename):
        self.config_filename = config_filename
        self.config = utils.get_configuration(config_filename)
        self.path = self.config["dataset"]
        self.languages = self.get_languages()

    def get_languages(self):
        ret = [d for d in os.listdir(self.path)]
        ret = filter(lambda x: os.path.isdir(os.path.join(self.path, x)), ret)
        return ret

    def get_authors(self, language=None):
        ret = []
        for ln in (self.languages if language is None else [language]):
            next_ln = [d for d in os.listdir(os.path.join(self.path, ln))]
            ret += next_ln

        return ret

    def get_author_language(self, id_):
        return filter(lambda x: x == id_[: len(x)], self.languages)[0]

    def get_author_path(self, id_):
        return os.path.join(self.path, self.get_author_language(id_), id_)

    def get_author_descriptor_file(self, id_):
        return os.path.join(self.get_author_path(id_), "author.json")

    def get_author_documents(self, id_):
        path = self.get_author_path(id_)
        ret = [d for d in os.listdir(path) \
                if os.path.isfile(os.path.join(path, d)) and \
                   d.startswith("known")
              ]
        return ret

    def initialize_author(self, id_):
        author = \
            {"id": id_,
             "documents": self.get_author_documents(id_),
             "corpus": [],
             "features": {},
             "path": self.get_author_path(id_),
            }

        self.update_author(author)
        return author

    def update_author(self, author):
        id_ = author["id"]
        path = self.get_author_path(id_)
        tmp_corpus = author["corpus"]
        author["corpus"] = None

        # Create directory
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # Force atomic write to file
        with tempfile.NamedTemporaryFile(
              'w', dir=os.path.dirname(self.path), delete=False) as tf:
            tf.write(json.dumps(author))
            tempname = tf.name
        try:
            os.rename(tempname, self.get_author_descriptor_file(id_))
        except:
            os.remove(self.get_author_descriptor_file(id_))
            os.rename(tempname, self.get_author_descriptor_file(id_))

        author["corpus"] = tmp_corpus

    def get_author(self, id_, reduced=False):
        author = {}
        filename = self.get_author_descriptor_file(id_)
        author_path = self.get_author_path(id_)

        if not os.path.isfile(filename):
            author = self.initialize_author(id_)
        else:
            f = open(filename)
            author = json.load(f)
            f.close()

        if not reduced:
            corpus = []
            author["documents"].sort()

            for d in author["documents"]:
                f = open(os.path.join(author_path, d))
                corpus.append(f.read().decode("utf-8"))
                f.close()

            author["corpus"] = corpus
        else:
            author["corpus"] = []

        return author

    def get_unknown_document(self, id_):
        path = os.path.join(self.get_author_path(id_), "unknown.txt")
        f = open(path)
        ret = f.read().decode("utf-8")
        f.close()

        return ret

    def set_feature(self, author, ft_name, ft_value, commit=False):
        author["features"][ft_name] = ft_value

        if commit:
            self.update_author(author)

        return author

    def clear_features(self, author, commit=False):
        if type(author) != dict:
            author = self.get_author(author)

        author["features"] = {}

        if commit:
            self.update_author(author)

        return author

    def feature_extractor_path(self, language):
        return os.path.join(self.config["pickle"], language, "fe.pickle")

    def store_feature_extractor(self, fe, language):
        fe_path = self.feature_extractor_path(language)

        if not os.path.exists(os.path.dirname(fe_path)):
            os.makedirs(os.path.dirname(fe_path))

        f = open(fe_path, 'wb')
        pickle.dump(fe, f, protocol=2)
        f.close()

    def get_feature_extractor(self, language):
        fe_path = self.feature_extractor_path(language)
        f = open(fe_path, 'rb')
        ret = pickle.load(f)
        f.close()

        return ret

    def get_ground_truth(self, language):
        f = open(os.path.join(self.config["dataset"], language + "_truth.txt"))
        ret = f.readlines()
        f.close()

        ret = [x.strip().split() for x in ret]
        ret = {x[0]: 1.0 if x[1] == 'Y' else 0.0 for x in ret}

        return ret
