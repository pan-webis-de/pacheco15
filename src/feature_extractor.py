# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut
# from nltk import BigramTagger as bt

from gensim import corpora, models
from collections import Counter
from datetime import datetime
from textblob import TextBlob
from operator import add
import numpy as np
import stop_words
import many_stop_words as msw
import tempfile
import copy
import json
import os
import re

import utils
import commands as cmd
import numpy
import math


class feature_extractor(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = utils.get_configuration(config_file)
        self.paragraph_re = r'.+\n'
        self.language = None
        self.regex = None
        self.db = None

    def set_db(self, db):
        self.db = db

    def get_paragraphs(self, documents):
        ret = [list(re.findall(self.paragraph_re, d + '\n')) \
                   for d in documents]
        ret = utils.flatten(ret)
        ret = [p for p in ret if len(p.split()) > 0]
        return ret

    def get_sentences(self, documents):
        ret = utils.flatten([TextBlob(d).sentences for d in documents])
        return [t.string for t in ret]

    def train(self, authors):
        pass

    def compute(self, author, known=True):

        if not type(author) == dict:
            author = self.db.get_author(author)

        if known:
            author = self.compute_features(author)

        else:
            documents_tmp = list(author["documents"])
            corpus_tmp = list(author["corpus"])
            features_tmp = copy.deepcopy(author["features"])

            author["documents"] = ["unknown.txt"]
            author["corpus"] = [self.db.get_unknown_document(author["id"])]
            author["features"] = {}

            author = self.compute_features(author)

            author["documents"] = documents_tmp
            author["corpus"] = corpus_tmp
            author["unknown_features"] = copy.deepcopy(author["features"])
            author["features"] = features_tmp

        self.db.update_author(author)

        return author

    def compute_features(author):
        return author

    def get_stop_words(self, lang):
        folder = self.config["stop_words"]
        f = open(self.config["stop_words"] + '/' + lang + '.sw')
        ret = f.read().decode('utf-8').split('\n')
        f.close()
        return ret

    def get_tokenizer(self):
        return RegexpTokenizer(self.regex)

class concat_fe(feature_extractor):
    def __init__(self, config_file, children=[]):
        self.children = list(children)
        super(concat_fe, self).__init__(config_file)

    def set_db(self, db):
        self.db = db
        for ch in self.children:
            ch.set_db(db)

    def compute_features(self, author):
        for ch in self.children:
            author = ch.compute_features(author)

        return author

    def train(self, authors):
        for ch in self.children:
            ch.train(authors)


class clear_fe(feature_extractor):
    def compute_features(self, author):
        return self.db.clear_features(author, commit=False)


class num_tokens_fe(feature_extractor):
    def __init__(self, config_file):
        super(num_tokens_fe, self).__init__(config_file)
        self.regex = r'\w+'

    def get_features(self, author, corpus, prefix):
        tokenizer = self.get_tokenizer()
        corpus = [tokenizer.tokenize(d) for d in corpus]
        unique_tokens = [list(set(t)) for t in corpus]

        # Number of tokens per document
        ntokens = map(len, corpus)
        author = self.db.set_feature(author, prefix + "tokens_avg",
                                     np.mean(ntokens))
        author = self.db.set_feature(author, prefix + "tokens_min",
                                     np.min(ntokens))
        author = self.db.set_feature(author, prefix + "tokens_max",
                                     np.max(ntokens))

        # Number of unique tokens per document (binary occurence)
        n_unique_tokens = [float(len(u)) / max(1, t) \
                            for u, t in zip(unique_tokens, ntokens)]
        author = self.db.set_feature(author, prefix + "unique_tokens_avg",
                                     np.mean(n_unique_tokens))
        author = self.db.set_feature(author, prefix + "unique_tokens_min",
                                     np.min(n_unique_tokens))
        author = self.db.set_feature(author, prefix + "unique_tokens_max",
                                     np.max(n_unique_tokens))
        return author

    def compute_features(self, author):
        author = self.get_features(author, author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author


class structure_fe(feature_extractor):
    def compute_features(self, author):
        documents = [TextBlob(d) for d in author["corpus"]]
        d_sentences = [d.sentences for d in documents]
        d_nsentences = [len(d) for d in d_sentences]

        paragraphs = [TextBlob(d) \
                        for d in self.get_paragraphs(author["corpus"])]
        p_sentences = [d.sentences for d in paragraphs]
        p_nsentences = [len(d) for d in p_sentences]

        author = self.db.set_feature(author, "document::sentences_min",
                                     np.min(d_nsentences))
        author = self.db.set_feature(author, "document::sentences_max",
                                     np.max(d_nsentences))
        author = self.db.set_feature(author, "document::sentences_avg",
                                     np.mean(d_nsentences))

        author = self.db.set_feature(author, "paragraph::sentences_min",
                                     np.min(p_nsentences))
        author = self.db.set_feature(author, "paragraph::sentences_max",
                                     np.max(p_nsentences))
        author = self.db.set_feature(author, "paragraph::sentences_avg",
                                     np.mean(p_nsentences))

        paragraphs_per_document = [len(self.get_paragraphs([d]))\
                                    for d in author["corpus"]]
        author = self.db.set_feature(author, "document::paragraph_min",
                                     np.min(paragraphs_per_document))
        author = self.db.set_feature(author, "document::paragraph_max",
                                     np.max(paragraphs_per_document))
        author = self.db.set_feature(author, "document::paragraph_avg",
                                     np.mean(paragraphs_per_document))

        return author


class stop_words_fe(feature_extractor):
    def __init__(self, config_file):
        super(stop_words_fe, self).__init__(config_file)
        self.regex = r'\w+'

    def get_features(self, author, corpus, prefix):
        
	tokenizer = self.get_tokenizer()
        corpus = [tokenizer.tokenize(d) for d in corpus]
        unique_tokens = [list(set(t)) for t in corpus]

        lang = self.db.get_author_language(author["id"])
        stopwords = self.stopwords[lang]
        documents = list(corpus)

        # Occurrences of the stop-words in the text
        ntokens = map(len, documents)
        stop_tokens = [[x for x in d if x in stopwords] for d in documents]
        n_sw = [float(len(sw)) / max(1, t)\
                  for sw, t in zip(stop_tokens, ntokens)]
        author = self.db.set_feature(author, prefix + "stop_words_avg",
                                     np.mean(n_sw))
        author = self.db.set_feature(author, prefix + "stop_words_min",
                                     np.min(n_sw))
        author = self.db.set_feature(author, prefix + "stop_words_max",
                                     np.max(n_sw))

        # Binary (unique) occurrences of the stop-words in the text
        unique_tks = [list(set(x)) for x in documents]
        n_unique_tokens = map(len, unique_tks)
        sw_unique = [[x for x in d if x in stopwords] for d in unique_tks]
        n_sw_unique = [float(len(sw)) / max(1, t)\
                        for sw, t in zip(sw_unique, n_unique_tokens)]
        author = self.db.set_feature(author, prefix + "unique_stop_words_avg",
                                     np.mean(n_sw_unique))
        author = self.db.set_feature(author, prefix + "unique_stop_words_min",
                                     np.min(n_sw_unique))
        author = self.db.set_feature(author, prefix + "unique_stop_words_max",
                                     np.max(n_sw_unique))

        ##TODO: include a BoW encoding the occurrences of each stop-word

        return author

    def compute_features(self, author):
        self.stopwords = {ln: self.get_stop_words(ln) \
                            for ln in self.db.get_languages()}
        
	author = self.get_features(author, author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author


class punctuation_fe(feature_extractor):
    def avg_min_max_char(self, documents, char):
        l = [d.count(char) for d in documents]
        if len(l) == 0:
            l = [0]

        return np.mean(l), np.min(l), np.max(l)

    def set_avg_min_max(self, author, punctuation, name, char):
        avg_char, min_char, max_char = self.avg_min_max_char(punctuation, char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_avg",
                                     avg_char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_min",
                                     min_char)
        author = self.db.set_feature(author,
                                     "punctuation_" + name + "_max",
                                     max_char)
        return author

    def get_features(self, author, corpus, prefix):
        punctuation = [filter(lambda x: not x.isalnum() and \
                                        not x.isspace(),
                              d) for d in corpus]
        len_punctuation = [len(x) for x in punctuation]

        author = self.db.set_feature(author, prefix + "punctuation_avg",
                                     np.mean(len_punctuation))
        author = self.db.set_feature(author, prefix + "punctuation_min",
                                     np.min(len_punctuation))
        author = self.db.set_feature(author, prefix + "punctuation_max",
                                     np.max(len_punctuation))

        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "points", '.')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "commas", ',')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "semi_colon", ';')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "question", '?')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "open_question", u'¿')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "exclamation", '!')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "open_exclamation", u'¡')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "double_quote", '"')
        author = self.set_avg_min_max(author, punctuation,
                                      prefix + "single_quote", '\'')

        return author

    def compute_features(self, author):
        author = self.get_features(author,
                                   author["corpus"],
                                   "document::")
        author = self.get_features(author,
                                   self.get_paragraphs(author["corpus"]),
                                   "paragraph::")
        author = self.get_features(author,
                                   self.get_sentences(author["corpus"]),
                                   "sentence::")
        return author


class spacing_fe(feature_extractor):

    def set_avg_min_max(self, author, name, elements):

        if len(elements) > 0:
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_avg",
                                         np.mean(elements))
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_min",
                                         np.min(elements))
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_max",
                                         np.max(elements))
        else:
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_avg",
                                         0.0)
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_min",
                                         0.0)
            author = self.db.set_feature(author,
                                         "spacing_" + name + "_max",
                                         0.0)
        return author

    def compute_features(self, author):

        num_consecutive_spaces = []
        num_empty_lines = []
        num_consecutive_spaces_beg = []
        num_consecutive_spaces_end = []

        for document in author["corpus"]:
            spaces = re.findall(r'\s+', document)
            num_consecutive_spaces += map(lambda x: len(x), spaces)

            spaces = re.findall(r'\.(\n+)', document)
            num_empty_lines += map(lambda x: len(x), spaces)

            lines = document.split("\n")
            lines = filter(lambda x: x is not "", lines)

            spaces = map(lambda x: re.match(r'^(\s+).*', x), lines)
            num_consecutive_spaces_beg += [len(x.group(1)) \
                                            if x is not None else 0 \
                                            for x in spaces]

            spaces = map(lambda x: re.match(r'.*(\s+)$', x), lines)
            num_consecutive_spaces_end += [len(x.group(1)) \
                                           if x is not None else 0 \
                                           for x in spaces]

        author = self.set_avg_min_max(author, "consecutive",
                                      num_consecutive_spaces)
        author = self.set_avg_min_max(author, "empty_lines",
                                      num_empty_lines)
        author = self.set_avg_min_max(author, "consecutive_beginning",
                                      num_empty_lines)
        author = self.set_avg_min_max(author, "consecutive_ending",
                                      num_empty_lines)

        return author


class char_distribution_fe(feature_extractor):
    def train(self, authors):
        self.chars = [self.db.get_author(a)["corpus"] for a in authors]
        self.chars = utils.flatten(utils.flatten(self.chars))
        self.chars = filter(lambda x: x.isalnum(), self.chars)
        self.chars = list(set([x.lower() for x in self.chars]))
        self.chars.sort()

    def compute_features(self, author):
        def get_distribution(document):
            distribution = filter(lambda x: x.isalnum(), document)
            distribution = [ch.lower() for ch in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for ch in self.chars:
                ret.append(float(distribution[ch]) / document_length)

            return ret

        # Bag-of-Words
        author_chars = [get_distribution(d) for d in author["corpus"]]
        author_chars = np.divide(np.sum(author_chars, axis=0),
                                 len(author_chars))

        for id_ch, (char, value) in enumerate(zip(self.chars, author_chars)):
            author = self.db.set_feature(author,
                                         "BoW::abc::" + char,
                                         value)

        # Digits, uppercase, lowercase
        doc_length = [len(d) for d in author["corpus"]]

        digits = [filter(lambda x: x.isdigit(), d) for d in author["corpus"]]
        digits = [float(len(x)) / y for (x, y) in zip(digits, doc_length)]
        author = self.db.set_feature(author, "num_digits_avg", np.mean(digits))
        author = self.db.set_feature(author, "num_digits_min", np.min(digits))
        author = self.db.set_feature(author, "num_digits_max", np.max(digits))

        upper = [filter(lambda x: x.isupper(), d) for d in author["corpus"]]
        upper = [float(len(x)) / y for (x, y) in zip(upper, doc_length)]
        author = self.db.set_feature(author, "uppercase_avg", np.mean(upper))
        author = self.db.set_feature(author, "uppercase_min", np.min(upper))
        author = self.db.set_feature(author, "uppercase_max", np.max(upper))

        lower = [filter(lambda x: x.islower(), d) for d in author["corpus"]]
        lower = [float(len(x)) / y for (x, y) in zip(lower, doc_length)]
        author = self.db.set_feature(author, "lower_case_avg", np.mean(lower))
        author = self.db.set_feature(author, "lower_case_min", np.min(lower))
        author = self.db.set_feature(author, "lower_case_max", np.max(lower))

        return author


class word_distribution_fe(feature_extractor):
    def __init__(self, config_file):
        super(word_distribution_fe, self).__init__(config_file)
        self.regex = r'\w+'

    def train(self, authors):
        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in self.db.get_languages()}
	lang = self.db.get_author_language(authors[0])
        self.words = [self.db.get_author(a)["corpus"] for a in authors]
        self.words = utils.flatten(self.words)
        tokenizer = self.get_tokenizer()
        self.words = map(lambda x: tokenizer.tokenize(x), self.words)
        self.words = utils.flatten(self.words)
        self.words = list(set([x.lower() for x in self.words]))
        self.words = filter(lambda x: x not in self.stopwords[lang],
                            self.words)
        self.words.sort()
        # print self.words

    def compute_features(self, author):
        def get_distribution(document):
            tokenizer = self.get_tokenizer()
            distribution = tokenizer.tokenize(document)
            distribution = [x.lower() for x in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for w in self.words:
                ret.append(float(distribution[w]) / document_length)

            return ret

        # Bag-of-Words
        author_words = [get_distribution(d) for d in author["corpus"]]
        # print sum(author_words[0])
        author_words = np.divide(np.sum(author_words, axis=0),
                                 len(author_words))
        # print sum(author_words)
        for id_w, (word, value) in enumerate(zip(self.words, author_words)):
            author = self.db.set_feature(author,
                                         "BoW::word::" + word,
                                         value)

        return author


class punctuation_ngrams_fe(feature_extractor):
    def __init__(self, config_file):
        super(punctuation_ngrams_fe, self).__init__(config_file)
        self.token_pattern = u'[,;\.:?!¿¡]+'
        self.ngram_x = 2
        self.ngram_y = 2

    def train(self, authors):
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        self.ngram_vectorizer = \
            CountVectorizer(ngram_range=(self.ngram_x, self.ngram_y),\
                                         token_pattern=self.token_pattern,\
                                         analyzer='word')
        self.ngram_vectorizer.fit(documents)
        # use only normalized term frequencies
        self.transformer = TfidfTransformer(use_idf=False)

    def compute_features(self, author):
        freq = self.ngram_vectorizer.transform(author["corpus"])
        freq = freq.toarray().astype(int)
        # normalized ngram frequencies
        norm_freq = self.transformer.fit_transform(freq).toarray()
        # average normalized frequencies among all author documents
        norm_freq = np.divide(np.sum(norm_freq, axis=0),
                                 len(norm_freq))

        ngrams = self.ngram_vectorizer.get_feature_names()
        for id_ngram, (ngram, value) in enumerate(zip(ngrams, norm_freq)):
            author = self.db.set_feature(author,
                                         "Ngram::punct::" + ngram,
                                         value)

        return author


class hapax_fe(feature_extractor):
    def __init__(self, config_file):

        super(hapax_fe, self).__init__(config_file)
        self.regex = r'\w+'

    def train(self, authors):
        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in self.db.get_languages()}
        
	def get_author_titles(author_corpus):
            author_titles = {}
            for i in author_corpus:
                author_titles[i.split('\n', 1)[0].strip().lower()] = i
            return author_titles

        lang = self.db.get_author_language(authors[0])
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        self.words = []
        self.titles = {}
        document_tokens = []
        title = ''
        tokenizer = self.get_tokenizer()
        for i in documents:
            title = i.strip().split('\n', 1)[0].lower()
            if not title in self.titles:
                document_tokens = [x.lower() \
                                   for x in tokenizer.tokenize(i)]
                self.titles[title] = Counter(document_tokens)
                self.titles[title] = set([i for i in self.titles[title]\
                                             if self.titles[title][i] == 1])
                self.words += document_tokens

        self.words = Counter([x.lower() for x in self.words])
        self.words = set([i for i in self.words if self.words[i] == 1])

        self.author_hapax = {}
        for a in authors:
            author_h = self.db.get_author(a)["corpus"]
            author_h = [i.strip().split('\n', 1)[0].lower() for i in \
                    author_h]
            author_h = utils.flatten([self.titles[i] for i in author_h])
            self.author_hapax[self.db.get_author(a)["id"]] = set(author_h)

    def compute_features(self, author):
        def get_author_titles(author_corpus):
            author_titles = {}
            for i in author_corpus:
                author_titles[i.strip().split('\n', 1)[0].lower()] = i
            return author_titles

        def uniqueness(title, document, _id):
            tokenizer = self.get_tokenizer()
            if not title in self.titles:
                words = [x.lower() \
                           for x in tokenizer.tokenize(document)]
                words = Counter(words)
                words = set([i for i in words if words[i] == 1])
            else:
                words = self.titles[title]

            #return len(self.author_hapax[_id].intersection(words)),\
            #                 len(words.difference(self.words))
            return len(words.intersection(self.words)),\
                   len(words.difference(self.words))

        # Bag-of-Words
        author_titles = get_author_titles(author["corpus"])
        uwf_max, hapax_max = -1, -1
        uwf_min, hapax_min = float('inf'), float('inf')
        uwf_avg, hapax_avg = 0, 0
        for i in author_titles:
            uwf, hp = uniqueness(i, author_titles[i], author["id"])
            uwf_max, hapax_max = max(uwf_max, uwf), max(hapax_max, hp)
            uwf_min, hapax_min = min(uwf_min, uwf), min(hapax_min, hp)
            uwf_avg += uwf
            hapax_avg += hp

        uwf_avg = uwf_avg / float(len(author_titles))
        hapax_avg = hapax_avg / float(len(author_titles))
        # print "avg: ", _avg, "      max: ", _max, "      min: ", _min
        # author_hapax = len(self.words.intersection(author_words))

        author = self.db.set_feature(author,
                    "style::single_unique_language_document_tokens_max",
                    hapax_max)
        author = self.db.set_feature(author,
                    "style::single_unique_language_document_tokens_min",
                    hapax_min)
        author = self.db.set_feature(author,
                    "style::single_unique_language_document_tokens_avg",
                    hapax_avg)
        author = self.db.set_feature(author,
                    "style::single_unique_language_tokens_repetition_max",
                    uwf_max)
        author = self.db.set_feature(author,
                    "style::single_unique_language_tokens_repetition_min",
                    uwf_min)
        author = self.db.set_feature(author,
                    "style::single_unique_language_tokens_repetition_avg",
                    uwf_avg)

        return author


class pos_fe(feature_extractor):
    #@ put k in config file
    def __init__(self, config_file, k=10):
        super(pos_fe, self).__init__(config_file)
        self.regex = r'\w+'
        self.lang_pos_defs = {}
        self.k_max = k

    def train(self, authors):

        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in self.db.get_languages()}
        
	lang = self.db.get_author_language(authors[0])
        if lang == 'GR':
            return

        a_titles = [self.db.get_author(a) for a in authors]
        a_titles = [[a['path'] + '/' + d for d in a['documents']] \
                    for a in a_titles]
        a_titles = utils.flatten(a_titles)
        tagger = 'src/pos_tagger/cmd/tree-tagger-' + \
                    self.config['languages'][lang] + ' '
        a_titles = ['cat ' + a + '|' + tagger for a in a_titles]
        self.lemmas = cmd.getoutput(';'.join(a_titles)).split('\n')
        self.lemmas = [i.split('\t') for i in self.lemmas if i[0] != '\t']
	self.lemmas = set([i[2] for i in self.lemmas if len(i) == 3])
        # unwanted = set(['<unknown>','@card@'])
        # [self.lemmas.discard(u) for u in unwanted]
        self.lemmas = list(self.lemmas)
        self.lemmas.sort()
        # print len(self.lemmas)

    def compute_features(self, author):
        #part of speech definition
        def get_pos_defs(lang):
            if lang in self.lang_pos_defs:
                return self.lang_pos_defs[lang]

            pos_defs = {}
            f = open('src/tagset/' + lang + '.csv', 'r')
            pos = ''
            for l in f.readlines():
                l = l.lower()
                aux = [i.rstrip().lstrip() for i in l.split(';')]
                #preserve order - pronoun contains noun
                if re.search('pronoun', aux[1]):
                    pos = 'pronoun'
                elif re.search('noun', aux[1]):
                    pos = 'noun'
                elif re.search('adverb', aux[1]):
                    pos = 'adverb'
                elif re.search('verb', aux[1]):
                    pos = 'verb'
                elif re.search('adjective', aux[1]):
                    pos = 'adjective'
                else:
                    pos = aux[1]

                pos_defs[aux[0]] = pos
                # Good for debug
                # print aux[0], ' -> ', pos, "\t|\t", aux[1]

            # print lang
            # for i in pos_defs:
            #     print i,": ", pos_defs[i]
            self.lang_pos_defs[lang] = pos_defs
            return pos_defs

        def lexical_density(lang, author, tagger_output):
            # LEXICAL DENSITY
            pd = get_pos_defs(lang)
            pos_x_doc = []
            for to in tagger_output:
                # print Counter([tag[1] for tag in to if not tag[1] in pd])
                pos_x_doc.append(Counter([pd[tag[1]] for tag in to \
                                            if tag[1] in pd]))

            # For Lexical density (Ld) -> Dense Content (Compact)
            # nlex: #interests, n: total tokens
            # ld = nlex/n
            interests = ['noun', 'adjective', 'verb', 'adverb']

            # ld for document
            ld_x_doc = []
            for doc in pos_x_doc:
                nlex = sum([doc[pos] for pos in doc if pos in interests])
                n = sum(doc.values())
                ld_x_doc.append(nlex / float(n))

            if len(ld_x_doc) == 0:
                max_ld = 0.0
                min_ld = 0.0
                mean_ld = 0.0
            else:
                max_ld = max(ld_x_doc)
                min_ld = min(ld_x_doc)
                mean_ld = numpy.mean(ld_x_doc)

            author = self.db.set_feature(author,
                        "style::lexical_density_max", max_ld)
            author = self.db.set_feature(author,
                        "style::lexical_density_min", min_ld)
            author = self.db.set_feature(author,
                        "style::lexical_density_avg", mean_ld)

        def word_diversity(author, tagger_output):
            # WORD DIVERSITY (WD)
            lemma_x_doc = []
            for to in tagger_output:
                lemma_x_doc.append(Counter([tag[2] for tag in to \
                                            if len(tag) > 2]))

            # WD = nlemma/n for each document
            # nlemma: #lemmas len(doc.keys()),
            # n: total tokens sum(doc.values())
            wd_x_doc = [len(doc.keys()) / float(sum(doc.values())) \
                        for doc in lemma_x_doc]
            if len(wd_x_doc) > 0:
                max_wd = max(wd_x_doc)
                min_wd = min(wd_x_doc)
                mean_wd = numpy.mean(wd_x_doc)
            else:
                max_wd = 0.0
                min_wd = 0.0
                mean_wd = 0.0
            author = self.db.set_feature(author,
                        "style::word_diversity_max", max_wd)
            author = self.db.set_feature(author,
                        "style::word_diversity_min", min_wd)
            author = self.db.set_feature(author,
                        "style::word_diversity_avg", mean_wd)

            return lemma_x_doc

        def lemmas_bog(author, lemma_x_doc):
            # Bag-of-Words of Lemmas
            author_lemmas = []
            for doc in lemma_x_doc:
                doc_len = float(sum(doc.values()))
                author_lemmas.append([doc[i] / doc_len for i in self.lemmas])

            #print len(author_lemmas), len(self.lemmas)
            if len(author_lemmas) > 0:
                author_lemmas = np.divide(np.sum(author_lemmas, axis=0),
                                          len(author_lemmas))
                #print author_lemmas
                for id_w, (word, value) in enumerate(zip(self.lemmas,
						     author_lemmas)):
		    author = self.db.set_feature(author,
						 "BoW::lemmas_avg::" + word,
						 value)
            else:
                for word in self.lemmas:
                    author = self.db.set_feature(author,
                                                 "BoW::lemmas_avg:" + word,
                                                 0.0)
                    

        def lemma_diversity(author, tagger_output, lemma_x_doc):
            # BOG - MAX K DIVERSE WORDS
            # unwanted = set(['<unknown>','@card@'])
            # Lemma distribution per document

            lemma_x_words_doc = []
            for to in tagger_output:
                lemma_x_words = {}
                for tag in to:
                    # print '\t'.join(tag)
                    if len(tag) < 3:  # or tag[2] in unwanted :
                        continue
                    if not tag[2] in lemma_x_words:
                        lemma_x_words[tag[2]] = set({})
                    lemma_x_words[tag[2]].add(tag[0])

                lemma_distrib = []
                n_log_lemmas = float(math.log(len(lemma_x_words) + 1) + 1)
                for l in self.lemmas:
                    freq = float(len(lemma_x_words[l])) \
                                if l in lemma_x_words else 0
                    lemma_distrib.append(freq / n_log_lemmas)
                lemma_x_words_doc.append(lemma_distrib)

            norm = math.log(numpy.mean([len(i) for i in lemma_x_doc]) + 1) + 1

            # Bag-of-Words Lemma Diversity Avg
            if len(lemma_x_words_doc) > 0:
                lemma_x_words_doc_avg = np.divide(np.sum(lemma_x_words_doc,
                                                     axis=0),
                                         len(lemma_x_words_doc))
                lemma_x_words_doc_avg = [i / norm for i in lemma_x_words_doc_avg]
                for id_w, (word, value) in enumerate(zip(self.lemmas,
                                                         lemma_x_words_doc_avg)):
                    author = self.db.set_feature(author,
                                     "BoW::word_diversity_per_lema_avg::" + word,
                                      value)
            else:
                for word in self.lemmas:
                    author = self.db.set_feature(author, 
                                                 "BoW::word_diversity_per_lema_avg::" + word,
                                                 0.0)
  
            # Bag-of-Words Lemma Diversity Max
            if len(lemma_x_words_doc) > 0:
                lemma_x_words_doc_max = np.max(lemma_x_words_doc, axis=0)
                lemma_x_words_doc_max = [i / norm for i in lemma_x_words_doc_max]
                for id_w, (word, value) in enumerate(zip(self.lemmas,
                                                     lemma_x_words_doc_max)):
                    author = self.db.set_feature(author,
                                     "BoW::word_diversity_per_lema_max::" + word,
                                      value)
            else:
                for word in self.lemmas:
                    author = self.db.set_feature(author,
                                     "BoW::word_diversity_per_lema_max::" + word,
                                      0.0)

        # Tag author documents
        lang = self.db.get_author_language(author['id'])
        if lang == 'GR':
            return author

        lang = self.config['languages'][lang]
        a_titles = [author['path'] + '/' + d for d in author['documents']]
        tagger = 'src/pos_tagger/cmd/tree-tagger-' + \
                    lang + ' '
        tagger_output = [cmd.getoutput('cat ' + at + '|' + tagger) \
                            for at in a_titles]
        # pxd[3:] because the first 3 lines of the output are flags
        tagger_output = [to.lower().split('\n') for to in tagger_output]
        tagger_output = [[p.split('\t') for p in to] for to in tagger_output]
	tagger_output = [[p for p in to if len(p) == 3] for to in tagger_output]

        lexical_density(lang, author, tagger_output)
        lemma_x_doc = word_diversity(author, tagger_output)
        lemmas_bog(author, lemma_x_doc)
        lemma_diversity(author, tagger_output, lemma_x_doc)

        return author


class stopword_distribution_fe(feature_extractor):
    def __init__(self, config_file):
        super(stopword_distribution_fe, self).__init__(config_file)
        self.regex = r'\w+'

    def train(self, authors):
        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in self.db.get_languages()}
        lang = self.db.get_author_language(authors[0])
        self.words = [self.db.get_author(a)["corpus"] for a in authors]
        self.words = utils.flatten(self.words)
        tokenizer = self.get_tokenizer()
        self.words = map(lambda x: tokenizer.tokenize(x), self.words)
        self.words = utils.flatten(self.words)
        self.words = list(set([x.lower() for x in self.words]))
        self.words = filter(lambda x: x in self.stopwords[lang], self.words)
        self.words.sort()

    def compute_features(self, author):
        def get_distribution(document):
            tokenizer = self.get_tokenizer()
            distribution = tokenizer.tokenize(document)
            distribution = [x.lower() for x in distribution]
            document_length = len(distribution)
            distribution = Counter(distribution)

            ret = []
            for w in self.words:
                ret.append(float(distribution[w]) / document_length)

            return ret

        # Bag-of-Words
        author_words = [get_distribution(d) for d in author["corpus"]]
        author_words = np.divide(np.sum(author_words, axis=0),
                                 len(author_words))

        for id_w, (word, value) in enumerate(zip(self.words, author_words)):
            author = self.db.set_feature(author,
                                         "BoW::word::" + word,
                                         value)
        return author


class word_topics_fe(feature_extractor):
    def __init__(self, config_file):
        super(word_topics_fe, self).__init__(config_file)
        self.regex = r'[\w\']+'
        self.k = 10

    def train(self, authors):
        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in self.db.get_languages()}
        lang = self.db.get_author_language(authors[0])
        # transform corpus into a list of preprocessed documents
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        tokenizer = self.get_tokenizer()
        documents = map(lambda x: tokenizer.tokenize(x), documents)
        documents = [map(lambda x: x.lower(), d) for d in documents]
        documents = [filter(lambda x: x not in self.stopwords[lang], d) \
                        for d in documents]
        # build topic model
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        documents = map(lambda x: self.dictionary.doc2bow(x), documents)
        self.model = models.LdaModel(documents, num_topics=self.k,
                                     id2word=self.dictionary, iterations=1000)


    def compute_features(self, author):
        topics = [0.0] * self.k
        lang = self.db.get_author_language(author["id"])
        for n, document in enumerate(author["corpus"]):
            tokenizer = self.get_tokenizer()
            document = tokenizer.tokenize(document)
            document = filter(lambda x: x not in self.stopwords[lang],
                              document)
            ext_topics = self.model[self.dictionary.doc2bow(document)]
            for _id, val in ext_topics:
                topics[_id] += val
        # average of the topic distribution
        topics = map(lambda x: x / (n + 1.0), topics)
        for (index, prop) in enumerate(topics):
            self.db.set_feature(author,
                                "LDA::word::" + str(index),
                                prop)

        return author


class stopword_topics_fe(feature_extractor):
    def __init__(self, config_file):
        super(stopword_topics_fe, self).__init__(config_file)
        self.regex = r'\w+'
        self.k = 10

    def train(self, authors):
        lang = self.db.get_author_language(authors[0])

        self.stopwords = {ln: self.get_stop_words(ln) \
                                for ln in [lang]}
        # print lang
        # print self.stopwords
        # transform corpus into a list of preprocessed documents
        documents = [self.db.get_author(a)["corpus"] for a in authors]
        documents = utils.flatten(documents)
        tokenizer = self.get_tokenizer()
        documents = map(lambda x: tokenizer.tokenize(x), documents)
        documents = [map(lambda x: x.lower(), d) for d in documents]
        # print documents
        documents = [filter(lambda x: x in self.stopwords[lang], d) \
                        for d in documents]
        # print documents
        # build topic model
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        documents = map(lambda x: self.dictionary.doc2bow(x), documents)
        self.model = models.LdaModel(documents, num_topics=self.k,
                                     id2word=self.dictionary, iterations=1000)
        #print self.model.show_topics()

    def compute_features(self, author):
        topics = [0.0] * self.k
        lang = self.db.get_author_language(author["id"])
        for n, document in enumerate(author["corpus"]):
            tokenizer = self.get_tokenizer()
            document = tokenizer.tokenize(document)
            document = filter(lambda x: x in self.stopwords[lang], document)
            ext_topics = self.model[self.dictionary.doc2bow(document)]
            for _id, val in ext_topics:
                topics[_id] += val
        # average of the topic distribution
        topics = map(lambda x: x / (n + 1.0), topics)
        for index, prop in enumerate(topics):
            self.db.set_feature(author,
                                "LDA::stopword::" + str(index),
                                prop)

        return author
