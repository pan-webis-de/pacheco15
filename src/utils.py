#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import json
import math
import os
from matplotlib import pyplot as plt
import re


def get_configuration(filename):
    try:
        f = open(filename)
        ret = json.loads(f.read())
        f.close()
        return ret
    except:
        return {}


def flatten(l):
    return [item for sublist in l for item in sublist]


def save_image_or_show(path, filename):
    if path is None:
        plt.show()
    else:
        f = os.path.join(path)

        if not os.path.exists(f):
            os.makedirs(f)

        f = os.path.join(f, filename)
        plt.savefig(f)
        plt.clf()


def get_auc(xs, ys):
    ret = 0.0
    for i in xrange(1, len(xs)):
        this_y = ys[i]
        prev_y = ys[i - 1]
        base = xs[i] - xs[i - 1]
        ret += base * (min(this_y, prev_y) + abs(this_y - prev_y) / 2)

    return ret


def get_feature_auc(feature, authors, plot=True, config=None, language=""):
    def distance(a, b):
        return abs(a - b)

    def up_to_k(values, threshold, first):
        id_v = first
        n = len(values)
        while id_v < n and values[id_v] <= threshold:
            id_v += 1
        return id_v

    values = [a["features"][feature] for a in authors]
    values.sort()

    all_distances = [distance(v, w) for v in values for w in values]
    all_distances = list(set(all_distances))
    all_distances.sort()

    distance_error = {d: 0 for d in all_distances}

    for v in values:
        v_distances = [distance(v, w) for w in values]
        v_distances.sort()
        remaining = list(v_distances)
        first = 0
        for d in all_distances:
            filtered_distances = up_to_k(remaining, d, first)
            first = filtered_distances
            distance_error[d] += 1.0 - float(filtered_distances - 1) / \
                                       len(values)

    xs = [d for d in distance_error]
    xs.sort()
    x_max_val = max(xs)
    x_min_val = min(xs)

    ys = [distance_error[d] for d in xs]
    y_max_val = max(ys)
    y_min_val = min(ys)

    xs = [(x - x_min_val) / (x_max_val - x_min_val + 1e-6) for x in xs]
    ys = [(y - y_min_val) / (y_max_val - y_min_val + 1e-6) for y in ys]

    area = get_auc(xs, ys)

    if plot:
        path = "output"
        if config is not None:
            path = os.path.join(config["results"], "features_auc", language)

        plt.plot(xs, ys)
        plt.title(feature + " - " + str(area))
        save_image_or_show(path, feature + ".png")

    return area

def remove_dirs(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
