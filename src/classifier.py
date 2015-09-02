from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GMM, DPGMM
from sklearn.metrics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.stats import multivariate_normal

import utils
import copy
import math

#test
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


class classifier:
    def __init__(self, config, language):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.language = language
        self.feature_list = []
	self.db = None

    def set_db(self, db):
        self.db = db

    def train(self, authors):
        pass

    def predict(self, author):
        return 0.100

    def accuracy(self, authors):
        ret = 0.0
        gt = self.db.get_ground_truth(self.language)

        for author in authors:
            if (self.predict(author) >= 0.5) == (gt[author] >= 0.5):
                ret += 1.0

        return ret / float(len(authors))

    def auc(self, authors):
        probabilities = []
        targets = []

        gt = self.db.get_ground_truth(self.language)

        for author in authors:
            probabilities.append(self.predict(author))
            targets.append(gt[author])

        return roc_auc_score(targets, probabilities)

    def c_at_one(self, authors):
        n = float(len(authors))
        nc = 0
        nu = 0
        gt = self.db.get_ground_truth(self.language)

        for author in authors:
            if self.predict(author) == 0.5:
                nu += 1
            elif (self.predict(author) >= 0.5) == (gt[author] >= 0.5):
                nc += 1.0

        return (nc + (nu * nc / n)) / n

    def metrics(self, authors):
        acc_ret = 0.0
        gt = self.db.get_ground_truth(self.language)
        probabilities = []
        targets = []
        n = float(len(authors))
        nc = 0
        nu = 0

        for author in authors:
            prediction = self.predict(author)
            probabilities.append(prediction)
            targets.append(gt[author])

            if (prediction >= 0.5) == (gt[author] >= 0.5):
                acc_ret += 1.0

            if prediction == 0.5:
                nu += 1
            elif (prediction >= 0.5) == (gt[author] >= 0.5):
                nc += 1.0

        return acc_ret / float(len(authors)), \
               roc_auc_score(targets, probabilities), \
               (nc + (nu * nc / n)) / n

    def get_matrix(self, authors, known=True):
        self.feature_list = [a["features"].keys() for a in authors]
        self.feature_list = list(set([f for fts in self.feature_list \
                                        for f in fts]))
        self.feature_list.sort()

        samples = [[s["features" if known else "unknown_features"].get(f,
                                                                       np.nan)\
                    for f in self.feature_list] for s in authors]

        return np.asarray(samples)


class weighted_distance_classifier(classifier):
    def __init__(self, config, language):
        classifier.__init__(self, config, language)
        self.weights = {}
        self.threshold = 0.0

    def normal_p(self, mu, sigma, x):
        diff_x_mu = x - mu
        sigma2 = sigma * sigma

        if sigma2 == 0.0:
            if x == mu:
                return 1.0
            else:
                return 0.0

        return np.exp(- diff_x_mu * diff_x_mu / (2 * sigma2)) / \
               math.sqrt(2.0 * np.pi * sigma2)

    def distance(self, weights, descriptor, unknown):
        return sum([w * (abs(x - d) ** 2 + 1) / (abs(x - m) ** 2 + 1) \
                            for (w, m, d, x) in zip(weights, self.mean,
                                                    descriptor, unknown)])

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)

        self.scaler = None
        self.scaler = MinMaxScaler()
        self.scaler.fit(samples)

        self.pca = None
        self.pca = PCA(n_components=100)
        self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)

        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)

        distances = []

        gt = self.db.get_ground_truth(self.language)

        values = []

        for id_, (author, descriptor) in enumerate(zip(authors_id, samples)):
            self.train_weights(author, descriptor)

            unknown = self.db.get_unknown_document(author)
            unknown_descriptor = self.get_matrix([authors[id_]], False)

            if self.scaler:
                unknown_descriptor = self.scaler.transform(unknown_descriptor)
            if self.pca:
                unknown_descriptor = self.pca.transform(unknown_descriptor)

            unknown_descriptor = unknown_descriptor[0]

            target = gt[author]

            values.append((self.distance(self.weights[author],
                                        descriptor, unknown_descriptor),
                           target)
                         )


        values.sort()

        best_threshold = 0
        best_accuracy = len(filter(lambda (_, t): t < 0.100, values))
        next_accuracy = best_accuracy

        for i, (v, t) in enumerate(values):
            if t > 0.100:
                next_accuracy += 1
            else:
                next_accuracy -= 1

            if next_accuracy >= best_accuracy:
                best_accuracy = next_accuracy
                best_threshold = i

        self.threshold = values[best_threshold][0]

        #print best_threshold, self.threshold, \
              #best_accuracy * 100.0 / len(values)

    def predict(self, author_id):
        author = self.db.get_author(author_id, reduced=True)

        descriptor = self.get_matrix([author], True)

        if self.scaler:
            descriptor = self.scaler.transform(descriptor)
        if self.pca:
            descriptor = self.pca.transform(descriptor)

        descriptor = descriptor[0]

        if self.weights.get(author_id) is None:
            self.train_weights(author_id, descriptor)

        unknown_descriptor = self.get_matrix([author], False)

        if self.scaler:
            unknown_descriptor = self.scaler.transform(unknown_descriptor)
        if self.pca:
            unknown_descriptor = self.pca.transform(unknown_descriptor)

        unknown_descriptor = unknown_descriptor[0]

        if self.distance(self.weights[author_id],
                         descriptor, unknown_descriptor) < self.threshold:
            return 1.0
        else:
            return 0.0

        return 0.100

    def train_weights(self, author, descriptor):
        bounded_d = [min(max(mu - 2 * sigma, d), mu + 2 * sigma) \
                        for (d, mu, sigma) in zip(descriptor,
                                                  self.mean, self.std)]

        #self.weights[author] = [abs(d - m) ** 2 / (2 * s + 1e-7) + 1.0\
                                    #for (d, m, s) in zip(bounded_d,
                                                      #self.mean, self.std)]
        self.weights[author] = [2.0 - self.normal_p(m, s, d)\
                                    for (d, m, s) in zip(bounded_d,
                                                         self.mean, self.std)]
        total_w = sum(self.weights[author])
        self.weights[author] = [x / total_w for x in self.weights[author]]


class rf_classifier(classifier):
    def __init__(self, config, language):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.language = language
        self.feature_list = []
        self.rf_criterion = self.config["rf"][self.language]["criterion"]
        self.rf_num_estimators = self.config["rf"][self.language]["estimators"]
        self.prob_degree = 5
        self.use_adjustment = True
	self.db = None

    def get_composed_descriptor(self, known_descriptor, unknown_descriptor):
        return [((x - d) ** 2 + 1) / ((x - m) ** 2 + 1) \
                for (m, d, x) in zip(self.mean,
                                     known_descriptor, unknown_descriptor)]

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)
        self.scaler = None

        self.pca = None
        #self.pca = PCA(n_components=100)
        #self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)

        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)

        gt = self.db.get_ground_truth(self.language)

        new_samples = []
        new_targets = []

        for id_, (author, descriptor) in enumerate(zip(authors_id, samples)):
            unknown = self.db.get_unknown_document(author)
            unknown_descriptor = self.get_matrix([authors[id_]], False)

            if self.scaler:
                unknown_descriptor = self.scaler.transform(unknown_descriptor)
            if self.pca:
                unknown_descriptor = self.pca.transform(unknown_descriptor)

            unknown_descriptor = unknown_descriptor[0]
            target = gt[author]

            new_samples.append(self.get_composed_descriptor(descriptor,
                                                        unknown_descriptor))
            new_targets.append(target)

        # Fir the Random Forest
        new_samples = np.asarray(new_samples)
        new_targets = np.asarray(new_targets)
        self.rf = RandomForestClassifier(n_estimators=self.rf_num_estimators,
                                         criterion=self.rf_criterion,
                                         n_jobs=-1)
        self.rf.fit(new_samples, new_targets)

    def expand_prob(self, p, degree):
        return [p ** d for d in range(degree)]

    def predict(self, author_id):
        author = self.db.get_author(author_id, reduced=True)
        descriptor = self.get_matrix([author], True)

        if self.scaler:
            descriptor = self.scaler.transform(descriptor)
        if self.pca:
            descriptor = self.pca.transform(descriptor)

        descriptor = descriptor[0]

        unknown_descriptor = self.get_matrix([author], False)

        if self.scaler:
            unknown_descriptor = self.scaler.transform(unknown_descriptor)
        if self.pca:
            unknown_descriptor = self.pca.transform(unknown_descriptor)

        unknown_descriptor = unknown_descriptor[0]

        composed = self.get_composed_descriptor(descriptor, unknown_descriptor)

        prob = self.rf.predict_proba(composed)[0][1]
        return prob


class ubm(classifier):

    def __init__(self, config, language, fe, n_pca = 5, n_gaussians=2, \
                 r=16, normals_type='diag'):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.language = language
        self.feature_list = []
        self.weights = {}
        self.threshold = 0.0
        self.n_pca = n_pca
        self.r = r
        self.tp = normals_type
        self.components=n_gaussians
	self.db = None

    def plot_test(self):
        # Number of samples per component
        n_samples = 70

        # Generate random sample, two components
        np.random.seed(0)
        C = np.array([[0., -0.1], [1.7, .4]])
        X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
                  .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
        X = np.vstack((X, np.array([[5,4], [5,4.1],  [5,4]])))
        print X
        # Fit a mixture of Gaussians with EM using five components
        gmm = GMM(n_components=2, covariance_type=self.tp)
        gmm.fit(X)

        agm = GMM(n_components=2, covariance_type=self.tp)
        agm.weights_, agm.means_, agm.covars_ = \
                    self.em(gmm.weights_, gmm.means_, gmm.covars_, \
                            [X[len(X)-1]], self.r, True)

        color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

        for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                          (agm, 'Bayesian Adaptation')]):
            splot = plt.subplot(2, 1, 1 + i)
            Y_ = clf.predict(X)
            for i, (mean, covar, color) in enumerate(zip(
                    clf.means_, clf._get_covars(), color_iter)):
                v, w = linalg.eigh(covar)
                u = w[0] / linalg.norm(w[0])
                # as the DP will not use every component it has access to
                # unless it needs it, we shouldn't plot the redundant
                # components.
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0]+0.1**7)
                angle = 180 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.100)
                splot.add_artist(ell)

            plt.xlim(-10, 10)
            plt.ylim(-3, 6)
            plt.xticks(())
            plt.yticks(())
            plt.title(title)

        plt.show()
        exit(-1)


    def mvnpdf(self, mean, covar, samples):
        multivariate_normal.pdf(samples, mean=mean, cov=covar)
        return multivariate_normal.pdf(samples, mean=mean, cov=covar)
               # allow_singular=True

    def alfa(self, n, r):
        return n/(n+r)

    # x is a vector of samples
    def em(self, weights, means, covars,  samples, r, debug=False):
        # print '\nAdaptation'
        for lll in range(0,1):
            pr = np.array(
                [
                    weights[i]*self.mvnpdf(means[i],covars[i],samples) + 0.1**7 \
                        for i in range(0,len(means))
                ]
            ).T

            if debug:
                print '\nSamples: ', samples
                print '\nPr: ', pr

            if len(samples) == 1:
                pr = np.array([pr])

            if debug:
                print '\nNormalizing Factor: ',map(sum, pr)
            pr = np.array([p/s for (p, s) in zip(pr,map(sum, pr))]).T
            ns = map(sum, pr)



            new_means = [sum([p*s for p,s in zip(ps,samples)])/ns[i] \
                                for i, ps in enumerate(pr)]

            new_covars = [sum([p*(s**2) for p,s in zip(ps,samples)])/ns[i] \
                                for i, ps in enumerate(pr)]


            alfas = [self.alfa(n, r) for n in ns]
            # Bayesian adaptation
            t = len(samples)
            adapted_weights = [a*n/t + (1-a)*w \
                            for a, n, w in zip(alfas, ns, weights)]
            adapted_weights = adapted_weights/sum(adapted_weights)

            adapted_means = [a*nm + (1-a)*m \
                            for a, nm, m in zip(alfas, new_means, means)]
            adapted_means = np.array(adapted_means)

            adapted_covars = [a*nc + ((1-a)*(c+m**2) - am**2) for a, nc, c, am, m \
                                in zip(alfas, new_covars, covars, adapted_means, means)]

            adapted_covars = np.array([ac + 0.1**7 for ac in adapted_covars])

            if debug:
                print '\nNs: ', ns
                print '\nNormalized -Prs*Samples: ', pr, '*', samples, '/', ns
                print '\nNew Means: ',new_means
                print '\nNew Covars: ',new_covars
                print '\nweights'
                print weights
                print adapted_weights
                print sum(adapted_weights)
                print '\nmeans'
                print means
                print adapted_means
                print '\ncovars'
                print covars
                print adapted_covars

            weights = adapted_weights
            means =adapted_means
            covars = adapted_covars

        return adapted_weights, adapted_means, adapted_covars

    def train(self, authors_id, debug=False):

        if debug:
            self.plot_test()

        authors = [self.db.get_author(a, True) for a in authors_id]
        samples = self.get_matrix(authors)

        self.scaler = MinMaxScaler()
        self.scaler.fit(samples)

        self.pca = None
        self.pca = PCA(n_components=self.n_pca)
        self.pca.fit(samples)

        if self.scaler:
            samples = self.scaler.transform(samples)
        if self.pca:
            samples = self.pca.transform(samples)

        self.mean = np.mean(samples, axis=0)
        self.std = np.std(samples, axis=0)

        # print samples.shape
        distances = []

        gt = self.db.get_ground_truth(self.language)

        n_classes = len(np.unique(gt.values()))
        self.bg_classifier = GMM(n_components=self.components, covariance_type=self.tp)

                           #'spherical', 'diag', 'tied', 'full'])
        self.bg_classifier.fit(samples)
        ws = self.bg_classifier.weights_
        ms = self.bg_classifier.means_
        cvs = self.bg_classifier.covars_
        values = []
        for id_, (author, descriptor) in enumerate(zip(authors_id, samples)):
            unknown = self.db.get_unknown_document(author)
            unknown_descriptor = self.get_matrix([authors[id_]], False)
            if self.scaler:
                unknown_descriptor = self.scaler.transform(unknown_descriptor)
            if self.pca:
                unknown_descriptor = self.pca.transform(unknown_descriptor)
            ud = unknown_descriptor[0]

            target = gt[author]

            agm = GMM(n_components=1,covariance_type=self.tp,init_params='')
            agm.weights_, agm.means_, agm.covars_ = \
                    self.em(ws, ms, cvs,  [descriptor], self.r)
            # print agm.score(descriptor) > self.bg_classifier.score(descriptor), agm.score(descriptor), self.bg_classifier.score(descriptor)
            values.append((agm.score(ud)/self.bg_classifier.score(ud),target))


        values.sort()
        best_threshold = 0
        best_accuracy = len(filter(lambda (_, t): t < 0.100, values))
        next_accuracy = best_accuracy

        for i, (v, t) in enumerate(values):
            if t > 0.100:
                next_accuracy += 1
            else:
                next_accuracy -= 1

            if next_accuracy >= best_accuracy:
                best_accuracy = next_accuracy
                best_threshold = i
        self.threshold = values[best_threshold][0]
        # print best_threshold, self.threshold, \
        #       best_accuracy * 100.0 / len(values)
        print '    train: ', (best_accuracy * 100.0 / len(values))

    def predict(self, author_id):
        author = self.db.get_author(author_id, reduced=True)

        descriptor = self.get_matrix([author], True)
        if self.scaler:
            descriptor = self.scaler.transform(descriptor)
        if self.pca:
            descriptor = self.pca.transform(descriptor)
        descriptor = descriptor[0]

        unknown_descriptor = self.get_matrix([author], False)
        if self.scaler:
            unknown_descriptor = self.scaler.transform(unknown_descriptor)
        if self.pca:
            unknown_descriptor = self.pca.transform(unknown_descriptor)
        ud = unknown_descriptor[0]

        ws = self.bg_classifier.weights_
        ms = self.bg_classifier.means_
        cvs = self.bg_classifier.covars_

        agm = GMM(n_components=self.components, covariance_type=self.tp)
        agm.weights_, agm.means_, agm.covars_ = \
                self.em(ws, ms, cvs,  [descriptor], self.r)

        if agm.score(ud)/self.bg_classifier.score(ud) < self.threshold:

            return 1.0
        else:
            return 0.0

        return 0.100

    def train_weights(self, author, descriptor):
        bounded_d = [min(max(mu - 2 * sigma, d), mu + 2 * sigma) \
                        for (d, mu, sigma) in zip(descriptor,
                                                  self.mean, self.std)]

        self.weights[author] = [abs(d - mu) / (2.0 * s + 1e-7) + 1.0 \
                                    for (d, s) in zip(bounded_d, self.std)]
        total_w = sum(self.weights[author])
        self.weights[author] = [x / total_w for x in self.weights[author]]


class adjustment_classifier(classifier):
    def __init__(self, config, language, classifier):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.language = language

        self.prob_degree = 3
        self.rate = 0.8

        self.classifier = classifier
	self.db = None

    def set_db(self, db):
        self.db = db
        self.classifier.set_db(db)

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        gt = self.db.get_ground_truth(self.language)

        pos = [a for a in authors_id if gt[a] == 1.0]
        neg = [a for a in authors_id if gt[a] == 0.0]

        tr = pos[: int(self.rate * len(pos))] + \
             neg[: int(self.rate * len(neg))]
        ts = pos[int(self.rate * len(pos)):] + neg[int(self.rate * len(neg)):]

        self.classifier.train(tr)

        # Fit a linear model to adjust the probabilities
        probs = [(self.classifier.predict(a), gt[a]) for a in ts]

        probs = [(self.expand_prob(p, self.prob_degree), t) for p, t in probs]
        probs_X = np.asarray([p for p, t in probs])
        probs_y = np.asarray([t for p, t in probs])

        self.lr_probs = LinearRegression()
        self.lr_probs.fit(probs_X, probs_y)

        self.classifier.train(authors_id)

    def expand_prob(self, p, degree):
        return [p ** d for d in range(degree)]

    def predict(self, author_id):
        prob = self.classifier.predict(author_id)
        expanded_prob = self.expand_prob(prob, self.prob_degree)
        adjusted_prob = self.lr_probs.predict(expanded_prob)[0]
        if (adjusted_prob >= 0.5) == (prob >= 0.5):
            return max(0.0, min(1.0, adjusted_prob))
        else:
            return prob

class reject_classifier(classifier):
    def __init__(self, config, language, classifier):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.db = None
        self.language = language

        self.left_threshold = 0.5
        self.right_threshold = 0.5

        self.classifier = classifier
        self.rate = 0.8

    def set_db(self, db):
        self.db = db
        self.classifier.set_db(db)

    def train(self, authors_id):
        def map_value(x):
            if x < self.left_threshold:
                return x
            elif x < self.right_threshold:
                return 0.5
            else:
                return x

        def c_at_one_aux(pt):
            n = float(len(pt))
            nc = 0
            nu = 0

            for p, t in pt:
                if map_value(p) == 0.5:
                    nu += 1
                elif (map_value(p) >= 0.5) == (t >= 0.5):
                    nc += 1.0

            return (nc + (nu * nc / n)) / n

        authors = [self.db.get_author(a, True) for a in authors_id]
        gt = self.db.get_ground_truth(self.language)

        pos = [a for a in authors_id if gt[a] == 1.0]
        neg = [a for a in authors_id if gt[a] == 0.0]

        tr = pos[: int(self.rate * len(pos))] + \
             neg[: int(self.rate * len(neg))]
        ts = pos[int(self.rate * len(pos)):] + neg[int(self.rate * len(neg)):]

        self.classifier.train(tr)

        # Fit a linear model to adjust the probabilities
        probs = [(self.classifier.predict(a), gt[a]) for a in ts] + \
                [(0.5, 0.5)]

        #print "len probs/test", len(probs), len(ts)
        #for x in probs:
        #    print "\tinner tuple", len(x)

        probs = list(set(probs))
        probs.sort()

        best_left = 0.5
        best_right = 0.5
        best_c_at_1 = 0.0

        for i in range(len(ts)):
            for j in range(i, len(ts)):
                #print "i", i, "j", j
                #print "probs[i]", probs[i]
                #print "probs[j]", probs[j]
                self.left_threshold = probs[i][0]
                self.right_threshold = probs[j][0]

                if self.left_threshold > 0.5 or \
                        self.right_threshold < 0.5:
                    break

                next_c_at_1 = c_at_one_aux(probs)

                if next_c_at_1 > best_c_at_1:
                    best_left = self.left_threshold
                    best_right = self.right_threshold
                    best_c_at_1 = next_c_at_1

        self.left_threshold = best_left
        self.right_threshold = best_right

        self.classifier.train(authors_id)

    def expand_prob(self, p, degree):
        return [p ** d for d in range(degree)]

    def predict(self, author_id):
        prob = self.classifier.predict(author_id)
        if prob < self.left_threshold:
            return prob
        elif prob <= self.right_threshold:
            return 0.5
        else:
            return prob


class model_selector(classifier):
    def __init__(self, config, language, classifier_list):
        self.config_file = config
        self.config = utils.get_configuration(config)
        self.language = language

        self.classifier = classifier
        self.classifier_list = list(classifier_list)
        self.rate = 0.8
        self.db = None

    def set_db(self, db):
        self.db = db
        for classifier_i in self.classifier_list:
            classifier_i.set_db(db)

    def train(self, authors_id):
        authors = [self.db.get_author(a, True) for a in authors_id]
        gt = self.db.get_ground_truth(self.language)

        #print gt

        pos = [a for a in authors_id if gt[a] == 1.0]
        neg = [a for a in authors_id if gt[a] == 0.0]

        tr = pos[: int(self.rate * len(pos))] + \
             neg[: int(self.rate * len(neg))]
        ts = pos[int(self.rate * len(pos)):] + neg[int(self.rate * len(neg)):]


        rankings = []

        for i, clf in enumerate(self.classifier_list):
            clf.train(tr)
            _, auc, catone = clf.metrics(ts)
            rankings.append((catone * auc, i))
            print i, auc, catone, auc * catone

        rankings.sort()

        print "best model", rankings[-1]
        self.classifier = self.classifier_list[rankings[-1][1]]
        self.classifier.train(authors_id)

    def predict(self, author_id):
        return self.classifier.predict(author_id)
