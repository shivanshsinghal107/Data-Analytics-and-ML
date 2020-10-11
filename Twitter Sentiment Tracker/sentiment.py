import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import random
import pickle
from statistics import mode
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
# from nltk.classify.scikitlearn import SklearnClassifier
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.svm import LinearSVC

class VotedClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for classifier in self._classifiers:
            v = classifier.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            v = classifier.classify(features)
            votes.append(v)
        majority = votes.count(mode(votes))
        return majority/len(votes)


docs_file = open('documents.pickle', 'rb')
documents = pickle.load(docs_file)
docs_file.close()

features_file = open('features.pickle', 'rb')
word_features = pickle.load(features_file)
features_file.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

feature_sets_file = open('feature_sets.pickle', 'rb')
feature_sets = pickle.load(feature_sets_file)
feature_sets_file.close()

random.shuffle(feature_sets)

train_set = feature_sets[10000:]
test_set = feature_sets[:10000]

open_file = open('original_nb.pickle', 'rb')
classifier = pickle.load(open_file)
open_file.close()

open_file = open('mnb_classifier.pickle', 'rb')
mnb_classifier = pickle.load(open_file)
open_file.close()

open_file = open('bnb_classifier.pickle', 'rb')
bnb_classifier = pickle.load(open_file)
open_file.close()

open_file = open('lr_classifier.pickle', 'rb')
lr_classifier = pickle.load(open_file)
open_file.close()

# open_file = open('sgd_classifier.pickle', 'rb')
# sgd_classifier = pickle.load(open_file)
# open_file.close()

open_file = open('svm_classifier.pickle', 'rb')
svm_classifier = pickle.load(open_file)
open_file.close()


voted_classifier = VotedClassifier(classifier,
                                   mnb_classifier,
                                   bnb_classifier,
                                   lr_classifier,
                                   svm_classifier)

def what_is_sentiment(text):
    features = find_features(text)
    return voted_classifier.classify(features), voted_classifier.confidence(features)
