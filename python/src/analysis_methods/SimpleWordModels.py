
import inspect, sys, os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
sys.path.append(os.path.join(cmd_folder,"../"))


from BaseAnalysisMethod import BaseAnalysisMethod

__author__ = 'kjoseph'

from collections import Counter, defaultdict
from utility import tab_stringify_newline

def prob_w_lambda(count, n, d, lambda_v):
    return (count + lambda_v) / (float(n + d * lambda_v))


class UnigramModel(BaseAnalysisMethod):
    def __init__(self, fold, data_dir, all_identities_filename, all_behaviors_filename,lambda_v, run_perplexity, run_prediction):

        BaseAnalysisMethod.__init__(self, fold, data_dir,
                                    all_identities_filename, all_behaviors_filename,
                                    run_perplexity, run_prediction,
                                    0, None)

        self.count_identities = Counter()
        self.count_behaviors = Counter()

        self.n_docs = 0

        self.n_identities = len(self.count_identities)
        self.n_behaviors = len(self.count_behaviors)
        self.lambda_v = lambda_v

    def train(self):
        for doc in self.docs:
            self.count_identities[doc.actor] += doc.count
            self.count_identities[doc.object] += doc.count
            self.count_behaviors[doc.behavior] += doc.count
            self.n_docs += doc.count

    def calculate_probability(self, actor,behavior,object, predict_only=''):

        if predict_only != '':
            if predict_only == 'a':
                return prob_w_lambda(self.count_identities[actor], self.n_docs, self.n_identities,self.lambda_v)
            elif predict_only == 'b':
                return prob_w_lambda(self.count_behaviors[behavior], self.n_docs, self.n_behaviors,self.lambda_v)
            elif predict_only == 'o':
                return prob_w_lambda(self.count_identities[object], self.n_docs, self.n_identities,self.lambda_v)
            else:
                raise Exception("WTF")

        return \
            prob_w_lambda(self.count_identities[actor], self.n_docs, self.n_identities,self.lambda_v) * \
            prob_w_lambda(self.count_behaviors[behavior], self.n_docs, self.n_behaviors,self.lambda_v) * \
            prob_w_lambda(self.count_identities[object], self.n_docs, self.n_identities,self.lambda_v)

    def get_name(self):
        return tab_stringify_newline(['Unigram',-1,"", "","","","",""],newline=False)


class BigramModel(BaseAnalysisMethod):
    def __init__(self, fold, data_dir, all_identities_filename, all_behaviors_filename,run_perplexity, run_prediction):
        BaseAnalysisMethod.__init__(self, fold, data_dir,
                                    all_identities_filename, all_behaviors_filename,
                                    run_perplexity, run_prediction,
                                    0, None)

        self.count_identities = Counter()
        self.count_behaviors = Counter()
        self.count_object_given_behavior = defaultdict(Counter)
        self.count_object_given_actor = defaultdict(Counter)
        self.count_behavior_given_actor = defaultdict(Counter)
        self.count_behavior_given_object = defaultdict(Counter)
        self.n_docs = 0
        self.n_identities = 0
        self.n_behaviors = 0

    def train(self):

        for doc in self.docs:
            self.count_identities[doc.actor] += doc.count
            self.count_identities[doc.object] += doc.count
            self.count_behaviors[doc.behavior] += doc.count

            self.n_docs += doc.count

            self.count_object_given_behavior[doc.object][doc.behavior] += doc.count
            self.count_object_given_actor[doc.object][doc.actor] += doc.count
            self.count_behavior_given_actor[doc.behavior][doc.actor] += doc.count
            self.count_behavior_given_object[doc.behavior][doc.object] += doc.count

        self.n_identities = len(self.count_identities)
        self.n_behaviors = len(self.count_behaviors)

    def calculate_probability(self, actor,behavior,object, predict_only=''):

        if predict_only != '':
            if predict_only == 'a':
                prob_w_lambda(self.count_identities[actor], self.n_docs, self.n_identities)
            elif predict_only == 'b':
                return prob_w_lambda(self.count_behaviors[behavior], self.n_docs, self.n_behaviors,1)*\
                       prob_w_lambda(self.count_behavior_given_actor[behavior][actor],
                                     self.count_behaviors[behavior], self.n_identities,1)*\
                       prob_w_lambda(self.count_behavior_given_object[behavior][object],
                                     self.count_behaviors[behavior], self.n_identities,1)
            elif predict_only == 'o':
                return prob_w_lambda(self.count_identities[object], self.n_docs, self.n_identities,1)*\
                       prob_w_lambda(self.count_object_given_behavior[object][behavior],
                                     self.count_identities[object], self.n_behaviors,1)*\
                       prob_w_lambda(self.count_object_given_actor[object][actor],
                                     self.count_identities[object], self.n_behaviors,1)
            else:
                raise Exception("WTF")

        return \
            prob_w_lambda(self.count_identities[actor], self.n_docs, self.n_identities)*\
            prob_w_lambda(self.count_behavior_given_identity[behavior][actor],
                          self.count_behaviors[behavior], self.n_identities) *\
            prob_w_lambda(self.count_identity_given_behavior[object][behavior],
                          self.count_identities[object], self.n_behaviors)

    def get_name(self):
        return tab_stringify_newline(['Bigram',-1,"", "","","","",""],newline=False)

