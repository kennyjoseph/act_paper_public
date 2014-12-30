
__author__ = 'kjoseph'


import inspect, sys, os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
sys.path.append(os.path.join(cmd_folder,"../"))

from analysis_methods import BaseAnalysisMethod

from initialize_functions import *
from functions import *
from deflection import *
import numpy as np
from array import array
from collections import Counter, defaultdict
from SimpleWordModels import prob_w_lambda
from math import exp

class SimpleACT(BaseAnalysisMethod):

    def __init__(self, fold,data_dir,
                 all_identities_filename, all_behaviors_filename,
                 run_perplexity, run_prediction,
                 act_data_filename, act_equation_filename,
                 rand_eq, rand_vals):

        BaseAnalysisMethod.__init__(self, fold, data_dir,
                                    all_identities_filename,all_behaviors_filename,
                                    run_perplexity,run_prediction,1,None)

        self.identity_to_mu_sigma, self.behavior_to_mu_sigma = read_in_act_data(act_data_filename, 1, 1, 1, 1,1)

        self.init_rand = rand_vals
        if rand_vals:
            self.randomize(self.identity_to_mu_sigma,1,1,1,1,1)
            self.randomize(self.behavior_to_mu_sigma,1,1,1,1,1)

        ##get equations
        self.t, self.M = get_t_and_M_from_file(act_equation_filename, FUNDAMENTALS)
        self.eq_shuffle = rand_eq
        if rand_eq:
            np.random.shuffle(self.M)
        #get our deflection equations
        self.deflection_computation = get_coefficients(self.M, self.t)

        self.count_identities = Counter()
        self.count_behaviors = Counter()
        self.count_object_given_behavior = defaultdict(Counter)
        self.count_object_given_actor = defaultdict(Counter)
        self.count_behavior_given_actor = defaultdict(Counter)
        self.count_behavior_given_object = defaultdict(Counter)
        self.n_docs = 0
        self.n_identities = 0
        self.n_behaviors = 0


    def randomize(self, dat, init_sd, n_topics,dir_prior,k_val,v_val):
        for k in dat.keys():
            rand_v = np.random.uniform(-4.3, 4.3, 3)
            ps = ParameterStruct(k, 0, k_val, v_val,
                                [rand_v[0], init_sd],
                                [rand_v[1], init_sd],
                                [rand_v[2], init_sd], n_topics, dir_prior)
            dat[k] = ps

    def calculate_probability(self, actor, behavior, object, predict_only = ''):

        a = self.identity_to_mu_sigma[actor]
        b = self.behavior_to_mu_sigma[behavior]
        o = self.identity_to_mu_sigma[object]

        value_list = array('f', [a.E_M[0], a.P_M[0], a.A_M[0],
                                 b.E_M[0], b.P_M[0], b.A_M[0],
                                 o.E_M[0], o.P_M[0], o.A_M[0]])

        deflection_val = compute_deflection_simple(value_list, self.deflection_computation)
        def_prob = exp(- deflection_val )

        if predict_only == '':
            return def_prob
        elif predict_only == 'a':
            return def_prob* prob_w_lambda(self.count_identities[actor], self.n_docs, self.n_identities,1.)
        elif predict_only == 'b':
            return def_prob**prob_w_lambda(self.count_behaviors[behavior], self.n_docs, self.n_behaviors,1)*\
                       prob_w_lambda(self.count_behavior_given_actor[behavior][actor],
                                     self.count_behaviors[behavior], self.n_identities,1)*\
                       prob_w_lambda(self.count_behavior_given_object[behavior][object],
                                     self.count_behaviors[behavior], self.n_identities,1)
        elif predict_only == 'o':
            return def_prob* prob_w_lambda(self.count_identities[object], self.n_docs, self.n_identities,1.)
        else:
            raise Exception("WTF")


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

        ##init unknown values
        initialize_unknown_fields(self.docs,
                                  self.M, self.t,
                                  self.all_identities, self.all_behaviors,
                                  self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                                  1, 1, 1, 1, 1)

    def get_name(self):
        return tab_stringify_newline(['Simple',-1,self.eq_shuffle, "","","","",""],newline=False)


