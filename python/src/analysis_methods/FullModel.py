

__author__ = 'kjoseph'

import inspect, sys, os

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
sys.path.append(os.path.join(cmd_folder,"../"))

from initialize_functions import *
from functions import *
from deflection import *
from utility import *
import numpy as np
from array import array
import os
from math import exp
from SimpleACT import SimpleACT
from BaseAnalysisMethod import BaseAnalysisMethod
import scipy.spatial
from collections import Counter, defaultdict
from SimpleWordModels import prob_w_lambda

# #################HYPERPARAMETERS###################

BETA = 1
INITIAL_V = 2
INITIAL_K_KNOWN = 50
INITIAL_K_UNKNOWN = 10
N_GIBBS_SAMPLES_FOR_PRED = 50


class FullModel(SimpleACT):
    def __init__(self, fold, data_dir,
                 all_identities_filename, all_behaviors_filename,
                 run_perplexity, run_prediction,
                 act_data_filename, act_equation_filename,
                 rand_eq, rand_vals,
                 n_latent_topics, n_iterations, burn_in,
                 dir_prior_topics, alpha,psi, init_sd_for_all, q,
                 use_lm_for_prediction=True):

        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.dirichlet_prior_topics = dir_prior_topics
        self.alpha = alpha
        self.psi = psi
        self.eq_shuffle = rand_eq
        self.init_rand = rand_vals
        self.n_latent_topics = n_latent_topics
        self.init_sd_for_all = init_sd_for_all
        self.already_trained = False
        self.q = q
        self.use_lm_for_prediction = use_lm_for_prediction
        self.output_directory = os.path.join(data_dir, "trained_models", self.get_name().replace("\t", "_"))

        # #Create output files##
        try:
            os.mkdir(self.output_directory)
        except OSError:
            print 'dir exists, ', self.output_directory, ' so using trained model!!!!'
            self.already_trained = True


        BaseAnalysisMethod.__init__(self, fold, data_dir,
                                    all_identities_filename,all_behaviors_filename,
                                    run_perplexity,run_prediction,self.n_latent_topics,
                                    os.path.join(self.output_directory, "output_document_basic.tsv"))

        ##get equations
        self.t, self.M = get_t_and_M_from_file(act_equation_filename, FUNDAMENTALS)
        self.eq_shuffle = rand_eq
        if rand_eq:
            np.random.shuffle(self.M)
        #get our deflection equations
        self.deflection_computation = get_coefficients(self.M, self.t)
        self.base_coeff, self.coeff_vector, self.vals_matrix, self.coefficient_indexes = \
            get_coefficient_matrices(self.deflection_computation)

        ##read in ACT data
        self.identity_to_mu_sigma, self.behavior_to_mu_sigma = read_in_act_data(act_data_filename,
                                                              INITIAL_K_KNOWN,INITIAL_V,
                                                              self.n_latent_topics,self.dirichlet_prior_topics,
                                                              self.init_sd_for_all)
        if self.init_rand:
            self.randomize(self.identity_to_mu_sigma,
                           self.init_sd_for_all,
                           n_latent_topics, dir_prior_topics,
                           INITIAL_K_KNOWN,INITIAL_V)
            self.randomize(self.behavior_to_mu_sigma,
                           self.init_sd_for_all,
                           n_latent_topics, dir_prior_topics,
                           INITIAL_K_KNOWN,INITIAL_V)

        ##init unknown values
        self.docs = np.array(self.docs)

        self.count_identities = Counter()
        self.count_behaviors = Counter()
        self.count_object_given_behavior = defaultdict(Counter)
        self.count_object_given_actor = defaultdict(Counter)
        self.count_behavior_given_actor = defaultdict(Counter)
        self.count_behavior_given_object = defaultdict(Counter)
        self.n_docs = 0
        self.n_identities = 0
        self.n_behaviors = 0


    def init_func(self, filename,max_iter,func_to_call):

        fil = open(filename)
        dat = [line.strip().split("\t") for line in fil.readlines()]
        fil.close()

        line_iter = 0
        while int(dat[line_iter][0]) < (max_iter-1):
            line_iter+=1

        while line_iter < len(dat):
            line = dat[line_iter]
            datatype = int(line[1])
            func_to_call(line,datatype,self.identity_to_mu_sigma,self.behavior_to_mu_sigma)
            line_iter+=1

    def load_model(self):
        print 'loading pretrained'
        ## all we have to set is mu0, sigma0, pi, theta, phi
        output_parameters_file = open(os.path.join(self.output_directory, "output_params.tsv"))

        max_iter = -1
        n_topics = -1
        for line in output_parameters_file:
            line_spl = line.split("\t")
            if line_spl[0] == 'n_iter':
                max_iter = int(line_spl[1])
            if line_spl[0] == 'n_topics':
                n_topics = int(line_spl[1])

        assert self.n_latent_topics == n_topics
        self.n_iterations = max_iter

        self.init_func(os.path.join(self.output_directory, "output_mu.tsv"),max_iter,init_mu_func)
        self.init_func(os.path.join(self.output_directory, "output_pi.tsv"),max_iter,init_pi_func)


    def train(self):

        initialize_unknown_fields(self.docs, self.M, self.t,
                          self.all_identities, self.all_behaviors,
                          self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                          self.init_sd_for_all, INITIAL_K_UNKNOWN, INITIAL_V,
                          self.n_latent_topics, self.dirichlet_prior_topics)

        ##train language model either way, its easy enough
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


        ##don't work w/ pretrained, for now

        #if self.already_trained:
        #    self.load_model()
        #    print 'loaded pretrained model'
        #    return

        output_parameters_file = open(os.path.join(self.output_directory, "output_params.tsv"),"w")
        for x in [['n_topics', self.n_latent_topics], ['n_iter', self.n_iterations], ['p',self.dirichlet_prior_topics],
                    ['beta', BETA], ["alpha",self.alpha], ["psi",self.psi],["s0",self.init_sd_for_all],
                    ['v0',INITIAL_V], ['init_k_unk',INITIAL_K_UNKNOWN], ['init_k_kn',INITIAL_K_KNOWN],
                    ['q',self.q]]:
            output_parameters_file.write(tab_stringify_newline(x))
        output_parameters_file.close()


        self.write_hyperparam_output()

        # ####FINISH INTIALZATION OF DOCS
        complete_initialization_of_docs(self.docs,
                                        self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                                        self.vals_matrix, self.coeff_vector, self.coefficient_indexes,
                                        self.dirichlet_prior_topics, self.n_latent_topics)

        iteration = 0

        print 'final len identities: ', len(self.identity_to_mu_sigma)
        print 'final len behaviors: ', len(self.behavior_to_mu_sigma)
        print 'len docs: ', len(self.docs)

        output_mu_filename = os.path.join(self.output_directory, "output_mu.tsv")
        output_pi = os.path.join(self.output_directory, "output_pi.tsv")
        output_doc_top = os.path.join(self.output_directory, "output_document_topic.tsv")

        mu_output_file = open(output_mu_filename, "w")
        z_output_file = open(output_pi, "w")
        document_topic_output_file = open(output_doc_top, "w")

        while iteration < self.n_iterations:
            if iteration % 20 == 0:
                print 'iteration: ', iteration

            ###E STEP######

            update_documents(iteration,
                             self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                             self.docs, BETA, self.n_latent_topics, self.coefficient_indexes, self.coeff_vector)

            if iteration >= self.burn_in:
                for d in self.docs:
                    document_topic_output_file.write(d.tostring(iteration))
            #print 'doc update finished'

            update_mu_sigma_z(0, iteration, iteration >= self.burn_in,
                              self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                              self.docs, self.n_latent_topics,
                              mu_output_file, z_output_file)
            #print 'done identities'
            update_mu_sigma_z(1, iteration, iteration >= self.burn_in,
                              self.identity_to_mu_sigma, self.behavior_to_mu_sigma,
                              self.docs, self.n_latent_topics,
                              mu_output_file, z_output_file)
            #print 'done behaviors'

            iteration += 1

        mu_output_file.close()
        z_output_file.close()
        document_topic_output_file.close()


    def write_hyperparam_output(self):
        hyper_out = os.path.join(self.output_directory, "output_hyper.tsv")
        hyper_outputfile = open(hyper_out, "w")
        for k, v in self.identity_to_mu_sigma.items():
            hyper_outputfile.write(v.to_string(k, "identity") + "\n")
        for k, v in self.behavior_to_mu_sigma.items():
            hyper_outputfile.write(v.to_string(k, "behavior") + "\n")
        hyper_outputfile.close()


    def get_mu_sample(self, entity):
        z = np.random.choice(self.n_latent_topics, size=None, p=entity.PI_MULTINOMIAL)
        return [np.random.normal(entity.E_MU_0[z], entity.E_SIGMA_0[z]),
                np.random.normal(entity.P_MU_0[z], entity.P_SIGMA_0[z]),
                np.random.normal(entity.A_MU_0[z], entity.A_SIGMA_0[z])]


    def calculate_probability(self, actor,behavior, object, predict_only=''):
        a = self.identity_to_mu_sigma[actor]
        b = self.behavior_to_mu_sigma[behavior]
        o = self.identity_to_mu_sigma[object]

        sum_prob_samples = 0
        for i in range(N_GIBBS_SAMPLES_FOR_PRED):
            ##sample from the Zs; sample from the mus;
            actor_mu = self.get_mu_sample(a)
            behavior_mu = self.get_mu_sample(b)
            object_mu = self.get_mu_sample(o)
            value_list = array('f', actor_mu + behavior_mu + object_mu)
            defl = compute_deflection_simple(value_list, self.deflection_computation)
            sum_prob_samples += exp(-defl)

        rv = sum_prob_samples / float(N_GIBBS_SAMPLES_FOR_PRED)

        if not self.use_lm_for_prediction:
            return rv

        if predict_only == 'b':
            #print "\t".join([str(x)for x in [behavior, self.phi[behavior], rv]])
            return rv*prob_w_lambda(self.count_behaviors[behavior], self.n_docs, self.n_behaviors,self.psi)*\
                       prob_w_lambda(self.count_behavior_given_actor[behavior][actor],
                                     self.count_behaviors[behavior], self.n_identities,self.q)*\
                       prob_w_lambda(self.count_behavior_given_object[behavior][object],
                                     self.count_behaviors[behavior], self.n_identities,self.q)
        raise Exception("wtf")

    def get_name(self):
        return tab_stringify_newline([
            'Full',
            self.n_latent_topics,
            self.eq_shuffle,
            self.dirichlet_prior_topics,
            self.alpha,
            self.init_sd_for_all,self.q,
            self.use_lm_for_prediction
        ],newline=False)





def init_mu_func(line,datatype,identity_to_mu_sigma,behavior_to_mu_sigma):
    if datatype == 0:
        identity_to_mu_sigma[line[3]].set_mu0_python(int(line[4]),int(line[5]),float(line[8]))
        identity_to_mu_sigma[line[3]].set_sigma0_python(int(line[4]),int(line[5]),float(line[9]))
    else:
        behavior_to_mu_sigma[line[3]].set_mu0_python(int(line[4]),int(line[5]),float(line[8]))
        behavior_to_mu_sigma[line[3]].set_sigma0_python(int(line[4]),int(line[5]),float(line[9]))

def init_pi_func(line,datatype, identity_to_mu_sigma,behavior_to_mu_sigma):
    if datatype == 0:
        identity_to_mu_sigma[line[3]].set_pi_python([float(f) for f in line[4:]])
    else:
        behavior_to_mu_sigma[line[3]].set_pi_python([float(f) for f in line[4:]])




