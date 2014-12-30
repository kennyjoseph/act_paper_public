#!python
#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: profile=True
#cython: cdivision=True

import codecs, random, sys
from math import gamma
from collections import defaultdict
import numpy as np
from libcpp.vector cimport vector
from cpython cimport array
cimport numpy as np
import scipy
from utility import *
from libcpp cimport bool as bool_t
cimport cython
from deflection import *
from libc.math cimport log, exp, pow, sqrt
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.stdlib cimport abs as c_abs
from scipy.stats import invgamma, norm
N_FUNDAMENTALS =9
EPA_LIST = ['e','p','a']

cpdef set_srand(int seed):
    srand(seed)



@cython.boundscheck(False)
@cython.cdivision(True)
cpdef update_mu_sigma_z(int datatype, int iteration,
                        bool_t write_output,
                          dict identity_to_mu_sigma,
                          dict behavior_to_mu_sigma,
                          Document[:]& docs,
                          int n_latent_topics,
                          mu_output_file,
                          z_output_file):

    cdef int epa_offset, epa_offset_iter, array_size, tmp_iter, doc_index_itr, term_iter, shuff_len, new_values_iter,new_values_count

    cdef float new_values_total, new_values_count_float,  new_values_mean, new_values_tmp
    cdef float v0, k0, s_sq0, SSD, m0, vN,kN,mN,vN_times_s_sqN,new_var,\
        new_scale_param, k_over_s0, one_over_s0, new_sd_0, new_mu_0, alpha, beta, mu_sum=0, m0_sum=0

    cdef ParameterStruct old_values
    cdef vector[double] new_values
    cdef Document doc

    ###set data type
    if datatype == 0:
        shuff = identity_to_mu_sigma.keys()
    else:
        shuff = behavior_to_mu_sigma.keys()

    shuff_len = len(shuff)

    ##for each term, we need to update all mu_0s, sigma_0s and Z
    for term_iter in range(shuff_len):
        term = shuff[term_iter]
        ##get the old values (soon to be the new ones)
        if datatype == 0:
            old_values = identity_to_mu_sigma[term]
        else:
            old_values = behavior_to_mu_sigma[term]


        ##for each EPA score
        for epa_offset in [0,1,2]:

            new_values.clear()

            ##for each topic/latent sense
            for topic_index in range(n_latent_topics):
                ###########GET "SAMPLES"###############
                #### Get the mean of the new "samples", really the mus from documents where this actor topic is used
                new_values_total = 0
                new_values_count = 0
                array_size =  old_values.DOC_IDS.size()

                for doc_index_itr in range(array_size):
                    tmp_iter = old_values.DOC_IDS[doc_index_itr]
                    doc = docs[tmp_iter]

                    index_to_update = old_values.DOC_INDEX_STARTS[doc_index_itr] + epa_offset

                    if doc.is_right_topic(index_to_update,topic_index):
                        new_values_total += doc.value_list[index_to_update]
                        new_values_count += 1
                        new_values.push_back(doc.value_list[index_to_update])


                if new_values_count == 0:
                    if write_output:
                        mu_output_file.write(tab_stringify_newline([
                            iteration,datatype,old_values.FROM_DATA,term,epa_offset,topic_index,
                            new_values_count, "", old_values.get_mu0(epa_offset,topic_index), old_values.get_sigma0(epa_offset,topic_index)]))
                    continue

                new_values_count_float = float(new_values_count)
                new_values_mean = new_values_total / new_values_count_float
                SSD = 0

                for new_values_iter in range(new_values_count):
                    new_values_tmp = new_values[new_values_iter]
                    SSD += (new_values_tmp - new_values_mean)**2

                ###########GET PRIOR PARAMS###############
                v0 = old_values.get_v(topic_index)
                k0 = old_values.get_k(topic_index)
                s_sq0 = old_values.get_s(epa_offset, topic_index)
                m0 = old_values.get_m(epa_offset, topic_index)

                ###########UPDATE SIGMA_0###############
                ###w/ help from Gelman BDA 3, pg. 68 and
                ##http://engineering.richrelevance.com/bayesian-analysis-of-normal-distributions-with-python/

                # combining the prior with the data
                # to make sense of this note that
                # inv-chi-sq(v,s^2) = inv-gamma(v/2,(v*s^2)/2)
                kN = float(k0 + new_values_count_float)
                mN = (k0/kN)*m0 + (new_values_count_float/kN)*new_values_mean
                vN = v0 + new_values_count_float
                vN_times_s_sqN = v0*s_sq0 + SSD + (new_values_count_float*k0*(m0-new_values_mean)**2)/kN

                # 1) Get the new MAP of sigma_0
                if vN > 2:
                    new_var = vN_times_s_sqN/vN * (vN/(vN-2))
                else:
                    new_var = 10
                new_sd_0 = sqrt(new_var)

                # 2) Get the new MAP of mu_0
                new_mu_0 = mN

                if topic_index == 0:
                    m0_sum  += abs(m0)
                    mu_sum += abs(new_mu_0)

                #if epa_offset == 0:
                #    print "\t", tab_stringify_newline([term, topic_index,
                #                                           "m0: ", m0,
                #                                           "nvm: ", new_values_mean,
                #                                           "sd: ", new_sd_0,
                #                                           "mean: ", new_mu_0,
                #                                           "count: ", new_values_count],newline=False)
                ###########REPLACE VALUES AND OUTPUT STEP###############
                old_values.set_mu0(epa_offset,topic_index,new_mu_0)
                old_values.set_sigma0(epa_offset,topic_index,new_sd_0)

                if write_output:
                    mu_output_file.write(tab_stringify_newline([
                        iteration,datatype,old_values.FROM_DATA,term,epa_offset,topic_index,
                        new_values_count, new_values_mean, new_mu_0, new_sd_0]))

        ###########NOW WE CAN UPDATE Z##############
        old_values.map_estimate_z()
        if write_output:
            z_output_file.write(tab_stringify_newline([iteration,datatype,old_values.FROM_DATA,term]+ old_values.PI_MULTINOMIAL))
    #print 'MU SUM: ', mu_sum, ' M0 SUM: ', m0_sum

cpdef update_documents(int iteration,
                       dict identity_to_mu_sigma,
                       dict behavior_to_mu_sigma,
                       Document[:]& docs,
                       double beta,
                       int n_latent_topics,
                       vector[vector[int]]& coefficient_indexes,
                       double[:]& coefficient_vector):

    cdef int doc_size, new_topic, doc_iter, tmp
    cdef Document doc
    cdef ParameterStruct a,b,o

    doc_size = len(docs)
    for doc_iter in range(doc_size):
        doc = docs[doc_iter]
        a = identity_to_mu_sigma[doc.actor]
        b = behavior_to_mu_sigma[doc.behavior]
        o = identity_to_mu_sigma[doc.object]

        #print 'sampling: ', a,b,o
        #print 'before doc topics: ', doc.abo_topics

        ##sample new topic
        #print '******************SAMPLING ACTOR: ', doc.actor
        a.increase_topic_count(doc.sample_topic(0,a),doc.count)
        #print '******************SAMPLING BEHAVIOR: ', doc.behavior
        b.increase_topic_count(doc.sample_topic(1,b),doc.count)
        #print '******************SAMPLING OBJECT: ', doc.object
        o.increase_topic_count(doc.sample_topic(2,o),doc.count)

        #print 'after doc topics: ', doc.abo_topics


        doc.update_all_mu(a,b,o,coefficient_vector,coefficient_indexes,beta)


#########################
########################
########################

cdef class Document:

    cdef public str actor
    cdef public str behavior
    cdef public str object
    cdef public int count

    cdef public vector[int] abo_topics
    cdef public vector[double] value_list

    cdef double[:,::1] value_array

    cdef int n_latent_topics

    cdef public int doc_id

    def __init__(self,doc_id, actor,behavior,object, count, n_latent_topics):
        self.doc_id = doc_id
        self.actor = actor
        self.behavior = behavior
        self.object = object
        self.count = count
        if n_latent_topics > 0:
            self.value_array = np.empty((333,9))
            self.n_latent_topics = n_latent_topics
            for i in range(3):
                self.abo_topics.push_back(-1)

    def initialize_topic(self, abo_index, multinomial):
        cdef int new_topic
        new_topic =  np.random.choice(self.n_latent_topics,p=multinomial)
        self.abo_topics[abo_index] = new_topic

        return int(new_topic)
        #if self.behavior == 'aid':
        #    self.abo_topics[abo_index]= 0
        #    return 0
        #else:
        #    self.abo_topics[abo_index] = 1
        #    return 1

    def initialize_mu_values(self,
                             ParameterStruct actor_params,
                             ParameterStruct behavior_params,
                             ParameterStruct object_params,
                             np.ndarray[dtype=np.double_t,ndim=2] vals_cop,
                             coefficient_vector,
                             const vector[vector[int]]& coefficient_indexes):
        cdef int i

        for i in range(3):
            self.value_list.push_back(actor_params.get_mu0(i,0))
        for i in range(3):
            self.value_list.push_back(behavior_params.get_mu0(i,0))
        for i in range(3):
            self.value_list.push_back(object_params.get_mu0(i,0))

        self.set_value_array(self.value_list,vals_cop,coefficient_indexes)

        shuff = [i for i in range(9)]
        random.shuffle(shuff)
        for index_to_update in shuff :
            c1 = self.get_c(index_to_update,coefficient_vector,coefficient_indexes[index_to_update])
            c0 = self.get_c(index_to_update+9, coefficient_vector,coefficient_indexes[index_to_update+9])
            mean = -c1/(2*c0)
            self.value_list[index_to_update] = mean
            self.update_value_array(index_to_update,
                                    coefficient_indexes[index_to_update],
                                    coefficient_indexes[index_to_update+9],
                                    self.value_list[index_to_update])

        self.set_value_array(self.value_list,vals_cop,coefficient_indexes)

    cdef int sample_topic(self, int abo_index, ParameterStruct curr_values):
        cdef vector[double] e_values,p_values,a_values, full_topic_values
        cdef double e_sum=0.000000000001, p_sum=0.000000000001, a_sum=0.000000000001, cumsum =0, e_val,p_val,a_val
        cdef int i
        cdef float rand_val

        ##get the log likelihood of each one
        for i in range(self.n_latent_topics):
            e_val = prob_normal(curr_values.get_mu0(0,i),
                                                curr_values.get_sigma0(0,i),
                                                self.value_list[abo_index*3])
            p_val = prob_normal(curr_values.get_mu0(1,i),
                                                curr_values.get_sigma0(1,i),
                                                self.value_list[abo_index*3+1])
            a_val = prob_normal(curr_values.get_mu0(2,i),
                                                curr_values.get_sigma0(2,i),
                                                self.value_list[abo_index*3+2])
            #print tab_stringify_newline(['pi[i]: ', curr_values.PI_MULTINOMIAL[i],
            #                                 'e likelihood: ',e_val,
            #                                 'curr mu: ', curr_values.get_mu0(0,i),
            #                                 'curr val: ', self.value_list[abo_index*3],
            #                                 'sd: ', curr_values.get_sigma0(0,i)])

            e_values.push_back(e_val)
            p_values.push_back(p_val)
            a_values.push_back(a_val)
            e_sum += e_val
            p_sum += p_val
            a_sum += a_val

        ##normalize to a probability distribution
        for i in range(self.n_latent_topics):
            #print tab_stringify_newline([
            #     self.behavior,
             #       'topic: ', i,
             #       'likelihood: ', e_values[i]/e_sum, p_values[i]/p_sum, a_values[i]/a_sum ,curr_values.PI_MULTINOMIAL[i]])
            cumsum += e_values[i]/e_sum * p_values[i]/p_sum * a_values[i]/a_sum * curr_values.PI_MULTINOMIAL[i]
            full_topic_values.push_back(cumsum)


        ##pick one
        rand_val = (float(rand()) / float(RAND_MAX))* cumsum
        #print 'RAND VAL: ', rand_val, " CUMSUM: ", cumsum
        for i in range(self.n_latent_topics):
            #if abo_index == 0:
                #print 'I: ', i , " top_val_here: ", full_topic_values[i]
            if rand_val <= full_topic_values[i]:
                #if abo_index == 0:
                #    print '\tpicked: ', i
                self.abo_topics[abo_index] = i
                return i

        ##everything is very, very random here, so just set it to the first topic
        self.abo_topics[abo_index] = 0
        return 0

    cdef double draw_new_mu(self, int index_to_update, int topic_index,
                                ParameterStruct params,
                               double mean_data, double var_data):# except -1000:
        cdef double var_prior, mean_prior, new_mu
        cdef int epa_index

        epa_index = index_to_update%3

        mean_prior = params.get_mu0(epa_index,topic_index)
        var_prior = params.get_sigma0(epa_index,topic_index)**2

        #if sqrt(var_data*var_prior/(var_data+var_prior)) <= 0:
        #    print '\tparams: ', params.term , params.get_m(epa_index,topic_index)
        #    print '\tEPA, TOPIC: ', epa_index, topic_index
        #    print '\tMEAN, VAR PRIOR: ', mean_prior, var_prior, " SIGMA: ", params.get_sigma0(epa_index,topic_index)
        #    print '\tMEAN, VAR DATA: ', mean_data, var_data
        #    print '\tDRAWING FROM: ', (mean_prior*var_data+ mean_data*var_prior)/(var_data+var_prior),sqrt(var_data*var_prior/(var_data+var_prior))
        #    raise Exception("ASDFAS")
        #    return -1000

        new_mu = np.random.normal(
            (mean_prior*var_data+ mean_data*var_prior)/(var_data+var_prior),
            sqrt(var_data*var_prior/(var_data+var_prior)))

        self.value_list[index_to_update] = new_mu

    cdef update_all_mu(self,
                         ParameterStruct actor_parameters,
                         ParameterStruct behavior_parameters,
                         ParameterStruct object_parameters,
                         double[:]& coeff_vector,
                         vector[vector[int]]& coefficient_indexes,
                         double beta):

        cdef double c0,c1,mean,variance,new_mean_to_draw_from,new_sd_to_draw_from, sum_sds
        cdef int index_to_updatef
        for index_to_update in range(9):
            c1 = self.get_c(index_to_update,coeff_vector,coefficient_indexes[index_to_update])
            c0 = self.get_c(index_to_update+9, coeff_vector,coefficient_indexes[index_to_update+9])
            mean = -c1/(2*c0)
            variance = beta/(2*abs(c0))

            if index_to_update <3:
                self.draw_new_mu(index_to_update, self.abo_topics[0], actor_parameters, mean, variance)
            elif index_to_update < 6:
                #print '\tbehavior'
                self.draw_new_mu(index_to_update, self.abo_topics[1], behavior_parameters,mean,variance)
            else:
                #print '\tobject'
                self.draw_new_mu(index_to_update, self.abo_topics[2], object_parameters,mean,variance)

            self.update_value_array(index_to_update,
                                    coefficient_indexes[index_to_update],
                                    coefficient_indexes[index_to_update+9],
                                    self.value_list[index_to_update])

    cdef set_value_array(self,
                         const vector[double]& value_list,
                         double[:,::1]& vals_cop,
                         const vector[vector[int]]& coefficient_indexes):
        cdef int ind,v, tmp, ind_plus_9
        cdef double val_1, val_2

        self.value_array = vals_cop
        for ind in range(9):
            val_1 = value_list[ind]
            val_2 = val_1*val_1
            ind_plus_9 = ind+9

            for v in range(coefficient_indexes[ind].size()):
                tmp = coefficient_indexes[ind][v]
                self.value_array[tmp,ind] = val_1

            for v in range(coefficient_indexes[ind_plus_9].size()):
                tmp = coefficient_indexes[ind_plus_9][v]
                self.value_array[tmp,ind_plus_9] = val_2

    cdef float get_c(self, int index_to_get_c_for,
                    const double[:]& coeff_vector,
                    const vector[int]& coefficient_indexes):
        cdef int index, ind_inner, tmp
        cdef float c_tmp, c = 0, orig

        for index in range(coefficient_indexes.size()):
            tmp = coefficient_indexes[index]
            c_tmp = 1
            orig = self.value_array[tmp ,index_to_get_c_for]
            self.value_array[tmp ,index_to_get_c_for] = 1
            for ind_inner in range(18):
                c_tmp *= self.value_array[tmp,ind_inner]
            c += coeff_vector[tmp]*c_tmp
            self.value_array[tmp ,index_to_get_c_for] = orig
        return c



    cdef int is_right_topic(self, int index_starts, int curr_topic_index):
        if index_starts < 3:
            return curr_topic_index == self.abo_topics[0]
        elif index_starts < 6:
            return curr_topic_index == self.abo_topics[1]
        else:
            return curr_topic_index == self.abo_topics[2]

    cdef update_value_array(self,
                            int index_to_update,
                            vector[int]& coefficient_indexes_single,
                            vector[int]& coefficient_indexes_squared,
                            float new_value):
        cdef float new_val_squared = (new_value*new_value)
        for ind in coefficient_indexes_single:
            self.value_array[ind,index_to_update] = new_value
        for ind2 in coefficient_indexes_squared:
            self.value_array[ind2,index_to_update+9] = new_val_squared

    def return_numpy_array(self):
        return np.asarray(self.value_array)


    def tostring(self, iteration):
        return tab_stringify_newline([
            iteration,
            self.doc_id,
            tab_stringify_newline(self.abo_topics,False),
            tab_stringify_newline(self.value_list,False)
        ])











cdef class ParameterStruct:

    cdef vector[double] K
    cdef vector[double] V
    ###EPA VALUES
    cdef public vector[double] E_M
    cdef public vector[double] E_S
    cdef public vector[double] E_MU_0
    cdef public vector[double] E_SIGMA_0

    cdef public vector[double] P_M
    cdef public vector[double] P_S
    cdef public vector[double] P_MU_0
    cdef public vector[double] P_SIGMA_0

    cdef public vector[double] A_M
    cdef public vector[double] A_S
    cdef public vector[double] A_MU_0
    cdef public vector[double] A_SIGMA_0

    ### FOR Z####
    cdef public vector[int] TOPIC_COUNTS
    cdef public double TOPIC_COUNT_SUM
    cdef public vector[double] PI_MULTINOMIAL
    cdef int DIRICHLET_PRIOR

    ###DATA
    cdef public int FROM_DATA
    cdef public vector[int] DOC_IDS
    cdef public vector[int] DOC_INDEX_STARTS

    cdef public str term

    cdef int N_LATENT_TOPICS

    def __init__(self,term, from_data, k_val, v_val,
                 e_values, p_values, a_values,
                 n_latent_topics, dirichlet_prior_topics):

        self.term = term

        self.K = [k_val if i == 0  else 1 for i in range(n_latent_topics)]
        self.V = [v_val for i in range(n_latent_topics)]

        self.E_M = [e_values[0] if i == 0 else np.random.uniform(-4.3,4.3) for i in range(n_latent_topics)]
        self.E_S =  [e_values[1] for i in range(n_latent_topics)]
        self.E_SIGMA_0 = [sqrt(draw_from_inverse_chi_square(self.V[i],self.E_S[i])) for i in range(n_latent_topics)]
        self.E_MU_0 = [np.random.normal(self.E_M[i],self.E_SIGMA_0[i]/sqrt(self.K[i])) for i in range(n_latent_topics)]

        self.P_M = [p_values[0] if i == 0 else np.random.uniform(-4.3,4.3) for i in range(n_latent_topics)]
        self.P_S = [p_values[1] for i in range(n_latent_topics)]
        self.P_SIGMA_0 = [sqrt(draw_from_inverse_chi_square(self.V[i],self.P_S[i]))  for i in range(n_latent_topics)]
        self.P_MU_0 = [np.random.normal(self.P_M[i],self.P_SIGMA_0[i]/self.K[i]) for i in range(n_latent_topics)]


        self.A_M = [a_values[0] if i == 0 else np.random.uniform(-4.3,4.3) for i in range(n_latent_topics)]
        self.A_S = [a_values[1] for i in range(n_latent_topics)]
        self.A_SIGMA_0 = [sqrt(draw_from_inverse_chi_square(self.V[i],self.A_S[i]))  for i in range(n_latent_topics)]
        self.A_MU_0 = [np.random.normal(self.A_M[i],self.A_SIGMA_0[i]/self.K[i]) for i in range(n_latent_topics)]
        self.FROM_DATA = from_data

        self.DIRICHLET_PRIOR = dirichlet_prior_topics
        self.N_LATENT_TOPICS = n_latent_topics
        self.TOPIC_COUNT_SUM = 0

        self.reinit_topics()

        self.PI_MULTINOMIAL = np.random.dirichlet(self.TOPIC_COUNTS)

    def add_doc(self, doc_id, index_start):
        self.DOC_IDS.push_back(doc_id)
        self.DOC_INDEX_STARTS.push_back(index_start)

    cdef reinit_topics(self):
        #cdef int n
        #n = self.DIRICHLET_PRIOR
        self.TOPIC_COUNTS.clear()
        #self.TOPIC_COUNTS.push_back(n)
        for i in range(self.N_LATENT_TOPICS):
            self.TOPIC_COUNTS.push_back(self.DIRICHLET_PRIOR)
        self.TOPIC_COUNT_SUM = self.DIRICHLET_PRIOR*self.N_LATENT_TOPICS#-1)#+n

    cpdef increase_topic_count(self, int topic_index, int amount):
        self.TOPIC_COUNTS[topic_index]+=  amount
        self.TOPIC_COUNT_SUM += amount
        #print 'topic counts: ', term, topic_index, self.TOPIC_COUNTS

    ########SETTERS###############
    cdef map_estimate_z(self):
        cdef int i

        ##set new value for pi

        self.PI_MULTINOMIAL = [self.TOPIC_COUNTS[i]/float(self.TOPIC_COUNT_SUM) for i in range(self.N_LATENT_TOPICS)]

        #reset topic counts for next iteration
        self.reinit_topics()

        #print 'topic counts: ', self.TOPIC_COUNTS
    cdef set_mu0(self, int epa, int topic, float value):
        if epa == 0:
            self.E_MU_0[topic] = value
        if epa == 1:
            self.P_MU_0[topic] = value
        if epa == 2:
            self.A_MU_0[topic] = value

    cdef set_sigma0(self,  int epa, int topic, float value):
        if epa == 0:
            self.E_SIGMA_0[topic] = value
        if epa == 1:
            self.P_SIGMA_0[topic] = value
        if epa == 2:
            self.A_SIGMA_0[topic] = value

    def set_mu0_python(self, int epa, int topic, float value):
        if epa == 0:
            self.E_MU_0[topic] = value
        if epa == 1:
            self.P_MU_0[topic] = value
        if epa == 2:
            self.A_MU_0[topic] = value

    def set_sigma0_python(self,  int epa, int topic, float value):
        if epa == 0:
            self.E_SIGMA_0[topic] = value
        if epa == 1:
            self.P_SIGMA_0[topic] = value
        if epa == 2:
            self.A_SIGMA_0[topic] = value

    def set_pi_python(self, pi):
        for ind, val in enumerate(pi):
            self.PI_MULTINOMIAL[ind] = val

    ###########GETTERS#############

    cdef float get_v(self, int topic):
        return self.V[topic]

    cdef float get_k(self, int topic):
        return self.K[topic]

    cdef float get_m(self, int index, int topic):
        if index == 0:
            return self.E_M[topic]
        if index == 1:
            return self.P_M[topic]
        if index == 2:
            return self.A_M[topic]

    cdef float get_s(self, int index, int topic):
        if index == 0:
            return self.E_S[topic]
        if index == 1:
            return self.P_S[topic]
        if index == 2:
            return self.A_S[topic]

    cdef float get_mu0(self, int epa, int topic):
        if epa == 0:
            return self.E_MU_0[topic]
        if epa == 1:
            return self.P_MU_0[topic]
        if epa == 2:
            return self.A_MU_0[topic]

    cdef float get_sigma0(self,  int epa, int topic):
        if epa == 0:
            return self.E_SIGMA_0[topic]
        if epa == 1:
            return self.P_SIGMA_0[topic]
        if epa == 2:
            return self.A_SIGMA_0[topic]


    def to_string(self, term, type):
        return "\t".join(
            [term,type,
             tab_stringify_newline(self.E_M,newline=False),
             tab_stringify_newline(self.E_S,newline=False),
             tab_stringify_newline(self.P_M,newline=False),
             tab_stringify_newline(self.P_S,newline=False),
             tab_stringify_newline(self.A_M,newline=False),
             tab_stringify_newline(self.A_S,newline=False),
             tab_stringify_newline(self.K,newline=False),
             tab_stringify_newline(self.V,newline=False)])





def draw_write_theta_phi(iteration, datatype, write_output, counts, output_file):
    items = counts.items()
    new_dist = np.random.dirichlet([v[1] for v in items], 1)[0]

    result_dict = {items[i][0]:new_dist[i] for i in range(len(new_dist))}

    if write_output:
        output_str = "\n".join(["{0}\t{1}\t{2}\t{3:0.6f}".format(iteration,datatype,k,v) for k,v in result_dict.items()])
        output_file.write(output_str + "\n")
    return result_dict


cpdef double prob_inverse_chi_square(double dof,double s,double x):
    return (pow(s*dof/2.,dof/2.))/gamma(dof/2.) * exp(-s*dof/(2.*x))/pow(x,1+dof/2.)

cdef double prob_normal(double mu,double sigma,double x):
    return (1./(sigma*sqrt(2*3.1415926535897)))*exp(- pow(x-mu,2)/(2.*pow(sigma,2)))
