__author__ = 'kjoseph'


import inspect, sys, os
from math import log, exp

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
sys.path.append(os.path.join(cmd_folder,"../"))


import multiprocessing
import os
from math import log

import numpy as np

from initialize_functions import get_docs
from utility import tab_stringify_newline


class BaseAnalysisMethod(multiprocessing.Process):
    def __init__(self, fold, data_dir,
                 identities_filename, behaviors_filename,
                 run_perplexity, run_prediction,
                 n_latent_topics, doc_output_fn):

        multiprocessing.Process.__init__(self)

        training_data_filename = os.path.join(data_dir, "train.tsv")

        self.fold = str(fold)

        self.docs, \
        self.all_identities, \
        self.all_behaviors = get_docs(training_data_filename,
                                      identities_filename,
                                      behaviors_filename,
                                      n_latent_topics, doc_output_fn)


        self.data_dir = data_dir

        self.run_perplexity = run_perplexity

        self.run_prediction = run_prediction

        self.test_data = []
        self.behaviors_to_test = []
        self.objects_to_test = []


    def get_name(self):
        raise Exception("not implemented")

    def predict_left_out(self, document):
        raise Exception("not implemented")

    def calculate_probability(self, actor,behavior,object, predict_only = ''):
        raise Exception("not implemented")

    def train(self):
        raise Exception("not implemented")

    def set_test_data(self, test_data):
        self.test_data = test_data


    def run_prediction_analysis(self):
        pred10_output_file_name = os.path.join(self.data_dir,
                                "pred_10_"+ str(self.fold) + "_"+ self.get_name().replace("\t","_"))
        predterm_output_file_name = os.path.join(self.data_dir,
                                "pred_term_" + str(self.fold) + "_"+self.get_name().replace("\t","_"))

        predfinal_output_file_name = os.path.join(self.data_dir,
                                                 "pred_final_" + str(self.fold)+ "_"+ self.get_name().replace("\t","_"))        
        behavior_out_pred10 = open(pred10_output_file_name + "_behavior", "w")
        behavior_out_predterm = open(predterm_output_file_name + "_behavior", "w")
        pred_final_out = open(predfinal_output_file_name,"w")

        name = self.get_name()

        beh_ranks = []

        all_behaviors_list = list(self.all_behaviors)
        b_mean_val = 0
        for i in range(len(self.test_data)):
            doc = self.test_data[i]

            if i % 50 == 0 and i != 0:
                print i, b_mean_val, self.get_name()
            behavior_scores = sorted([(t, self.calculate_probability(doc.actor,t,doc.object,'b'))
                                            for t in all_behaviors_list],
                                    key=lambda val: -val[1])
            b_sum = sum([x[1] for x in behavior_scores])
            behavior_scores = [(x[0],x[1]/b_sum) for x in behavior_scores]

            ##write top 10
            for j in range(10):
                behavior_out_pred10.write(tab_stringify_newline([name,i,behavior_scores[j][0],behavior_scores[j][1]]))

            ##write actual
            for index, term_score in enumerate(behavior_scores):
                if term_score[0] == doc.behavior:
                    #print tab_stringify_newline([name,i,term_score[0],term_score[1], index],newline=False)
                    behavior_out_predterm.write(tab_stringify_newline([name,i,term_score[0],term_score[1], index]))
                    beh_ranks.append(index)
                    try:
                        b_mean_val += log(term_score[1])
                    except:
                        print 'exception'
                        print term_score[1]
                        b_mean_val += -20
                    break

        behavior_out_pred10.close()
        behavior_out_predterm.close()

        percent_top_10_beh = len([x for x in beh_ranks if x < 10])/float(len(beh_ranks))
        behavior_results = tab_stringify_newline([self.fold,self.get_name(),
                                                    np.median(beh_ranks), np.mean(beh_ranks),
                                                    percent_top_10_beh, b_mean_val,len(beh_ranks)])
        pred_final_out.write(behavior_results)
        pred_final_out.close()

        print behavior_results#, object_results
    def run_perplexity_analysis(self):
        sum_perp = 0
        i = 0
        output_file_name = os.path.join(self.data_dir,
                                "perp_" + self.get_name().replace("\t","_"))

        outfil = open(output_file_name, "w")

        name = self.get_name()
        for document in self.test_data:
            i += 1
            prob = self.calculate_probability(document.actor, document.behavior, document.object)
            outfil.write(self.fold + "\t" + name + "\t" + str(i) + "\t" + str(prob) + "\n")
            # #if its an underflow
            if prob == 0:
                log_perp = 100
            else:
                log_perp = log(prob, 2)
            sum_perp += log_perp

        outfil.write(self.fold + "\t" + name + '\t' + "-1\t" + str(sum_perp / float(len(self.test_data)) * -2.) + "\n")
        outfil.close()

    def run(self):
        self.train()

        if self.run_perplexity:
            self.run_perplexity_analysis()

        if self.run_prediction:
            self.run_prediction_analysis()
