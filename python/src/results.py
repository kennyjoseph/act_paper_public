__author__ = 'kjoseph'

import os
import glob
import random
from multiprocessing.managers import SyncManager

import numpy as np

from analysis_methods import FullModel, UnigramModel, BigramModel, SimpleACT

from initialize_functions import get_docs
from functions import set_srand
import utility

##SET THIS TO THE FULL PATH OF WHEREEVER YOU PUT THIS
top_dir = "/Users/kjoseph/git/thesis/act_paper/python/"
#top_dir = "/usr1/kjoseph/act_paper/python/"

IDENTITIES_FILENAME = os.path.join(top_dir, "data/final_identities.txt")
BEHAVIORS_FILENAME = os.path.join(top_dir, "data/final_behaviors.txt")
ACT_DATA_FILENAME = os.path.join(top_dir, "data/act_init_scores.tsv")
ACT_EQUATIONS_FILENAME = os.path.join(top_dir, "data/sexes_avg_new.txt")

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
set_srand(RANDOM_SEED)

N_ITERATIONS = 200
BURN_IN = 197

fold = 0
for data_dir in glob.glob(os.path.join(top_dir, "data/k_fold_tmp_test/fold_*/")):

    print 'data dir: ', data_dir

    output_dir = os.path.join(data_dir, "trained_models")
    try:
        os.mkdir(output_dir)
    except:
        pass

    analysis_methods = []


    # compare to simplest possible
    analysis_methods.append(UnigramModel(fold,data_dir, IDENTITIES_FILENAME, BEHAVIORS_FILENAME, 1, False,True))
    analysis_methods.append(BigramModel(fold,data_dir, IDENTITIES_FILENAME, BEHAVIORS_FILENAME, False,True))


    # do we improve on the few events that are pure ACT by adding in ACT?
    for rand_eq, rand_val in [[False, False]]:#, [True,True]]:
        analysis_methods.append(SimpleACT(fold,data_dir,
                                          IDENTITIES_FILENAME, BEHAVIORS_FILENAME, True,True,
                                          ACT_DATA_FILENAME, ACT_EQUATIONS_FILENAME,
                                          rand_eq, rand_val))

    for i in [0,1,2,4,6]:
        for pi_prior in [3,50, 1000]:
            for alpha_psi in [1]:
                for rand_eq, rand_val in [[False,False], [True,True] ]:
                    for init_sd in [.1]:
                        fm = FullModel(fold,data_dir,
                                       IDENTITIES_FILENAME, BEHAVIORS_FILENAME, False,True,
                                       ACT_DATA_FILENAME, ACT_EQUATIONS_FILENAME,
                                       rand_eq, rand_val,
                                       i + 1, N_ITERATIONS, BURN_IN, pi_prior, alpha_psi,alpha_psi,init_sd,1)
                        analysis_methods.append(fm)

    manager = SyncManager()
    manager.start(utility.mgr_init)

    test_data, all_identities, all_behaviors = get_docs(os.path.join(data_dir, "test.tsv"),IDENTITIES_FILENAME,BEHAVIORS_FILENAME, 0, None)

    for method in analysis_methods:
        method.set_test_data(test_data)
        method.start()

    try:
        for method in analysis_methods:
            method.join()
    except KeyboardInterrupt:
        print 'keyboard interrupt'

    fold +=1
