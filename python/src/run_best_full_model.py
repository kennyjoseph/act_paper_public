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

full_dataset_location =  os.path.join(top_dir,"data","full_dataset")
output_dir = os.path.join(full_dataset_location,"trained_models")
try:
    os.mkdir(output_dir)
except:
    pass

#bm = BigramModel(-1, full_dataset_location, IDENTITIES_FILENAME, BEHAVIORS_FILENAME, False,True)

fm = FullModel(-1,full_dataset_location,
               IDENTITIES_FILENAME, BEHAVIORS_FILENAME, False,False,
               ACT_DATA_FILENAME, ACT_EQUATIONS_FILENAME,
               False,False,
               5, N_ITERATIONS, BURN_IN, 3, 1,1,.1,1)

#test_data, all_identities, all_behaviors = get_docs(os.path.join(full_dataset_location, "test.tsv"),IDENTITIES_FILENAME,BEHAVIORS_FILENAME, 0, None)
#bm.set_test_data(test_data)
#bm.start()
#fm.set_test_data(test_data)
fm.start()