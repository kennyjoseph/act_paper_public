from collections import defaultdict
import random, codecs

from functions import *
from deflection import *

EPA_LIST = ['e','p','a']
FUNDAMENTALS = ['ae','ap','aa','be','bp','ba','oe','op','oa']


def get_docs(input_file_name, identities_filename, behaviors_filename, n_latent_topics, document_output_filename):
    docs = []
    doc_ids = 0
    data_file = codecs.open(input_file_name)

    print 'data file: ', data_file
    for line in data_file:
        line_spl = line.strip().split("\t")
        actor = line_spl[0]
        behavior = line_spl[1]
        object = line_spl[2]
        count = int(line_spl[3])
        docs += [Document(doc_ids+i, actor,behavior,object,1,n_latent_topics) for i in range(count)]
        doc_ids += count
    data_file.close()

    if document_output_filename is not None:
        document_output_file = codecs.open(document_output_filename,"w","utf8")
        for doc in docs:
            document_output_file.write(tab_stringify_newline([doc.doc_id, doc.actor,doc.behavior,doc.object]))
        document_output_file.close()

    all_identities = set([line.strip() for line in open(identities_filename).readlines()])

    all_behaviors = set([line.strip() for line in open(behaviors_filename).readlines()])

    return docs, all_identities, all_behaviors


def read_in_act_data(act_filename, init_k, init_v, n_latent_sense, dirichlet_prior, init_sd):
    identity_to_mu_sigma = {}
    behavior_to_mu_sigma = {}
    mean_sd_file = codecs.open(act_filename)
    for line in mean_sd_file:
        line_spl = line.strip().split("\t")
        dat = ParameterStruct(line_spl[0].lower(),1,
                              init_k, init_v,
                              [float(line_spl[2]), init_sd],
                              [float(line_spl[4]), init_sd],
                              [float(line_spl[6]), init_sd],
                              n_latent_sense,dirichlet_prior)

        if line_spl[1] == 'identity':
            identity_to_mu_sigma[line_spl[0].lower()] = dat
        elif line_spl[1] == 'behavior':
            behavior_to_mu_sigma[line_spl[0].lower()] = dat
    mean_sd_file.close()
    return identity_to_mu_sigma, behavior_to_mu_sigma



def initialize_unknown_fields(docs, M,t,
                              all_identities, all_behaviors,
                              identity_data, behavior_data,
                              init_sd, init_k_unknown, init_v_unknown, n_latent_topics, dirichlet_prior_topics):

    unknown_identities = set(all_identities) - set(identity_data.keys())
    unknown_behaviors = set(all_behaviors) - set(behavior_data.keys())

    ##copy docs so we can remove from it
    docs = [d for d in docs]

    ##deal with terms we don't have any data for ... just give them random for now.
    identities_not_appearing = set([s for s in unknown_identities])
    behaviors_not_appearing = set([s for s in unknown_behaviors])
    for d in docs:
        if d.actor in identities_not_appearing:
            identities_not_appearing.remove(d.actor)
        if d.object in identities_not_appearing:
            identities_not_appearing.remove(d.object)
        if d.behavior in behaviors_not_appearing:
            behaviors_not_appearing.remove(d.behavior)


    print 'N identities not appearing: ', len(identities_not_appearing)


    for identity in identities_not_appearing:
        rand_v = np.random.uniform(-4.3,4.3,3)
        identity_data[identity] = ParameterStruct(identity, 0, init_k_unknown,init_v_unknown,
                                    [rand_v[0], init_sd],[rand_v[1], init_sd],[rand_v[2], init_sd],
                                    n_latent_topics, dirichlet_prior_topics)
        unknown_identities.remove(identity)

    print 'N behaviors not appearing: ', len(behaviors_not_appearing)
    for behavior in behaviors_not_appearing:
        rand_v = np.random.uniform(-4.3,4.3,3)
        behavior_data[behavior] = ParameterStruct(behavior, 0, init_k_unknown,init_v_unknown,
                                    [rand_v[0], init_sd],[rand_v[1], init_sd],[rand_v[2], init_sd],
                                    n_latent_topics, dirichlet_prior_topics)
        unknown_behaviors.remove(behavior)


    ##real initialization
    H = np.dot(np.vstack((np.identity(9),-M)), np.hstack((np.identity(9),-M.transpose())))
    S_a,g_a = compute_s_and_g(M,t,'a',EPA_LIST)
    S_b,g_b = compute_s_and_g(M,t,'b',EPA_LIST)
    S_o,g_o = compute_s_and_g(M,t,'o',EPA_LIST)

    while (len(unknown_identities) + len(unknown_behaviors)) > 0:
        docs_to_rem = []
        identities_to_add = defaultdict(list)
        behaviors_to_add = defaultdict(list)


        np.random.shuffle(docs)
        for doc_id in range(len(docs)):
            ##get all we can fill in
            doc = docs[doc_id]
            known_actor = doc.actor in identity_data
            known_behavior = doc.behavior in behavior_data
            known_object = doc.object in identity_data
            if known_actor and known_behavior and known_object:
                docs_to_rem.append(doc_id)
            elif known_actor and known_behavior:
                profile = get_optimal(M,t,H,S_o,g_o,['a','b'], identity_data[doc.actor],behavior_data[doc.behavior])
                identities_to_add[doc.object].append([profile,doc.count])
            elif known_actor and known_object:
                profile = get_optimal(M,t,H,S_b,g_b,['a','o'], identity_data[doc.actor],identity_data[doc.object])
                behaviors_to_add[doc.behavior].append([profile,doc.count])
            elif known_object and known_behavior:
                profile = get_optimal(M,t,H,S_a,g_a,['b','o'], behavior_data[doc.behavior],identity_data[doc.object])
                identities_to_add[doc.actor].append([profile,doc.count])

        ##set value as mean of results
        for k, v in identities_to_add.items():
            add_unknown_value(identity_data,unknown_identities,k,v,init_sd,
                              init_k_unknown,init_v_unknown,
                              n_latent_topics, dirichlet_prior_topics)
        for k, v in behaviors_to_add.items():
            add_unknown_value(behavior_data,unknown_behaviors,k,v,init_sd,
                              init_k_unknown,init_v_unknown,
                              n_latent_topics, dirichlet_prior_topics)

        print 'CHANGED: ', len(identities_to_add) + len(behaviors_to_add)
        print 'LEFT: ', len(unknown_behaviors) + len(unknown_identities)

        ##get rid of docs that we have "all" information from
        for doc_id in sorted(docs_to_rem,reverse=True):
            del docs[doc_id]

        if len(identities_to_add) == 0 and len(behaviors_to_add) == 0:
            doc = docs[-1]
            known_actor = doc.actor in identity_data
            known_behavior = doc.behavior in behavior_data
            known_object = doc.object in identity_data
            random_scores = ParameterStruct("rand", 0, 1, 1,
                                            [np.random.uniform(-4.3,4.3),1],
                                            [np.random.uniform(-4.3,4.3),1],
                                            [np.random.uniform(-4.3,4.3),1],
                                            1, 1)
            rand_choice = bool(random.getrandbits(1))
            if known_actor:
                if rand_choice:
                    ##set behavior
                    profile = get_optimal(M,t,H,S_b,g_b,['a','o'], identity_data[doc.actor],random_scores)
                    add_unknown_value(behavior_data,unknown_behaviors,
                                      doc.behavior,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
                else:
                    ##set object
                    profile = get_optimal(M,t,H,S_o,g_o,['a','b'], identity_data[doc.actor],random_scores)
                    add_unknown_value(identity_data,unknown_identities,
                                      doc.object,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
            elif known_behavior:
                if rand_choice:
                    ##set object
                    profile = get_optimal(M,t,H,S_o,g_o,['a','b'], random_scores,behavior_data[doc.behavior])
                    add_unknown_value(identity_data,unknown_identities,
                                      doc.object,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
                else:
                    ##set actor
                    profile = get_optimal(M,t,H,S_a,g_a,['b','o'], behavior_data[doc.behavior],random_scores)
                    add_unknown_value(identity_data,unknown_identities,
                                      doc.actor,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
            elif known_object :
                if rand_choice:
                    ##set behavior
                    profile = get_optimal(M,t,H,S_b,g_b,['a','o'], identity_data[doc.actor],random_scores)
                    add_unknown_value(behavior_data,unknown_behaviors,
                                      doc.behavior,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
                else:
                    ##set actor
                    profile = get_optimal(M,t,H,S_a,g_a,['b','o'], behavior_data[doc.behavior],random_scores)
                    add_unknown_value(identity_data,unknown_identities,
                                      doc.actor,[[profile,1]],init_sd,init_k_unknown,init_v_unknown,
                                      n_latent_topics, dirichlet_prior_topics)
            else:
                print doc.actor, doc.behavior, doc.object, identity_data.keys(), behavior_data.keys()
                raise Exception("ASSUMING WE KNOW ONE!!!!")



def add_unknown_value(data_store,unknown,term,profiles,init_sd,init_k_unknown,init_v_unknown, n_latent_topics, dirichlet_prior_topics):
    count_sum = 0.
    e_mean_vector = []
    p_mean_vector = []
    a_mean_vector = []
    for profile,count in profiles:
        count_sum += count
        e_mean_vector += [profile[0]] * count
        p_mean_vector += [profile[1]] * count
        a_mean_vector += [profile[2]] * count

    #print term, np.mean(e_mean_vector)

    data_store[term] = ParameterStruct(term, 0, init_k_unknown, init_v_unknown,
                        [np.mean(e_mean_vector), init_sd],
                        [np.mean(p_mean_vector), init_sd],
                        [np.mean(a_mean_vector), init_sd],
                        n_latent_topics, dirichlet_prior_topics)

    unknown.remove(term)






def complete_initialization_of_docs(training_docs,
                identity_to_mu_sigma, behavior_to_mu_sigma,
                vals_matrix, coefficient_vector, coefficient_indexes,
                dirichlet_prior_topics, n_latent_topics):

    topic_prior = [dirichlet_prior_topics]*n_latent_topics
    topic_prior_sum = float(sum(topic_prior))
    for doc_iter in range(len(training_docs)):
        doc = training_docs[doc_iter]

        doc.initialize_mu_values( identity_to_mu_sigma[doc.actor],
                                  behavior_to_mu_sigma[doc.behavior],
                                  identity_to_mu_sigma[doc.object],
                                  np.copy(vals_matrix),coefficient_vector, coefficient_indexes)



        ##remember what we need to update for this actors for this doc
        identity_to_mu_sigma[doc.actor].add_doc(doc_iter,0)
        identity_to_mu_sigma[doc.object].add_doc(doc_iter, 6)
        behavior_to_mu_sigma[doc.behavior].add_doc(doc_iter,3)
