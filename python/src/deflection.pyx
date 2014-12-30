#!python
#distutils: language = c++
#cython: boundscheck=False
#cython: wraparound=False


import numpy as np
from libcpp.vector cimport vector
from cpython cimport array
cimport numpy as np
from utility import *

cdef str index_to_abo_epa(int option):
    if option == 0:
        return 'ae'
    if option == 1:
        return 'ap'
    if option == 2:
        return 'aa'
    if option == 3:
        return 'be'
    if option == 4:
        return 'bp'
    if option == 5:
        return 'ba'
    if option == 6:
        return 'oe'
    if option == 7:
        return 'op'
    if option == 8:
        return 'oa'


def get_coefficient_matrices(deflection_computation):
    base_coeff = deflection_computation[0].coefficient
    coeff_vector = np.array([d.coefficient for d in deflection_computation[1:]])
    vals_matrix = np.ones((len(deflection_computation)-1,18))
    coefficient_indexes = []
    for x in range(18):
        coefficient_indexes.append([])
    for d_it in range(1,len(deflection_computation)):
        var_counter = deflection_computation[d_it].variable_counter
        for i in range(9):
            if var_counter[i] == 1:
                coefficient_indexes[i].append(d_it-1)
            elif var_counter[i] == 2:
                coefficient_indexes[i+9].append(d_it-1)

    return base_coeff, coeff_vector, vals_matrix, coefficient_indexes




def get_optimal(M,t,H,S,g,scores_had, scores_had_one, scores_had_two):
    scores_had_one = [ scores_had_one.E_M[0],scores_had_one.P_M[0],scores_had_one.A_M[0]]
    scores_had_two =[ scores_had_two.E_M[0],scores_had_two.P_M[0],scores_had_two.A_M[0]]

    I = np.identity(9+M.shape[0])
    abo_epa_codes_scores_had = [score_had+epa_val for score_had in scores_had for epa_val in ['e','p','a']]
    ind_scores_had = [epa_abo_to_index(code) for code in abo_epa_codes_scores_had ]
    all_scores_had = scores_had_one + scores_had_two

    for ind_iter in range(len(ind_scores_had)):
        I[ind_scores_had[ind_iter],ind_scores_had[ind_iter]] = all_scores_had[ind_iter]

    for x in range(M.shape[0]):
        ib_index = x+9
        for code_iter in range(len(abo_epa_codes_scores_had)):
            if abo_epa_codes_scores_had[code_iter] in t[x]:
                I[ib_index] *= all_scores_had[code_iter]
    ##compute optimal behavior
    tmp_compute = S.transpose().dot(I).dot(H).dot(I)
    optimal_epa_scores = -np.dot(np.linalg.inv(tmp_compute.dot(S)),tmp_compute.dot(g.transpose()))
    return optimal_epa_scores


cdef class DeflectionScoreEntity:
    cdef public vector[int] variable_counter
    cdef public float coefficient
    cdef public str variable_str


    def __init__(self, set_one_vars, float set_one_coeff, set_two_vars=[], float set_two_coeff=1.):

        for i in range(9):
            self.variable_counter.push_back(0)

        for x in set_one_vars:
            val = epa_abo_to_index(x)
            self.variable_counter[val] += 1
        for x in set_two_vars:
            val = epa_abo_to_index(x)
            self.variable_counter[val] += 1


        self.coefficient = set_one_coeff * set_two_coeff

    def gen_variable_str(self):
        self.variable_str = ""
        terms = []
        for i in range(9):
            if self.variable_counter[i] == 0:
                continue
            terms.append(index_to_abo_epa(i) + "^" + str(self.variable_counter[i]))

        terms = sorted(terms)

        temp = " ".join(terms)
        self.variable_str = temp

    def add_in_other(self, other):
        assert self == other
        self.coefficient += other.coefficient

    cpdef sub_in_value(self, int abo_epa_index, float value):
        cdef int counter_val
        counter_val = self.variable_counter[abo_epa_index]

        if counter_val == 0:
            return

        self.coefficient *= value
        if counter_val == 2:
            self.coefficient *= value

        self.variable_counter[abo_epa_index] = 0
        self.gen_variable_str()

    cpdef float get_value_for_sub_all_but(self, int focus_index, vector[float] value_list):
        cdef float to_return = self.coefficient

        for i in range(9):

            if i != focus_index and self.variable_counter[i] != 0:
                to_return *= value_list[i]
                if self.variable_counter[i] == 2:
                    to_return *= value_list[i]
        return to_return

    cpdef int get_v(self, int index):
        return self.variable_counter[index]

    def __richcmp__(DeflectionScoreEntity self, DeflectionScoreEntity other, int cmp):
        return COMPARE_SCORE_ENTITIES[cmp](self.variable_str, other.variable_str)

    def __str__(self):
        return str(self.coefficient) + " " + self.variable_str




def compute_square(coeff_array, variables):
    result = {}
    for j in range(len(coeff_array)):
        j_coeff_value = coeff_array[j]
        if abs(j_coeff_value) < 0.000001:
            continue
        for k in range(len(coeff_array)):
            k_coeff_value = coeff_array[k]

            if abs(k_coeff_value) < 0.000001:
                continue
            j_variables = variables[j]
            k_variables = variables[k]

            defentity = DeflectionScoreEntity(j_variables, j_coeff_value,
                                              k_variables, k_coeff_value)
            defentity.gen_variable_str()
            if defentity.variable_str in result:
                result[defentity.variable_str].add_in_other(defentity)
            else:
                result[defentity.variable_str] = defentity
                #result.append(defentity)
    return result



def get_coefficients(M, t):
    all_deflections = []

    for i in range(9):
        M_copy = np.copy(-M)
        M_copy[i + 1][i] += 1
        deflections = compute_square(M_copy[:, i], t)
        all_deflections.append(deflections)

    final_deflections = {}
    for defl_vector in all_deflections:
        for d, v in defl_vector.items():
            if d in final_deflections:
                final_deflections[d].add_in_other(v)
            else:
                final_deflections[d] = v

    return final_deflections.values()

def compute_s_and_g(M, t, abo_option, EPA_LIST):
    S_init = np.zeros((9, 3))
    abo_index = 3 * abo_to_index(abo_option)
    S_init[abo_index, 0] = 1
    S_init[abo_index + 1, 1] = 1
    S_init[abo_index + 2, 2] = 1

    S = np.zeros((M.shape[0], 3))
    for x in range(3):
        for y in range(M.shape[0]):
            if abo_option + EPA_LIST[x] in t[y]:
                S[y][x] = 1
    S = np.vstack((S_init, S))
    return S, np.ones(S.shape[0]) - S.dot(np.ones(3))

def get_t_and_M_from_file(eq_filename, fundamentals):
    M = []
    t = []
    equation_file = open(eq_filename)
    i = 0
    for line in equation_file:
        t.append(set())
        line_spl = [l.strip() for l in line.split("\t")]
        M.append([float(x) for x in line_spl[1:]])

        coeff = line_spl[0].replace("Z","")
        for j in range(len(coeff)):
            if coeff[j] == '1':
                t[i].add(fundamentals[j])
        i+=1

    equation_file.close()
    return t, np.array(M)




cdef c_get_constants_for_fundamental(int focus_index,
                                     deflection_computation,
                                    vector[float] value_list):
    cdef float coeff_0 = 0.
    cdef float coeff_1 = 0.
    cdef float coeff_2 = 0.
    cdef int counter_val
    cdef DeflectionScoreEntity defentity

    for entity_it in range(len(deflection_computation)):
        defentity = deflection_computation[entity_it]
        counter_val = defentity.variable_counter[focus_index]
        if counter_val == 2:
            coeff_0 += defentity.get_value_for_sub_all_but(focus_index, value_list)
        elif counter_val == 1:
            coeff_1 += defentity.get_value_for_sub_all_but(focus_index, value_list)
        else:
            coeff_2 += defentity.get_value_for_sub_all_but(focus_index, value_list)

    return [coeff_0, coeff_1, coeff_2]

def compute_deflection_bad(values, deflection_computation):
    defs = []
    for val in range(9):
        deflection = 0
        c0,c1,c2 = c_get_constants_for_fundamental(val,deflection_computation,values)
        deflection += c0*values[val]*values[val] + c1 * values[val] + c2
        defs.append([c0,c1,c2,deflection])
    return defs


def compute_deflection_simple(values, deflection_computation):
    c0,c1,c2 = c_get_constants_for_fundamental(0,deflection_computation,values)
    return c0*values[0]*values[0] + c1 * values[0] + c2

def get_constants_for_fundamental(int focus_index, deflection_computation, array.array value_list):
    cdef vector[float] values_vector = value_list

    cdef np.ndarray[dtype=object,ndim=1] a = deflection_computation
    deflection_computation = a
    return c_get_constants_for_fundamental(focus_index,deflection_computation,value_list)


def compute_deflection(values, deflection_computation):
    ##assert len(values) == len(FUNDAMENTALS)
    for val in range(len(values)):
        for defentity in deflection_computation:
            defentity.sub_in_value(val, values[val])

    #assert [defentity.variable_str == "" for defentity in deflection_computation]
    return np.array([defentity.coefficient for defentity in deflection_computation])


COMPARE_SCORE_ENTITIES = {
    0: lambda x, y: x < y,
    1: lambda x, y: x <= y,
    2: lambda x, y: x == y,
    3: lambda x, y: x != y,
    4: lambda x, y: x > y,
    5: lambda x, y: x >= y,
}
