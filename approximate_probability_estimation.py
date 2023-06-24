import logging
import gml_utils
from pyds import MassFunction
import math
import pickle
from scipy.special import expit, logit

class ApproximateProbabilityEstimation:
    '''
    Approximate probability calculation
    '''
    def __init__(self, variables,features):
        self.variables = variables
        self.features = features
        self.classNum = 5

    def approximate_probability_estimation_by_interval(self, variable_set):
        for i in range(self.classNum):
            if type(variable_set) == set or type(variable_set) == list:
                for id in variable_set:
                    if 'approximate_probability' not in self.variables[id]:
                        self.variables[id]['approximate_probability'] = [.0] * self.classNum
                    self.variables[id]['approximate_probability'][i] = gml_utils.open_p(self.variables[id]['approximate_weight'][i])
                    self.variables[id]['entropy'] = gml_utils.entropy(max(self.variables[id]['approximate_probability']))
            else:
                raise ValueError('input type error')
