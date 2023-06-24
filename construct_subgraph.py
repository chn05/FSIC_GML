import random
from collections import namedtuple
import logging
from numbskull_extend.numbskulltypes import *

class ConstructSubgraph:
    def __init__(self, variables, features,balance):
        self.variables = variables
        self.features = features
        self.balance = balance      #if need to balance the number of 0 and 1 in the evidence variable
        self.classNum = 5

    def construct_subgraph_for_multi(self,var_id,evidences): # both noPara
        connected_var_set, connected_edge_set, connected_feature_set = evidences
        var_map = dict()
        var_num = len(connected_var_set)
        variable = np.zeros(var_num, Variable)
        variable_index = 0
        for id in connected_var_set:
            variable[variable_index]["isEvidence"] = self.variables[id]['is_evidence']
            if variable[variable_index]["isEvidence"]:
                variable[variable_index]["initialValue"] = self.variables[id]['true_label']
            else :
                variable[variable_index]["initialValue"] = self.variables[id]['label']
            variable[variable_index]["dataType"] = 1
            variable[variable_index]["cardinality"] = self.classNum
            var_map[id] = variable_index
            variable_index += 1

        binary_feature_edge = list()
        unary_feature_edge = list()
        for elem in connected_edge_set:
            if self.features[elem[0]]['feature_type'] == 'unary_feature':
                unary_feature_edge.append(elem)
            elif self.features[elem[0]]['feature_type'] == 'binary_feature':
                binary_feature_edge.append(elem)

        binary_feature_set = set()
        unary_feature_set = set()
        for fea_id in connected_feature_set:
            if self.features[fea_id]['feature_type'] == 'unary_feature':
                unary_feature_set.add(fea_id)
            elif self.features[fea_id]['feature_type'] == 'binary_feature':
                binary_feature_set.add(fea_id)

        weight = np.zeros(len(binary_feature_set) + len(unary_feature_set), Weight)
        wmap = np.zeros(len(weight), WeightToFactor)
        alpha_bound = np.zeros(len(weight), AlphaBound)
        tau_bound = np.zeros(len(weight), TauBound)
        feature_map_weight = dict()
        weight_map = dict()
        weight_dict = dict()
        weight_index = 0
        for feature_id in binary_feature_set:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = 1
            if self.features[feature_id]['parameterize'] :
                weight[weight_index]["a"] = self.features[feature_id]['tau']
                weight[weight_index]["b"] = self.features[feature_id]['alpha']
                alpha_bound[weight_index]['lowerBound'] = -5
                alpha_bound[weight_index]['upperBound'] = 10
                tau_bound[weight_index]['lowerBound'] = 0.2
                tau_bound[weight_index]['upperBound'] = 0.999
            else:
                weight[weight_index]["initialValue"] = 5
            feature_map_weight[feature_id] = weight_index
            weight_map[weight_index] = feature_id
            weight_dict[weight_index] = set()
            weight_index += 1
        for feature_id in unary_feature_set:
            weight[weight_index]["isFixed"] = False
            weight[weight_index]["parameterize"] = self.features[feature_id]['parameterize']
            if self.features[feature_id]['parameterize']:
                weight[weight_index]["a"] = self.features[feature_id]['tau']
                weight[weight_index]["b"] = self.features[feature_id]['alpha']
                alpha_bound[weight_index]['lowerBound'] = -5
                alpha_bound[weight_index]['upperBound'] = 10
                tau_bound[weight_index]['lowerBound'] = 0.2
                tau_bound[weight_index]['upperBound'] = 0.999
            else:
                weight[weight_index]["initialValue"] = 5
            feature_map_weight[feature_id] = weight_index
            weight_map[weight_index] = feature_id
            weight_dict[weight_index] = set()
            weight_index += 1

        edges_num = len(unary_feature_edge) + len(binary_feature_edge) * 2
        factor = np.zeros(len(unary_feature_edge) + len(binary_feature_edge), Factor)
        fmap = np.zeros(edges_num, FactorToVar)
        wfactor = np.zeros(len(factor), FactorToWeight)
        domain_mask = np.zeros(var_num, np.bool)

        factor_index = 0
        fmp_index = 0
        edge_index = 0
        for elem in binary_feature_edge:
            factor[factor_index]["factorFunction"] = 9
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]
            factor[factor_index]["featureValue"] = 1
            factor[factor_index]["arity"] = 2
            factor[factor_index]["ftv_offset"] = fmp_index
            weight_dict[feature_map_weight[elem[0]]].add(factor_index)
            fmap[fmp_index]["vid"] = var_map[elem[1][0]]
            fmap[fmp_index]["theta"] = 1.0
            fmap[fmp_index]["x"] =self.features[elem[0]]['weight'][(elem[1][0],elem[1][1])][1]
            fmap[fmp_index]["dense_equal_to"] = self.variables[elem[1][1]]['true_label'] if var_id == elem[1][0] else \
            self.variables[elem[1][0]]['true_label']

            fmp_index += 1

            fmap[fmp_index]["vid"] = var_map[elem[1][1]]
            fmap[fmp_index]["theta"] = 1.0
            fmap[fmp_index]["x"] = self.features[elem[0]]['weight'][(elem[1][0],elem[1][1])][1]
            fmap[fmp_index]["dense_equal_to"] = self.variables[elem[1][1]]['true_label'] if var_id ==elem[1][0] else self.variables[elem[1][0]]['true_label']
            fmp_index += 1
            factor_index += 1
            edge_index += 1

        for elem in unary_feature_edge:

            factor[factor_index]["factorFunction"] = 12
            factor[factor_index]["weightId"] = feature_map_weight[elem[0]]
            factor[factor_index]["featureValue"] = 1
            factor[factor_index]["arity"] = 1
            factor[factor_index]["ftv_offset"] = fmp_index
            weight_dict[feature_map_weight[elem[0]]].add(factor_index)
            fmap[fmp_index]["vid"] = var_map[elem[1]]
            fmap[fmp_index]["theta"] = 1.0
            fmap[fmp_index]["x"] = self.variables[elem[1]]['feature_set'][elem[0]][1]
            fmap[fmp_index]["dense_equal_to"] = self.features[elem[0]]['Association_category']
            fmp_index += 1
            factor_index += 1
            edge_index += 1

        sample_list = None
        wfactor_index = 0
        for weightId, factorSet in weight_dict.items():
            count = 0
            wmap[weightId]["weightId"] = weightId
            wmap[weightId]["weight_index_offset"] = wfactor_index
            for factorId in factorSet:
                wfactor[wfactor_index]["factorId"] = factorId
                count += 1
                wfactor_index += 1
            wmap[weightId]["weight_index_length"] = count
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map, sample_list, wmap, wfactor