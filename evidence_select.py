import logging
import random
from copy import copy


class EvidenceSelect:
    def __init__(self, variables, features, interval_evidence_count= 200,subgraph_limit_num=1000,each_feature_evidence_limit = 2000,k_hop=1):
        self.variables = variables
        self.features = features
        self.subgraph_limit_num = subgraph_limit_num    #Maximum number of variables allowed in the subgraph
        self.interval_evidence_count = interval_evidence_count   #Uniformly divided into 10 intervals, the number of evidence variables sampled in each interval
        self.k_hop = k_hop
        self.each_feature_evidence_limit = each_feature_evidence_limit   #Limit the number of evidence variables for each single factor in the subgraph

    def select_evidence_by_multi(self, var_id):
        '''
        Uniform evidence selection method
        @param var_id:
        @return:
        connected_var_set :  Subgraph variable set
        connected_edge_set:  Subgraph egde set
        connected_feature_set: Subgraph feature set
        '''
        if type(var_id) == int:
            subgraph_max_num = self.subgraph_limit_num
            random_sample_num = self.interval_evidence_count  # The number of evidences to be sampled for each single factor when there is no featureValue
            connected_var_set = set()  # Finally add the hidden variable id
            connected_edge_set = set()  # [feature_id,var_id]  or [feature_id,(id1,id2)]
            connected_feature_set = set()
            feature_set = self.variables[var_id]['feature_set']
            binary_feature_set = set()
            unary_feature_set = set()
            current_var_set = set() #The basic variables of the current hop
            current_var_set.add(var_id)
            next_var_set = set()
            # Divide the double factor and single factor of this hidden variable
            for feature_id in feature_set.keys():
                if self.features[feature_id]['feature_type'] == 'binary_feature':
                    binary_feature_set.add(feature_id)
                elif self.features[feature_id]['feature_type'] == 'unary_feature':
                    unary_feature_set.add(feature_id)

            for k in range(self.k_hop):
                # Each round adds the adjacent evidence variable of the previous round of hidden variables
                for varid in current_var_set:
                    feature_set = self.variables[varid]['feature_set']
                    for feature_id in feature_set.keys():
                        # If this latent variable id is contained in two variable ids connected by a two-factor, and the other variable is an evidence variable,
                        # then this evidence variable is added to the next round of next_var_set, and the relevant features and edges are counted
                        if self.features[feature_id]['feature_type'] == 'binary_feature':
                            weight = self.features[feature_id]['weight']
                            for id in weight.keys():
                                if type(id) == tuple and varid in id:
                                    another_var_id = id[0] if id[0] != varid else id[1]
                                    if self.variables[another_var_id]['is_evidence'] == True:
                                        connected_feature_set.add(feature_id)
                                        connected_edge_set.add((feature_id, id))
                                        connected_var_set.add(another_var_id)
                                    elif self.variables[another_var_id]['is_evidence'] == False:
                                        next_var_set.add(another_var_id)
                    connected_var_set = connected_var_set.union(current_var_set)
                current_var_set.clear()
                current_var_set = copy(next_var_set)
                next_var_set.clear()

            # Deal with single factor: limit the number of evidence variables for each single factor, and sample when it exceeds
            subgraph_capacity = subgraph_max_num - len(connected_var_set) - 1  # Maximum number of variables that can be added
            unary_evidence_set = set()
            unary_potential_set = set()
            # First judge whether all single-factor evidence is added to whether the maximum limit is exceeded
            for feature_id in unary_feature_set:
                weight = self.features[feature_id]['weight']
                for vid in weight.keys():
                    if self.variables[vid]['is_evidence'] == True:
                        unary_evidence_set.add(vid)
                    else:
                        unary_potential_set.add(vid)

            return connected_var_set, connected_edge_set, connected_feature_set
        else:
            raise ValueError('input type error')