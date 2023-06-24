import random
from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np

# import global_var
import gml_utils
from pyds import MassFunction
# from global_var import LESS_CLUSTER,FACTOR_NAME2ID
import math
import pickle

class Regression:
    '''
    Calculate evidence support by linear regression
    '''
    def __init__(self, evidences, n_job, effective_training_count_threshold = 2,para=None,factor_type='unary'):
        '''
        @param evidences:
        @param n_job:
        @param effective_training_count_threshold:
        '''
        self.para = para
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        if len(evidences) > 0:
            XY = np.array(evidences)
            self.X = XY[:, 0].reshape(-1, 1)
            self.Y = XY[:, 1].reshape(-1, 1)
        else:
            self.X = np.array([]).reshape(-1, 1)
            self.Y = np.array([]).reshape(-1, 1)
        self.balance_weight_y0_count = 0
        self.balance_weight_y1_count = 0
        for y in self.Y:
            if y > 0:
                self.balance_weight_y1_count += 1
            else:
                self.balance_weight_y0_count += 1
        self.sample_weight_list = None
        if factor_type == 'unary':
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                self.sample_weight_list = list()
                # adjust_coefficient = 0.5
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                # sample_weight *= (adjust_coefficient)
                sample_weight = 10
                for y in self.Y:
                    if y[0] > 0:
                        self.sample_weight_list.append(sample_weight)
                    else:
                        self.sample_weight_list.append(1)
        # elif factor_type == 'bert':
        #     if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
        #         self.sample_weight_list = list()
        #         sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
        #         sample_weight *=0.5
        #         for y in self.Y:
        #             if y[0] > 0:
        #                 self.sample_weight_list.append(sample_weight)
        #             else:
        #                 self.sample_weight_list.append(1)
        #         print('bert Weight', sample_weight)
        elif factor_type == 'binary':

            self.balance_weight_y0_count = 0
            self.balance_weight_y1_count = 0
            for y in self.Y:
                if y > 0:
                    self.balance_weight_y1_count += 1
                else:
                    self.balance_weight_y0_count += 1
        self.perform()

    def perform(self):
        '''
        Perform linear regression
        @return:
        '''
        self.N = np.size(self.X)
        if self.N <= self.effective_training_count:
            self.regression = None
            self.residual = None
            self.meanX = None
            self.variance = None
            self.k = None
            self.b = None
        else:
            sample_weight_list = None
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(sample_weight)
                    else:
                        sample_weight_list.append(1)
            self.Y = np.array(self.Y, dtype=np.float)
            # print('sample_weight_list: ',self.sample_weight_list)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y, sample_weight= sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  #The average value of feature_value of all evidence variables of this feature
            self.variance = np.sum((self.X - self.meanX) ** 2)
            self.k = self.regression.coef_[0][0]
            self.b = self.regression.intercept_[0]

class EvidentialSupport:
    '''
    Calculated evidence support
    '''
    def __init__(self,variables,features):
        '''
        @param variables:
        @param features:
        '''
        self.variables = variables
        self.features = features
        self.features_easys = dict()  # Store all easy feature values of all features    :feature_id:[[value1,bound],[value2,bound]...]
        self.tau_and_regression_bound = 10
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.observed_variables_set = set()
        self.poential_variables_set = set()
        self.classNum = 5
        self.varname2varid = {}

    def separate_feature_value(self,c):
        '''
        Select the easy feature value of each feature for linear regression
        :return:选择每个特征的easy特征值进行线性回归
        '''
        binary_featureid_set = set() # 双因子集
        each_feature_easys = list() # 每一个easy
        self.features_easys.clear()
        for feature in self.features:
            if feature['Association_category'] != c : continue
            if 'sliding' in feature['feature_name']:
                sliding_window_feature_easys=[]
                for var_id,value in feature['weight'].items():
                    if type(var_id) == int:
                        if var_id in self.observed_variables_set:
                            sliding_window_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -0.25) * 10])

                # print('sliding_window evidence: ',len(sliding_window_feature_easys))
                self.features_easys[feature['feature_id']] = copy(sliding_window_feature_easys)

            if 'distance' in feature['feature_name']:
                distance_window_feature_easys=[]
                for var_id,value in feature['weight'].items():
                    if type(var_id) == int:
                        if var_id in self.observed_variables_set:
                            distance_window_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == feature['Association_category'] else 0)])
                # print('distance_output_factor evidence: ',len(distance_window_feature_easys))
                self.features_easys[feature['feature_id']] = copy(distance_window_feature_easys)

            if feature['parameterize'] == 1 and feature['feature_type']=='binary_feature':
                # each_feature_easys.clear()
                for var_id, value in feature['weight'].items():
                    if type(var_id) ==tuple:
                        binary_featureid_set.add(feature['feature_id']) # 判断两个是不是证据集变量中
                        if self.variables[var_id[0]]['is_evidence'] and self.variables[var_id[1]]['is_evidence']:
                            if self.variables[var_id[0]]['label'] == self.variables[var_id[1]]['label']: # 判断是否属于同一类
                                each_feature_easys.append([value[1], 1]) # 将两个证据集变量为同类的特征加入进去[特征，10]
                            else :
                                each_feature_easys.append([value[1], 0])#将两个证据集变量不为同类的特征加入进去[特征，-10]
                self.features_easys[feature['feature_id']] = copy(each_feature_easys) # 将证据变量之间的特征加进去

    # 用到了
    def influence_modeling(self,update_feature_set):
        '''
        Perform linear regression on the updated feature
        @param update_feature_set:影响力建模
        @return:
        '''

        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            for feature_id in update_feature_set:
                # For some features whose features_easys is empty, regression is none after regression
                if self.features[feature_id]['parameterize'] == 1:
                    if self.features[feature_id]['feature_type'] == 'unary_feature':
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],
                                                                            n_job=self.n_job, factor_type='unary')
                    elif self.features[feature_id]['feature_type'] == 'binary_feature':
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job,factor_type='binary')

    def init_tau_and_alpha(self, feature_set):
        '''
        Calculate tau and alpha for a given feature
        @param feature_set:
        @return:
        '''
        if type(feature_set) != list and type(feature_set) != set:
            raise ValueError('feature_set must be set or list')
        else:
            for feature_id in feature_set:
                if self.features[feature_id]['parameterize'] == 1 and self.features[feature_id]['feature_type'] != 'binary_feature':
                    #tau value is fixed as the upper bound
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound
                    weight = self.features[feature_id]["weight"]
                    labelvalue0 = 0
                    num0 = 0
                    labelvalue1 = 0
                    num1 = 0
                    for key in weight:
                        if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
                            labelvalue0 += weight[key][1]
                            num0 += 1
                        elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
                            labelvalue1 += weight[key][1]
                            num1 += 1
                    if num0 == 0 and num1 == 0:
                        continue
                    if num0 == 0:
                        # If there is no label0 connected to the feature, the value is assigned to the lower bound of value, which is currently set to 1
                        #labelvalue0 = min(set(self.features[feature_id].keys()))
                        tmp = []
                        for k,v in self.features[feature_id]['weight'].items():
                            tmp.append(v[1])
                        nptmp = np.array(tmp)
                        labelvalue0 = nptmp.min()
                    else:
                        # The average value of the feature value with a label of 0
                        labelvalue0 /= num0
                    if num1 == 0:
                        tmp = []
                        #labelvalue1 = max(set(self.features[feature_id].keys()))
                        for k,v in self.features[feature_id]['weight'].items():
                            tmp.append(v[1])
                            nptmp = np.array(tmp)
                            labelvalue1 = nptmp.max()
                    else:
                        # The average value of the feature value with label of 1
                        labelvalue1 /= num1
                    alpha = (labelvalue0 + labelvalue1) / 2
                    # alpha == labelvalue0 == labelvalue1, init influence always equals 0.
                    if labelvalue0 == labelvalue1:
                        if num0 > num1:
                            alpha += 1
                        elif num0 < num1:
                            alpha -= 1
                    self.features[feature_id]["alpha"] = alpha
                else:
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound #10
                    self.features[feature_id]["alpha"] = 0.5

    def evidential_support_by_regression(self,variable_set,update_feature_set,c=0):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        cur_update_set = update_feature_set.copy() #

        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables) # 获取样本证据变量和隐变量
        self.separate_feature_value(c)
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                if feature['Association_category'] == c:
                    cur_update_set.add(fid)
        del_set = set()
        for fid in cur_update_set:
            if self.features[fid]['Association_category']!=c :
                del_set.add(fid)
        cur_update_set -= del_set
        if len(cur_update_set)!=0:
            self.influence_modeling(cur_update_set)
            coo_data, mapping,mapping2= self.create_csr_matrix(c)
            coo_data = coo_data.tocoo()
            row, col, data = coo_data.row, coo_data.col, coo_data.data
            coefs = []
            intercept = []
            residuals = []
            Ns = []
            meanX = []
            variance = []
            delta = self.delta
            zero_confidence = []
            feature_is_sliding_window = []
            # unary_feature_count =global_var.UNARY_COUNT
            for fid in cur_update_set:
                feature = self.features[fid]
                if feature['feature_type'] != 'binary_feature':

                    if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >0:
                        feature['monotonicity'] = True
                    else:
                        feature['monotonicity'] = False
                    if feature['regression'].regression is not None and feature['monotonicity'] == True:
                        coefs.append(feature['regression'].regression.coef_[0][0])
                        intercept.append(feature['regression'].regression.intercept_[0])
                        zero_confidence.append(1)
                    else:
                        coefs.append(0)
                        intercept.append(0)
                        zero_confidence.append(0)

                    Ns.append(feature['regression'].N if feature['regression'].N > feature['regression'].effective_training_count else np.NaN)
                    residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
                    meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
                    if feature['regression'].variance is not None:
                        if feature['regression'].variance == 0:
                            variance.append(1e-6)
                        else:
                            variance.append(feature['regression'].variance)
                    else:
                        variance.append(np.NaN)
            coefs = np.array(coefs)[col]
            intercept = np.array(intercept)[col]
            zero_confidence = np.array(zero_confidence)[col]
            residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                             np.array(variance)[col]
            tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
            confidence = np.ones_like(data)
            confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
            confidence = confidence * zero_confidence
            evidential_support = (1 + confidence) / 2  # 正则化

            csr_evidential_support = csr_matrix((evidential_support, (row, col)), shape=(len(self.variables), len(mapping2)))
            for index, var in enumerate(self.variables):
                for feature_id in var['feature_set']:
                    if self.features[feature_id]['Association_category'] ==c and self.features[feature_id]['feature_type'] =='unary_feature':
                        var['feature_set'][feature_id][0] = csr_evidential_support[index, mapping2[feature_id]]
            predict = data * coefs + intercept
            espredict = predict * evidential_support
            espredict = espredict * zero_confidence

            assert len(np.where(evidential_support < 0)[0]) == 0
            assert len(np.where((1 - evidential_support) < 0)[0]) == 0
            unary_feature_count = len(mapping2)
            loges = np.log(evidential_support)
            evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), unary_feature_count))

            logunes = np.log(1 - evidential_support)
            evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), unary_feature_count))

            p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
            p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))

            approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables),unary_feature_count))
            approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)

            for index, var in enumerate(self.variables):
                if 'approximate_probability' not in var.keys():
                    var['approximate_probability'] = [.0] * self.classNum
                if 'approximate_weight' not in var.keys():
                    var['approximate_weight'] = [.0] * self.classNum # 赋值为5个0

                var['approximate_weight'][c] = approximate_weight[index]

            for var_id in self.poential_variables_set:

                index = var_id
                var_p_es = p_es[index]
                var_p_unes = p_unes[index]
                self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))

    def ev_by_regression(self,update_feature_set,c=-1):
        delta = self.delta
        cur_update_set = update_feature_set.copy()
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value(c)
        if update_feature_set == None or (type(update_feature_set) == set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid, feature in enumerate(self.features):
                if feature['Association_category'] == c:
                    cur_update_set.add(fid)
        del_set = set()
        for fid in cur_update_set:
            if self.features[fid]['Association_category'] != c:
                del_set.add(fid)
        cur_update_set -= del_set
        self.influence_modeling(cur_update_set)
        binary_approximate_weight = np.zeros(shape=(len(self.variables),self.classNum), dtype=np.float) # 赋给binary_approximate_weight全为0的矩阵[100,5]
        p_es = np.ones(shape=(len(self.variables),))
        p_unes = np.ones(shape=(len(self.variables),))

        for feature in self.features :
            if feature['feature_type'] == 'binary_feature':

                feature['binary_weight'] = {}
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
                if feature['regression'].regression is not None and feature['monotonicity'] == True:
                    zero_confidence = 1
                else:
                    zero_confidence = 0
                Ns = feature['regression'].N if feature['regression'].N > feature[
                    'regression'].effective_training_count else np.NaN
                residuals = feature['regression'].residual if feature['regression'].residual is not None else np.NaN
                meanX = feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN
                variance = feature['regression'].variance if feature['regression'].variance is not None else np.NaN
                regression = feature['regression'].regression
                if regression != None and zero_confidence == 1 and np.isnan(Ns) == False and np.isnan(
                        residuals) == False and np.isnan(meanX) == False and np.isnan(variance) == False:
                    regression_k = regression.coef_[0][0]
                    regression_b = regression.intercept_[0]
                    for weight in feature['weight']:
                        vid1 = weight[0]
                        vid2 = weight[1]
                        featurevalue = feature['weight'][weight][1]
                        tvalue = float(delta) / (
                                    residuals * np.sqrt(1 + 1.0 / Ns + np.power(featurevalue - meanX, 2) / variance))
                        confidence = 1 - t.sf(tvalue, (Ns - 2)) * 2
                        evidential_support = (1 + confidence) / 2
                        self.variables[vid1]['feature_set'][feature['feature_id']][0] = evidential_support
                        self.variables[vid2]['feature_set'][feature['feature_id']][0] = evidential_support
                        thisweight = evidential_support * (regression_k * featurevalue + regression_b) * zero_confidence
                        if self.variables[vid2]['is_evidence'] == True and thisweight >= 0:

                            binary_approximate_weight[vid1][self.variables[vid2]['label']] += thisweight
                            p_es[vid1]*= evidential_support
                            p_unes[vid1] *= (1 - evidential_support)
                        if self.variables[vid1]['is_evidence'] == True and thisweight >= 0:
                            binary_approximate_weight[vid2][self.variables[vid1]['label']] += thisweight
                            p_es[vid2] *= evidential_support
                            p_unes[vid2] *= (1 - evidential_support)

        for index, var in enumerate(self.variables):
            if 'approximate_weight' not in var.items():
                var['approximate_weight'] = [.0] * self.classNum
            var['approximate_weight'] += binary_approximate_weight[index]

        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            unary_evidential_support  = 0.5
            if 'evidential_support' in self.variables[index]:
                unary_evidential_support =  self.variables[index]['evidential_support']
            binary_evidential_support =  float(var_p_es / (var_p_es + var_p_unes))

            ds_numerator = unary_evidential_support * binary_evidential_support
            ds_denominator_1 = (1-unary_evidential_support)* (1-binary_evidential_support)
            self.variables[index]['evidential_support'] = float(ds_numerator /(ds_numerator+ds_denominator_1))


    def create_csr_matrix(self,c):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        unary_feature_count = 0
        mapping = {}
        mapping2 = {}

        for eachfea in self.features:
            if eachfea['feature_type'] == 'unary_feature' and  eachfea['Association_category'] == c:
                mapping[unary_feature_count] = eachfea['feature_id']
                mapping2[eachfea['feature_id']] = unary_feature_count
                unary_feature_count += 1

        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                if feature_id not in mapping2: continue
                if self.features[feature_id]['feature_type'] != 'binary_feature':
                    data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)  # feature_value
                    row.append(index)
                    col.append(mapping2[feature_id])
        return csr_matrix((data, (row, col)), shape=(len(self.variables), unary_feature_count)),mapping,mapping2


