import pickle
import time
import warnings
from gml import GML



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    begin_time = time.time()
    # data preparationasd
    path = '/home/ssd1/chn/easy-all/result_5.22/code/FSIC_GML_git/pkl/'
    with open(path+'v_feature_p.pkl', 'rb') as v:
        variables = pickle.load(v)
    with open(path+'f_feature_p.pkl', 'rb') as f:
        features = pickle.load(f)

    graph = GML.initial("example.config", variables, features)
    #inference
    graph.inference()
    #Output reasoning time 
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - begin_time))
