import numpy as np
#import xgboost as xgb
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.cross_validation import StratifiedKFold
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import normalize
#from ml_metrics import auc
#from scipy.sparse import csr_matrix
#from sklearn.decomposition import TruncatedSVD
#from sklearn.preprocessing import StandardScaler
#from tsne import bh_sne
#import os
#from sklearn.datasets import dump_svmlight_file
#import scipy.sparse as sp
from sklearn.cluster import KMeans
#from santander_preprocess import *
from sklearn.cluster import MiniBatchKMeans
PATH = 'data/'

train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'test.csv')

#train, test = process_base(train, test)
#train, test = drop_sparse(train, test)
#train, test = drop_duplicated(train, test)
#train, test = add_features(train, test, ['SumZeros'])
#train, test = normalize_features(train, test)

features = [x for x in train.columns if not x in ['ID_code','target']]

kmeans = []
for cluster in [1000, 3000,8000]:
    cls = MiniBatchKMeans(n_clusters=cluster, batch_size=40000)
    cls.fit_predict(train[features].values)
    train['kmeans_cluster'+str(cluster)] = cls.predict(train[features].values)
    test['kmeans_cluster'+str(cluster)] = cls.predict(test[features].values)
    kmeans.append('kmeans_cluster'+str(cluster))

train[['ID_code']+kmeans].append(test[['ID_code']+kmeans], ignore_index=True).to_csv(PATH + 'kmeans_feat_1000.csv', index=False)