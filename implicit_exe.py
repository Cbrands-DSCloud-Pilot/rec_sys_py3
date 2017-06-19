'''Executing implicit code and writes recommendations for products by account'''

import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import random
import time
import itertools
from scipy.stats import mstats

from implicit.als import AlternatingLeastSquares
import implicit2
from ratings import ratings
from S3_helper import S3_helper
import civis

#Load product data from S3
client = civis.APIClient()
access = client.credentials.get(client.get_aws_credential_id('S3_ACCESS_JON'))['username']
secret = client.credentials.get(client.get_aws_credential_id('S3_SECRET_JON'))['username']
bucket = 's3-ds-work-test'
cb_s3 = S3_helper(access,secret,bucket)

#Load product information
key = 'Rec_Eng/Product_info.csv'
path = 'Product_info.csv'
cb_s3.pull_file_from_s3(key, path)
prod_info = pd.read_csv(path, encoding = "ISO-8859-1")
prod_info = prod_info[prod_info['STOCK_TYPE_CD'].isin(['S','D'])]
master_sku = list(prod_info.MASTER_SKU_CD.unique())
master_sku = [str(sku) for sku in master_sku]
master_pkg = list(prod_info.MASTER_PKG_CD.unique())

#Load depletions data
data = civis.io.read_civis(table="cbi.IL_OFF_L90",
                           database="Constellation Brands",
                           use_pandas=True)

il_off_data = ratings(data,quantCol='l90_ty_qty',
                      storeCol = 'tdlinx_store_cd',
                     productCol = 'master_pkg_sku_cd')

#Data preprocessing
il_off_data.lte_quantTrans() #Remove negatives
il_off_data.winsorize_quantTrans() #Winsorize
il_off_data.natLog_rateTrans() #Convert to natural log scale
il_sparse = il_off_data.sparse_matrix()

#Prediction
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
param = {'alpha':[1,100],'factors': [10,80],'regularization': [0.001]}
d_test,opt_model,pred = implicit2.grid_search(il_sparse, param,itera=1,n_threads = 3,verbose = False, test_p=0.3)

#Clean-up and write to table
pred_results = pd.DataFrame(pred,columns = il_off_data.product)
pred_results['tdlinx_store_cd']=il_off_data.store
pred_results = pd.melt(pred_results,id_vars = 'tdlinx_store_cd')
pred_results.rename(columns={'tdlinx_store_cd': 'TDLINX_STORE_CD',
                   'variable': 'MASTER_PKG_SKU_CD',
                  'value':'PREDICTION'}, inplace=True)
pred_results = pred_results.merge(prod_info[['MASTER_SKU_CD','MASTER_SKU_DSC']].drop_duplicates(),
                           how='left',left_on = 'MASTER_PKG_SKU_CD',right_on = 'MASTER_SKU_CD')
pred_results = pred_results.drop('MASTER_SKU_CD',axis=1)
pred_table = civis.io.dataframe_to_civis(pred_results, 'Constellation Brands','cbi.IL_OFF_PRED')
pred_table.result()
