import math
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import random
import itertools
from implicit.als import AlternatingLeastSquares

def pred_all(model):
    '''Predicts for all stores and products
    
    Model parameter will go away once integrated into the implicit library'''
    return (model.user_factors).dot(model.item_factors.T)

def rmse(act,pred):
    '''Calculates to RMSE for the model
    
    PARAMETERS
    
    act: actual data
    
    pred: predicted results
    
    RETURNS
    
    root mean squared error'''
    
    return math.sqrt(1.0*sum((np.ravel(pred) - np.ravel(act))**2)/np.ravel(act).shape[0])

def train_test(test_set, neg_thresh=0, test_p = 0.3, seed=None):
    '''Given a sparse matrix, splits data into training and testing sets based on the percent, test_p, 
    which represents the percentage of the data which should be the testing data. 
    It then randomly removes actual ratings from the test data 
    for a given random seed (or None if no seed is needed)
    
    ASSUMPTIONS: Negative value is zero
    
    TODO
        1) Unit tests
    
    PARAMETERS
    test_set:   sparse matrix containing the ratings
    
    neg_thresh: threshold which represents positive and negatve ratings. Default is zero so any values less than or
                equal to neg_thresh would be negative and anything greater than would be positive
                
    test_p:      the precentage of positive value data points which should be removed from the testing data
    
    seed:        random seed value for consistent analysis
    
    RETURNS
    
    train_set:  training data which has the percent of positive values removed
    
    test_set:   original testing data
    
    samples: list of row and column combinations which were adjusted
    
    '''
    
    train_set = test_set.copy() 
    
    binary_set = test_set > neg_thresh
        
    #index of nonzero values in matrix
    nonzero_inds = binary_set.nonzero() 
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))
        
    random.seed(seed)
    n_sample = int(np.ceil(test_p*len(nonzero_pairs)))
    samples = random.sample(nonzero_pairs, n_sample)
    
    
    i_row = [i[0] for i in samples]
    i_col = [i[1] for i in samples]
    
    #TODO: Need to think about not hardcoding this and allowing it to be dynamic 
    train_set[i_row, i_col] = 0       
        
    return train_set, test_set, samples

def grid_search(data, param = {'alpha':[1],'factors': [10],'regularization': [0.01]},n_threads = 0,
                 itera = 30, metric = 'RMSE',verbose = True,**kwargs):
    '''Perform grid search to find optimal model based on varying parameters and provide optimal parameters
    based on the given performance metric. The model is trainined using the training data and then the 
    predicted value is compared to the test data based on the performance metric. This is how the model
    is optimized based on the parameters.
    
    TODO:
        1) Leverage cross-fold validation to perform the grid search
        2) Parallelize
        3) Create own class 
        4) Integrate alpha parameter to implicit library
    
    PARAMETERS
    data:    entire data set
    
    #model:   implicit model
    
    param:  set of parameters for the grid search.
    
            factors - matrix factorization
            regular - regularization to prevent overfitting
            alpha - confidence scaling
    
    n_threads: number of threads for parallelizing

    itera:  number of iterations to run
    
    metric: metric used to optimize the grid search
    
    **kwargs: Arguments for the train_test function
                
    
    RETURNS
    
    d_param: a dictionary with keys as the performance metric value from metric and values as the associated
             parameters
        
    opt_model: returns the optimal model
    
    pred:      returns all the predictions for the entire dataset'''
    
    keys, values = zip(*param.items())
    d_param = {}
    
    param_i = 0
    
    metric_ave = metric.lower()+'_ave'
    
    for i in range(itera):
        train, test, samples = train_test(data, **kwargs)
        for v in itertools.product(*values):
            params = dict(zip(keys, v))
            for k, v in params.items():
                model = AlternatingLeastSquares(num_threads=n_threads)
                if k != 'alpha':
                    setattr(model, k, v)
                #Eventually remove this and integrate the alpha parameter into implicit class. 
                #Once this is done, alpha can be treated like the previous line of code.
                else: 
                    alpha=v

            model.fit((train*alpha).transpose().astype('double'))
            pred = pred_all(model).astype('double')
            if metric == 'RMSE':
                err = rmse(test.toarray(),pred)
            else:
                raise ValueError('Invalid performance metric. Must be RMSE, ..., or ')
            
            parameter_set = 'PARAMETERS_'+str(param_i) 
            if i == 0:
                d_param[parameter_set] = params
                d_param[parameter_set][metric_ave] = err
                d_param[parameter_set]['prediction'] = pred
            else:
                ave_err = (d_param[parameter_set][metric_ave] + err)/2.0
                d_param[parameter_set][metric_ave] = ave_err
                
                ave_pred = (d_param[parameter_set]['prediction']+pred)/2
                d_param[parameter_set]['prediction']=ave_pred
            if verbose:
                print ('Iteration: {}\nParameters: {}\nRMSE: {}\n'.format(i,d_param[parameter_set],err))
            
            param_i+=1
        param_i=0
    
    #Optimal parameters
    min_metric = list(d_param.values())[0]
    for i in range(1,len(d_param.keys())):
        if list(d_param.values())[i]['rmse_ave']<min_metric['rmse_ave']:
            min_metric = list(d_param.values())[i]

    print ('Optimal Model: {}'.format(min_metric))
    #opt_model = AlternatingLeastSquares(factors=min_metric['factors'], regularization = min_metric['regularization'])
    #opt_model.fit((data*min_metric['alpha']).transpose().astype('double'))
    
    #pred = pred_all(opt_model)
    
    return d_param, min_metric, min_metric['prediction']

def store_index(rate_obj,tdlinx):
    '''Given a ratings object and tdlinx_store_cd, finds the corresponding index of that store'''
    
    return rate_obj.store.index(tdlinx)
    

def predict_store(pred,act,rate_obj,prod_info,tdlinx,n):
    '''Given a TDLINX_STORE_CD returns the top n products (Master_pkg_sku_cd & Master_pkg_sku_cd)
    TODO
    
    PARAMETERS
    
    pred:   All predictions for a given implicit model
    
    rating: ratings object
    
    prod_info: Product information
    
    tdlinx: The 7-digit tdlinx store code
    
    n:      The number of top n products to return
    
    RETURNS
    
    scores: The score of the recommendation
    
    prod_cd: Product description
    
    prod_dsc: Product code
    
    df:       dataframe containing all products and a comparison between actual and predicted values
    '''
    
    store_index = rate_obj.store.index(tdlinx)
    
    d = {'MASTER_PKG_SKU_CD':rate_obj.product,
         'PREDICT':np.around(pred[store_index],decimals=2),
         'ACTUAL':np.around(act.toarray()[store_index],decimals=2)}
    df = pd.DataFrame(data=d)
    df = df.merge(prod_info[['MASTER_SKU_CD','MASTER_SKU_DSC']].drop_duplicates(),
                           how='left',left_on = 'MASTER_PKG_SKU_CD',right_on = 'MASTER_SKU_CD')
    
    df_nonbuy = df[df['ACTUAL']==0.0].sort('PREDICT',ascending=False)
    
    scores = df_nonbuy['PREDICT'].tolist()[:n]
    prod_cd = df_nonbuy['MASTER_PKG_SKU_CD'].tolist()[:n]
    prod_dsc = df_nonbuy['MASTER_SKU_DSC'].tolist()[:n]
    
    keep = ['MASTER_PKG_SKU_CD','MASTER_SKU_DSC','ACTUAL','PREDICT']
    return scores,prod_cd,prod_dsc,df[keep].sort_values('PREDICT',ascending=False)
