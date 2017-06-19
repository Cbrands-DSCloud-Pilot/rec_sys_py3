
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import random
import time
import itertools
from scipy.stats import mstats



class ratings(object):
    
    def __init__(self,data,quantCol = 'L90_TY_QTY',storeCol = 'TDLINX_STORE_CD',productCol = 'MASTER_PKG_SKU_CD'):
        self.data = data.sort_values([storeCol,productCol], axis=0, ascending=True).reset_index()
        self.store = sorted(list(self.data[storeCol].unique()))
        self.product = sorted(list(self.data[productCol].unique()))
        self.quantity = np.array(self.data[quantCol],dtype=np.float32)     
        self.rating = self.quantity
        
        self._quantity = {'Data Column':quantCol,'Transformation':[]}
        self._store = storeCol
        self._product = productCol
        self._rating = {'Data Column':quantCol,'Transformation':[]}
    
    def lte_quantTrans(self, thresh = 0, value = 0):
        '''Adjust quantity so that values less than or equal to a thresh is converted to the value '''
        
        self.quantity[self.quantity<=thresh] = value
        self._quantity['Transformation'].append('Values less than {} converted to {}'.format(thresh,value))
    
    def winsorize_quantTrans(self,lower = 0.10, upper = 0.10,ignore_zero = True):
        '''Winsorizes quantity
        
        
        PARAMETERS
        
        lower:  lower percentile in which to convert values. All values below this threshold will be converted to the
                lower percentile value
                
        upper:  upper percentile in which to convert values. All values above this threshold will be converted to the
                upper percentile value
        
        ignore_zero: winsorize on non-zero values
        
        RETURNS
        
        converts ratings to winsorized values
        
        '''
        if ignore_zero:
            nonzero_ind = np.nonzero(self.quantity)[0]
            self.quantity[nonzero_ind] =  mstats.winsorize(self.quantity[nonzero_ind], limits=[lower, upper])
            self._quantity['Transformation'].append('Winsorized nonzeros based on limits {}, {}'.format(lower,
                                                                                                        upper))
        else:
            self.quantity = np.array(mstats.winsorize(self.quantity,limits=[lower, upper]))
            self._quantity['Transformation'].append('Winsorized based on limits {}, {}'.format(lower,upper))

    
    def binary_rateTrans(self, p=1,n=0,thresh = 0):
        '''Converts ratings to positive, pos, or negative, neg based on the given threshold, thresh.'''

        self.rating = np.array([p if i>thresh else n for i in self.quantity],dtype=np.float32)
        
        self._rating['Transformation'].append('Ratings binarized p,n,thresh = {}, {}, {}'.format(p,n,thresh))
    
    def natLog_rateTrans(self):
        '''Converts ratings to natural log. Adds 1 to ensure values are greater than zero
        
        PARAMETERS
        
        RETURNS'''

        self.rating = np.log(self.quantity + 1)
        
        self._rating['Transformation'].append('Tranform using natural log')
        
    def percStore_rateTrans(self):
        '''Converts rating to the percentage of that product at a store
        
        TODO
            1) Allow this to be in addition to any other transformations
            2) Unit tests
        
        PARAMETERS
        
        RETURNS'''
            
        store_total = np.array(self.data.groupby(by = [self._store])[self._quantity['Data Column']].transform(sum))
        
        self.rating = self.quantity/store_total
        
        self._rating['Transformation'].append('Each rating represents the percentage of the store')
    
        
    def sparse_matrix(self):
        '''Creates a sparse matrix
        
        TODO:
            1) Unit tests
        
        PARAMETERS
        
        RETURNS'''

        df = self.data.copy()
        df['RATING'] = self.rating
        
        df_pivot = df.pivot(index=self._store, columns=self._product, values='RATING')
        df_pivot = df_pivot.fillna(value=0)
        
        self.product = list(df_pivot.columns.values)
        
        return sparse.csr_matrix(df_pivot.values)

