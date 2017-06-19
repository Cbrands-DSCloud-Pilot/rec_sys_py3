import unittest
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
from ratings import ratings

class TestRating(unittest.TestCase):

    def test_lte(self):
        #Check negatives are being converted and assigned to zero
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[0,-1,1,-0.1,-10]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.lte_quantTrans()
        self.assertEqual(sum(rate.quantity==np.array([0,0,1,0,0])),5)
        
        #Check negatives are being converted to -1 and varying threshold from default
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[1,-1,10,-0.1,-10]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.lte_quantTrans(thresh=1,value=-1)
        self.assertEqual(sum(rate.quantity==np.array([-1,-1,10,-1,-1])),5)
    
    def test_winsorize(self):
        #Check that winsorizing occurred at top and bottom 20%
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[1,2,3,4,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.winsorize_quantTrans(lower=0.2,upper=0.2)
        self.assertEqual(sum(rate.quantity==np.array([2,2,3,4,4])),5)
        
        #Check that winsorizing occurred at top and bottom 50% with zeros being ignored
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[0,0,3,4,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.winsorize_quantTrans(lower=0.5,upper=0.5)
        self.assertEqual(sum(rate.quantity==np.array([0,0,4,4,4])),5)
        
        #Check that winsorizing occurred at top and bottom 20% with zeros being included
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[0,2,3,4,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.winsorize_quantTrans(lower=0.2,upper=0.2,ignore_zero=False)
        self.assertEqual(sum(rate.quantity==np.array([2,2,3,4,4])),5)
        
    def test_binary(self):
        #Check that values are converted to 0,1 based on a 0 threshold
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[-1,-2,0,4,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.binary_rateTrans()
        self.assertEqual(sum(rate.rating==np.array([0,0,0,1,1])),5)
        
        #Check that values are converted to -2,2 based on a 2 threshold
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[-1,-2,0,4,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.binary_rateTrans(p=2,n=-2,thresh=2)
        self.assertEqual(sum(rate.rating==np.array([-2,-2,-2,2,2])),5)
    
    def test_natLog(self):
        #Check that values are converted to natural log and that 0's have 1 add to it before transforming
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[0,0,1,2,3]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.natLog_rateTrans()
        self.assertEqual(round(sum(rate.rating),3),3.178)
    
    def test_percStore(self):
        #Check that percent store is working properly
        data = {'TDLINX_STORE_CD':['A1234','A1234','A1234','A1234','A1234'],
                'MASTER_PKG_SKU_CD':['1234','1234','1234','1234','1234'],
                'L90_TY_QTY':[20,50,15,10,5]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        rate.percStore_rateTrans()
        self.assertEqual(sum(rate.rating==np.array([0.20,0.50,0.15,0.10,0.05])),5)

    def test_sparseMatrix(self):
        #Check that data is being properly pivoted
        data = {'TDLINX_STORE_CD':['A1234','A1234','B1234','B1234','B1234'],
                'MASTER_PKG_SKU_CD':['1234','5678','1234','5678','9012'],
                'L90_TY_QTY':[10,20,30,40,50]}

        df = pd.DataFrame(data)
        rate = ratings(df)
        sp_matrix = rate.sparse_matrix()
        self.assertTrue(sp_matrix.shape==(2,3))
        self.assertEqual(sum((sp_matrix.toarray() == np.array([[ 10.,  20.,   0.],[ 30.,  40.,  50.]])).ravel()),6)

        #check if items are ordered that data is properly pivoted
        data = {'TDLINX_STORE_CD':['A1234','A1234','B1234','B1234','B1234'],
        'MASTER_PKG_SKU_CD':['1234','5678','5678','1234','9012'],
        'L90_TY_QTY':[10,20,30,40,50]}
        df = pd.DataFrame(data)
        rate = ratings(df)
        sp_matrix = rate.sparse_matrix()
        self.assertEqual(sum((sp_matrix.toarray() == np.array([[ 10.,  20.,   0.],[ 40.,  30.,  50.]])).ravel()),6)


if __name__ == '__main__':
    unittest.main()
