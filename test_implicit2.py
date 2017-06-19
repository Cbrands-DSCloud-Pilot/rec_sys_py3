import unittest
import pandas as pd
import scipy.sparse as sparse
import numpy as np
import random
import implicit2

class TestImplicit2(unittest.TestCase):

    def test_rmse(self):
        #Confirm if arrays are the same values then return zero
        act = np.array([1,2,3])
        pred = np.array([1,2,3])
        self.assertEqual(implicit2.rmse(act,pred), 0)
        
        #Confirm calculation is correct to 5 decimal places 0.57735
        act = np.array([1,2,3])
        pred = np.array([2,10,4])
        self.assertEqual(round(implicit2.rmse(act,pred),5), 4.69042)
        
        #Confirm calculation is correct when for multi-dimensions
        act = np.array([[1],[2],[3]])
        pred = np.array([[2],[10],[4]])
        self.assertEqual(round(implicit2.rmse(act,pred),5), 4.69042)
    
    def test_train_test(self):
        A = sparse.csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
        A_train = sparse.csr_matrix([[0,2,0],[0,0,3],[0,0,5]])
        train, test, samples = implicit2.train_test(A,0,0.3,1)
        
        self.assertTrue(A.shape==train.shape==test.shape)
        self.assertTrue((A!=test).nnz==0)
        self.assertTrue((A_train!=train).nnz==0)
        self.assertTrue(samples==[(0,0),(2,0)])

if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestImplicit)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()
