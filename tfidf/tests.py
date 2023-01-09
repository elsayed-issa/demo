from typing import List
import unittest
from assignment import TFIDF
import unittest
import numpy as np

class TestTFIDF(unittest.TestCase):
    np.random.seed(0)
    tfidf = TFIDF()

    def test_vocabulary(self):
        res: List = self.tfidf.vocabulary()
        assert len(res) == 5
        assert res == {'the', 'man', 'big', 'tall', 'woman'}

    def test_td_matrix(self):
        matrix_shape = (3, 5)
        td_matrix = self.tfidf.td_matrix()
        
        self.assertEqual(
            matrix_shape,
            td_matrix.shape,
            f"`TFIDF.td_matrix()` should return {td_matrix.shape}",
            )
    
    def test_tf(self):
        matrix_shape = (3, 5)
        tf = self.tfidf.tf()
        
        self.assertEqual(
            matrix_shape,
            tf.shape,
            f"`TFIDF.tf()` should return {tf.shape}",
            )

    def test_df(self):
        array_size = (5,)
        df = self.tfidf.df()

        self.assertEqual(
            array_size,
            df.shape,
            f"`TFIDF.df()` should return {df.shape}",
            )

    def test_idf(self):
        array_size = (5,)
        idf = self.tfidf.idf()

        self.assertEqual(
            array_size,
            idf.shape,
            f"`TFIDF.idf()` should return {idf.shape}",
            )
    
    def test_tfidf(self):
        array_size = (3, 5)
        x = self.tfidf.tfidf()

        self.assertEqual(
            array_size,
            x.shape,
            f"`TFIDF.tfidf()` should return {x.shape}",
            )

if __name__ == "__main__":
    unittest.main()
