from typing import List, Text
import math
import numpy as np


class TFIDF:
    """Implements TFIDF from scratch
    TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)
    To calculate TF-IDF, we need:
    - term (t) for Term Frequency (TF) and Inverse Document Frequency (IDF)
    - document (d) for Term Frequency (TF)
    - Collection of documents (D) for Inverse Document Frequency (IDF)
    """
    np.random.seed(0)

    def __init__(self) -> None:
        """ This is the collections of documents `DOCUMENTS`"""
        self.DOCUMENTS = ["the tall big man", "the tall woman", "tall big man"]

        self.VOCAB = self.vocabulary()
        self.TERM_DOCUMENT_MATRIX = self.td_matrix()
        self.TERM_FREQUENCY_MATRIX = self.tf()
        self.DOCUMENT_FREQUENCY_MATRIX = self.df()
        self.IDF = self.idf()
        self.x = self.tfidf()

    def vocabulary(self) -> List[Text]:
        """Tokenize on whitespace and generate the unique set of vocabulary
        Iterate over the collections of documents `self.DOCUMENTS` 
        to calculate the unique set of vocabulary. 

        ::Return: List[Text]
        """
        # YOUR CODE HERE
        

    def td_matrix(self) -> np.ndarray:
        """Generate a term-document matrix. This can done by many ways. Here is a simple way:

        1. Create a bag of words (i.e., list) for each document in DOCUMENTS. You can use `split()` method.
        2. Calculate the counts of words in these bags of words. You can use `count()` method. 
           You will need to use `self.VOCAB` as well. Append all counts to a list. This results in a list of lists.
        3. You can use `np.array()` to convert this list of lists into a `np.ndarray`

        ::Return: np.ndarray
        """
        # YOUR CODE HERE
       

    def tf(self) -> np.ndarray: 
        """Compute term frequency. Now you have the term-document matrix. You can compute the `tf = (t,d)` using 
        tf(t,d) = log_10(count(t,d)+1). You will need to use `self.TERM_DOCUMENT_MATRIX` and math.log10() or numpy.log10()
        Then, you can use np.array() to return ndarray
        
        ::Return: np.ndarray
        """
        # YOUR CODE HERE
        

   
    
    def df(self) -> np.ndarray:
        """
        Calculate document frequency. You will need to use `self.TERM_DOCUMENT_MATRIX` which is 
        np.ndarray. You will need to read about np.array.sum()
        https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        """
        # YOUR CODE HERE
        


    def idf(self) -> np.ndarray:
        """
        Calculate Inverse Document Frquency (IDF) by using `idf(t,D)`. 
        You will need to use `self.DOCUMENTS` and `self.DOCUMENT_FREQUENCY_MATRIX`. 
        You can use math.log10() or numpy.log10()
        """
        # YOUR CODE HERE



    def tfidf(self) -> np.ndarray:
        """
        Calculate `tfidf` by multiplying `self.TERM_FREQUENCY_MATRIX` and `self.IDF`
        """
        # YOUR CODE HERE




if __name__ == '__main__':
    TFIDF()

