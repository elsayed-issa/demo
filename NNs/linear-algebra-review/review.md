# Basic Linear Algebra and Numpy Tutorial


- Both basic knowledge of matrix processes and numpy are important to understand for linguists who practice machine learning

## Assignment Goals
The goal of this assignment is to familiarize yourself with:

1. Parsing HTML data
2. Text classification using ngram language models
3. Text classification using supervised machine learning algorithms
4. Tools for sentiment analysis

The assignment combines tutorial components, with learning exercises that you must complete and submit. The learning exercise sections are clearly demarcated within the assignments.

## Before you start
1. PULL THE LATEST VERSION OF THE `course-materials` REPOSITORY, AND COPY `homework/HW2/` INTO THE CORRESPONDING DIRECTORY OF YOUR SUBMISSION FOLDER
2. CREATE AND ATTACH TO A VIRTUAL ENVIRONMENT, AND INSTALL THE REQUIREMENTS IN `requirements.txt`
3. IMPORT THE COURSE UTILITIES BY RUNNING THE CODE BLOCK BELOW


### Notation
1. s: scalar
2. v: vector
2. M: matrix

## Scalar:

## Vector

<h1>Matrix</h1>

<h2>Definitions</h2>
- Matrices (sg. matrix) are the core concepts in linear algebra.
- The Matrix is a rectangular array of numbers.

$$
M =    
\begin{bmatrix}
a_{11} & a_{12} & ... & a_{1n}\\
a_{21} & a_{22} & ... & a_{2n}\\
... &  &  & \\
a_{m1} & a_{m2} & ... & a_{mn}\\
\end{bmatrix}
$$

$M$ is called `mxn` matrix (m by n) where `m` are the **rows** and `n` are the **columns**.

**Example**

$$
A =    
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
\end{bmatrix}
$$

- A is **2x3** (2 by 3) matrix and to access any of the matrix `6` elements, the following:

$$a_{ij}$$
- where `i` represents the `rows` and `j` represents the `columns`.

**Question**

- What element is repreented by $a_{21}$? 
- Answer should be `4`

**Practice**

```python
import numpy as np

# set the matrix
matrix_A = np.array([[1,2,3], 
              [4,5,6]])
# slice the matrix at a21 
# 2 is the second row with index (1) and 1 is the first element with index (0)
matrix_A[1,0]
```
**NOTE:** 2 matrices are identical if they are:
1. the same size
2. $a_{ij}$ = $b_{ij}$

$$
A =    
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}

B =    
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}

C =    
\begin{bmatrix}
1 & 2  \\
3 & 3  \\
\end{bmatrix}
$$

- $A$ = $B$ but $A$ != $B$

<h2>Matrix Transpose:</h2>
- Matrix A has its transpose by swapping rows with columns.

$$
A =    
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6  \\
\end{bmatrix}

A^T =    
\begin{bmatrix}
1 & 2 & 3  \\
4 & 5 & 6  \\
\end{bmatrix}
$$
- This means that `A` is `3x2` matrix and its transpose becomes `2x3` matrix.

**Practice**

```python
import numpy as np

# set the matrix
a = np.array([[1,2], [3,4], [5,6]])

# transpose
a.T # or a.transpose()
```
<h2>Matrix Addition</h2>

**scalar:**
- add `2` to `A` matrix.
$$
A =    
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}
+ 2 = 
\begin{bmatrix}
1 +2 & 2+2  \\
3+2 & 4+2  \\
\end{bmatrix}
= 
\begin{bmatrix}
3 & 4  \\
5 & 6  \\
\end{bmatrix}
$$

**vector**

- add `a=[1,2]` vector to `A` matrix.

$$
A + a =     
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
5 & 6
\end{bmatrix}
+

\begin{bmatrix}
1  \\
2  \\
\end{bmatrix}
$$

- `A` is `3x2` matrix and `a` is `2x1` vector so, it is important that have the same dimensionality. The trick here is to find the shapes and if the inner numbers are equal, we can add them.

**matrix**

- add `A` and `B` matrices.

$$
A+B = 
A
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}
+
B
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}
= 
\begin{bmatrix}
1+1 & 2+2  \\
3+3 & 4+4  \\
\end{bmatrix}
=
\begin{bmatrix}
2 & 4  \\
6 & 8  \\
\end{bmatrix}
$$

**NOTE:** the two matrices should have the same dimensionality.

**NOTE:** matrix subtraction works the same way.

**properties of matrix addition and subtraction**
1. A+B = B+A (commutative)
2. ...

<h2>Matrix Multiplication:</h2>

- CONDITION: `AB` (A times B) is defined only if `columns of A` = `rows of B`

**scalar**

- multiply `A` matrix by `2`
$$
A =    
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
\end{bmatrix}
* 2 = 
\begin{bmatrix}
1*2 & 2*2  \\
3*2 & 4*2  \\
\end{bmatrix}
= 
\begin{bmatrix}
2 & 4  \\
6 & 8  \\
\end{bmatrix}
$$

**vector**

- multiply `a=[1,2]` vector by `A` matrix.

$$
A_{(3x2)} + a_{(2x1)} =     
\begin{bmatrix}
1 & 2  \\
3 & 4  \\
5 & 6
\end{bmatrix}_{(3x2)}
+
\begin{bmatrix}
1  \\
2  \\
\end{bmatrix}_{(2x1)}
=
\begin{bmatrix}
1*1+2*2 \\
3*1+4*2 \\
5*1+6*2
\end{bmatrix}
=\begin{bmatrix}
1+4 \\
3+8 \\
5+12
\end{bmatrix}
=\begin{bmatrix}
5 \\
11 \\
17
\end{bmatrix}_{(3x1)}
$$

**matrix**

- multiply `A` and `B`

$$
A_{(3x3)} * B_{(3x2)} =     
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6  \\
7 & 8 & 9
\end{bmatrix}_{(3x3)}
+
\begin{bmatrix}
1 & 1 \\
2 & 2  \\
3 & 3
\end{bmatrix}_{(3x2)}
=
\begin{bmatrix}
1*1+2*2+3*3 &  1*1+2*2+3*3 \\
4*1+5*2+6*3 & 4*1+5*2+6*3 \\
7*1+8*2+9*3 & 7*1+8*2+9*3
\end{bmatrix}

$$

$$
=
\begin{bmatrix}
1+4+9 &  1+4+9 \\
4+10+18 & 4+10+18 \\
7+16+27 & 7+16+27
\end{bmatrix}
= 
\begin{bmatrix}
14 &  14 \\
32 & 32 \\
50 & 50
\end{bmatrix}_{(3x2)}
$$

**Properties of Matrix multiplication:**
1. ,,

<h2>Matrix Inverse</h2>

<h2>Matrix Division</h2>

<h2>Dot Product</h2>










<h2>References</h2>
- Hill, C. (2020). Learning scientific programming with python. Cambridge University Press.