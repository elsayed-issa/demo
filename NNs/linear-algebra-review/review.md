# Basic Linear Algebra and Numpy Tutorial


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








