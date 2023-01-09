# Homework assignment (DEMO)

- This is just a demo and the assignment is subject to several changes.

# Programming `TFIDF` from scratch

- This assignment teaches you to implement a term frequency–inverse document frequency (TFIDF) from scratch.

## The Assignment:

- You can a file called `assignment.py` in which you need to read instrcutions in docstrings carefully. Then you write your code under `YOUR CODE HERE`
- In order to test your implementation whether it is right or wrong, you can run the test cases in `tests.py`
- In `tests.py`, there are 6 tests. If they all passed you will have a message as in `Passed`, but if one or more failed, you will have a message as in `Failed` (`FAILED (failures=1)`)

### Passed

```python
----------------------------------------
Ran 6 tests in 0.001s

OK
```
### Failed
```python
----------------------------------------
Ran 6 tests in 0.002s

FAILED (failures=1)
```

## Requiremnets:

To run this assignment, you need:

- In oder to run the assignment, you will need to install [`docker`](https://docs.docker.com/install/). 

- You can use any code editor you like.  You may want to try [Visual Studio Code](https://code.visualstudio.com/).


## Getting started:

- First, use `git` to clone the repo on your machine: [`git tutorial`: https://www.atlassian.com/git/tutorials]

```bash
git clone https://github.com/elsayed-issa/demo.git
```

- Second, build your docker image:

```bash
docker build -t [NAME-OF-IMAGE] .
```

- Third, start conding. If you want to run `assignment.py` or `tests.py`, you can use docker as follows:

```bash
docker run -it --rm -v $PWD:/app [NAME-OF-IMAGE] python assignment.py
```

```bash
docker run -it --rm -v $PWD:/app [NAME-OF-IMAGE] python tests.py
```

NOTE: if you do not want to use docker, you can use build a virtual environment using `anaconda` as follows:

```bash
conda create --name [NAME-OF-VENV] python=3.8 ipython
conda activate [NAME-OF-VENV]
# execute the following command from the project root:
pip install numpy
```

[`see more about conda here:` https://www.anaconda.com]