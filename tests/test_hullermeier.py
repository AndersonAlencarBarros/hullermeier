import numpy as np
from random import SystemRandom
from src.hullermeier import hullermeier

def test_comparacao_com_outra_implementacao():

    sr = SystemRandom()

    U = np.array([[1.0, 0, 0], [0, 1.0, 1.0]])
    V = np.array([[1.0, 1, 0], [0, 0, 1]])

    n = 16  # Cluster
    k = 10  # Dimensao
    for _ in range(10000):
        U = np.array([[sr.random() for _ in range(k)] for _ in range(n)])
        V = np.full((n, k), 1 / n)

        U = U / np.sum(U, axis=0, keepdims=1)

        h = hullermeier(U, V)

        assert h >= 0