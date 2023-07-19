import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets as s
import matplotlib.pyplot as plt

def low_rank_approx(SVD=None, A=None, r=1):
    if not SVD:
        SVD = np.linalg.svd(A, full_matrices=False)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar

if __name__ == "__main__":
    x = s.face(gray=True)
    plt.imshow(x, cmap='gray')
    plt.show()
    U,S,V = np.linalg.svd(x, full_matrices=False)
    for i in range(25):
        new_image = np.matrix(U[:, :i]) * np.diag(S[:i]) * np.matrix(V[:i, :])
        plt.imshow(new_image, cmap='gray')
        title = "singular values used = %s" % i
        plt.title(title)
        plt.show()
