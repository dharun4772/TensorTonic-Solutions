import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    #initialize random weights or W and B
    #start loop
    #caluclate Z
    #Loss L
    #take step for W and b
    # if W does not increase more than a delta: stop
    #end loop
    N = X.shape[0]
    rng = np.random.default_rng(seed=42)
    W = np.zeros((X.shape[1],))
    b = np.zeros((1,))
    for i in range(steps):
        z = X@W + b
        p = _sigmoid(z)
        L = - (1/N) * np.sum(y*np.log(p) + (1-y)*(np.log(1-p)))
        dw = np.dot(X.T, (p-y)) / N
        db = np.sum(p-y, axis =0, keepdims=True) / N
        W-=lr*dw
        b-=lr*db
    return W, b