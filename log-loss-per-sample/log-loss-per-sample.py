import math
def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    out = []
    for i in range(len(y_pred)):
        pred = min(1-eps, max(eps, y_pred[i]))
        L = -(y_true[i]*math.log(pred) + (1-y_true[i])*math.log(1-pred))
        out.append(L)
    return out