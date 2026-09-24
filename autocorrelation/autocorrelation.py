def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    mean_series = sum(series)/len(series)
    std_series = sum([ (ele - mean_series)**2 for ele in series])
    corr_vals = [1]
    for k in range(1, max_lag+1):
        corr_val = 0
        for i in range(len(series)-k):
            ele = series[i]
            corr_val += (ele - mean_series)*(series[i+k] - mean_series)
        if std_series==0:
            corr_vals.append(0)
            continue
        corr_vals.append(corr_val/std_series)
    return corr_vals