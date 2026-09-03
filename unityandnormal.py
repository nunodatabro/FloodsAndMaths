import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def sd_normal(data):
    
    #ranked data to uniform
    M = data.shape[0]
    ranks = data.rank(axis=0)
    u_hat = ranks / (M + 1)
    sd_data = stats.norm.ppf(u_hat)
    return pd.DataFrame(sd_data, columns=data.columns)

def unity(data):

    if isinstance(data, (np.ndarray, np.generic)):
        data = pd.DataFrame(data)
    elif hasattr(data, 'to_dataframe'):  # xarray DataArray or Dataset
        data = data.to_dataframe()
        
    M = data.shape[0]  # Reading number of observations per node
    ranks = data.rank(axis=0)
    u_hat = ranks / (M + 1)
    return u_hat

def corrmatrixplot(columns, corr,pvals):
    nam = columns
    px = list(range(len(nam)))
    fig, axes = plt.subplots(1,2, figsize=(15,6), layout='constrained')
    im1=axes[0].imshow(corr, cmap='Blues', vmin=-1, vmax=1)
    axes[0].set_xticks(px, nam, rotation=90)
    axes[0].set_yticks(px, nam)
    axes[0].set_title('Observed rank correlations',fontsize=18)
    zz2 = np.round(corr, 2)
    zz = zz2.astype(str)
    for i in range(len(nam)):
        for j in range(len(nam)):
            if zz2[i,j]>0.5:
                color = 'w'
            else:
                color = 'k'
            axes[0].text(j, i, zz[i, j],
                            ha="center", va="center",
                            fontsize=14, color=color)

    im1=axes[1].imshow(pvals, cmap='Blues', vmin=-1, vmax=1)
    axes[1].set_xticks(px, nam, rotation=90)
    axes[1].set_yticks(px, nam)
    axes[1].set_title('Corresponding p-values',fontsize=18)
    zz2 = np.round(pvals, 2)
    zz = zz2.astype(str)
    for i in range(len(nam)):
        for j in range(len(nam)):
            if zz2[i,j]>0.5:
                color = 'w'
            else:
                color = 'k'
            axes[1].text(j, i, zz[i, j],
                            ha="center", va="center",
                            fontsize=14, color=color)
            
    cbar = fig.colorbar(im1,ax=axes[1], fraction=0.05, pad=0.04)