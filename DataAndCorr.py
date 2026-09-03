import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import rioxarray
import xcdat as xc
import seaborn as sns
import unityandnormal as u_n
#C:\Users\nunoh\Desktop\THESIS\OriginalData\gaugesRhine
from datetime import timedelta
import warnings  
warnings.filterwarnings("ignore")
from pyextremes import EVA
from pyextremes import plot_threshold_stability

def spearmanr_pval(x,y):
    return stats.spearmanr(x,y).pvalue


def matrixplot(data, figname):
    merged_data_i = pd.DataFrame(data.reset_index())
    merged_data_i = merged_data_i.drop(columns=["time"])

    temp_expl_corr = np.array(merged_data_i.corr(method='spearman'))

    # temp_expl_pval = np.array(merged_data_i.corr(method=spearmanr_pval))

    # u_n.corrmatrixplot(merged_data_i.columns, temp_expl_corr, temp_expl_pval)
    # plt.savefig(figname, dpi=300)
    return temp_expl_corr

def Qtomonth(loc, col_name):

    Q = xr.open_dataset(loc)
    Q = Q.assign_coords(Date=Q["Date"]).swap_dims({"index": "Date"})
    Q = Q.drop_vars("index")
    Q = Q.rename({"Date": "time"})
    Q = Q.rename({"Q":col_name})
    return Q

def prepareQs(folderloc):
    p = folderloc
    listofdischarges  = []

    for i,e in enumerate(os.scandir(p)):
        if e.is_file():
            with open(e.path, "r") as f:
                print("Reading:", e.name)
                Q = Qtomonth(e.path, str(i))
                Q = Q[[str(i)]].to_dataframe()#.reset_index()[["time", str(i)]]
                listofdischarges.append(Q)

    
    Qs = pd.concat(listofdischarges, axis=1, join="inner")
    # Qs.dropna(inplace=True)

    return Qs

def combineQ_P_T(P,T,Qloc):
    qs = prepareQs(Qloc)

    # P_df = P[["rr"]].to_dataframe()
    # T_df = T[["tg"]].to_dataframe()

    catchment = pd.concat([P,T,qs],axis=1,join="inner")

            # pisuerga = pisuerga.set_index("time")
    catchment = catchment.rename(columns={'rr':'P', 'tg':'T'})
    f, axs = plt.subplots(3,1,sharex=True, figsize=(12,8))
    axs[0].plot(catchment['P'])
    axs[1].plot(catchment['T'])
    axs[2].plot(catchment[['0','1','2']])
    return catchment

def lagcheck(df, columnname, lagname, thresholdvalue, declustime, lag):

    df_origin = df.copy()

    df = df.iloc[lag:-lag]

    argmax = []


    model = EVA(df['0'])
    model.get_extremes(method="POT", threshold=thresholdvalue, r=declustime)

    df_thresh = model.extremes
    
    df_thresh.index.name = "time"

    Q_1_lags = pd.DataFrame(df_origin[lagname].loc[df_thresh.index])
    Q_1_lags.rename(columns={lagname:f'0'}, inplace=True)
    Q_1_lags.reset_index(drop=True, inplace=True)

    for i in range(lag-1):
        laggedseries = pd.DataFrame(df_origin[lagname].loc[df_thresh.index-timedelta(days=i+1)])
        laggedseries.rename(columns={lagname:f'{i+1}'}, inplace=True)
        laggedseries.reset_index(drop=True, inplace=True)

        # print(laggedseries)
        Q_1_lags = pd.concat([Q_1_lags, laggedseries], axis=1)

    return Q_1_lags, df_thresh