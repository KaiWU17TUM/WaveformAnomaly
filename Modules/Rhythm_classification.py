import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import neurokit2 as nk
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_features(ecg):
    # Automatically process the (raw) ECG signal
    signals, info = nk.ecg_peaks(ecg, sampling_rate=31.2, correct_artifacts=False, show=False)

    # Extract clean ECG and R-peaks location
    rpeaks = info["ECG_R_Peaks"]

    # RR-intervals are the differences between successive peaks
    rr = np.diff(rpeaks)

    #Features
    hr = 60 / rr
    meanrr = np.mean(rr)
    stdrr = np.std(rr)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    meanhr = np.mean(hr)
    stdhr = np.std(hr)
    minhr = np.min(hr)
    maxhr = np.max(hr)
    nn = np.sum(np.abs(np.diff(rr)) > 50)*1
    pnn = 100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / len(rr)

    return meanrr, stdrr, rmssd, meanhr, stdhr, minhr, maxhr, nn, pnn

def get_patient_features(segment, patient):
    features_only = pd.DataFrame(columns = ['MeanRR', 'SDRR', 'RMSSD', 'MeanHR', 'STDHR', 'MinHR', 'MaxHR', 'NN50', 'PNN50'])
    ecg_signal_processed = []

    for i in range(len(segment)):
        try:
            meanrr, stdrr, rmssd, meanhr, stdhr, minhr, maxhr, nn, pnn = extract_features(segment[i])
            features_only = features_only.append({'MeanRR' : meanrr, 'SDRR' : stdrr, 'RMSSD' : rmssd, 'MeanHR': meanhr,
                                                  'STDHR': stdhr, 'MinHR': minhr, 'MaxHR': maxhr, 'NN50':nn, 'PNN50':pnn},
                                                 ignore_index = True)
            ecg_signal_processed.append(segment[i])
        except:
            continue

    features_only = features_only.fillna(0)


    # Create a StandardScaler object
    scaler = StandardScaler()
    # Fit and transform the data
    features_only[['MeanRR', 'SDRR', 'RMSSD', 'MeanHR', 'STDHR', 'MinHR', 'MaxHR', 'NN50', 'PNN50']] = scaler.fit_transform(features_only[['MeanRR', 'SDRR', 'RMSSD', 'MeanHR', 'STDHR', 'MinHR', 'MaxHR', 'NN50', 'PNN50']])

    return features_only, ecg_signal_processed

def cluster(features_only):

    clustering = DBSCAN(eps=1, min_samples=30).fit(features_only)
    DBSCAN_dataset = features_only.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_

    clusters = DBSCAN_dataset.Cluster.value_counts().to_frame()
    y_pred = clustering.fit_predict(features_only)

    anomoly_indices = np.where(y_pred == -1)[0]

    return clusters, y_pred, anomoly_indices

def pca_plots(patient, features_only, y_pred):
    pca = PCA(n_components=2)
    pca.fit(features_only)
    components = pca.fit_transform(features_only)

    # Label to color dict (manual)
    label_color_dict = {0: "green", -1: "red", 1: "blue", 2: "purple", 3:"orange", 4:"yellow", 5:"black"}
    cvec = [label_color_dict[label] for label in y_pred]
    fig, ax = plt.subplots(figsize=(18,8))
    ax.scatter(components[:,0],components[:,1], color=cvec)

    plt.savefig('anomalies_clustering/Rhythm/PCA/'+patient+'.png')
    plt.close(fig)


def plot_anomalous_segments(ecg_signal_processed, anomoly_indices, patient):
    test = ecg_signal_processed
    segment_len = len(ecg_signal_processed[0])
    num = 0

    for i in range(0, int(len(test)/10), 10):
        num=num+1
        fig, ax = plt.subplots(figsize=(18,8))
        ax.set_title("Anomolous signals")
        ax.set_xlabel("Time (seconds)")

        test2 = list(chain.from_iterable(test[i:i+10]))
        t = np.arange(0,len(test2))
        ax.plot(t, test2, color = "blue")
        for j in range(i,i+10):
            if j in anomoly_indices:
                start = (j - i)* segment_len
                ax.plot(t[start:start+segment_len],test2[start:start+segment_len], color = "red")
        plt.savefig('anomalies_clustering/Rhythm/'+patient+'/'+str(num)+'.png')
        plt.close(fig)