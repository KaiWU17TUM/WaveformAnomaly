import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def ecg_segment(ecg_cleaned, rpeaks=None, sampling_rate=1000, **kwargs):
    """
    Returns
     -------
     dict
         A dict containing DataFrames for all segmented heartbeats.

     """
    if rpeaks is None:
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, correct_artifacts=True)
        rpeaks = rpeaks["ECG_R_Peaks"]

    if len(ecg_cleaned) < sampling_rate * 4:
        raise ValueError("The data length is too small to be segmented.")

    epochs_start, epochs_end, average_hr = _ecg_segment_window(
        rpeaks=rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    heartbeats = nk.epochs_create(
        ecg_cleaned,
        rpeaks,
        sampling_rate=sampling_rate,
        epochs_start=epochs_start,
        epochs_end=epochs_end,
    )

    # Pad last heartbeats with nan so that segments are equal length
    last_heartbeat_key = str(np.max(np.array(list(heartbeats.keys()), dtype=int)))
    after_last_index = heartbeats[last_heartbeat_key]["Index"] < len(ecg_cleaned)
    for col in ["Signal", "ECG_Raw", "ECG_Clean"]:
        if col in heartbeats[last_heartbeat_key].columns:
            heartbeats[last_heartbeat_key].loc[after_last_index, col] = np.nan

    # Plot or return plot axis (feature meant to be used internally in ecg_plot)

    return heartbeats


def _ecg_segment_window(
        heart_rate=None,
        rpeaks=None,
        sampling_rate=1000,
        desired_length=None,
        ratio_pre=0.35,
):
    # Extract heart rate
    if heart_rate is not None:
        heart_rate = np.mean(heart_rate)
    if rpeaks is not None:
        heart_rate = np.mean(
            nk.signal_rate(
                rpeaks, sampling_rate=sampling_rate, desired_length=desired_length
            )
        )

    window_size = 60 / 75  # Beats per second

    # Window
    epochs_start = ratio_pre * window_size
    epochs_end = (1 - ratio_pre) * window_size

    return -epochs_start, epochs_end, heart_rate



def extract_heartbeats(patient_ecg):
    signals, info = nk.ecg_process(patient_ecg, sampling_rate=62.475)

    # Extract clean ECG and R-peaks location
    rpeaks = info["ECG_R_Peaks"]
    cleaned_ecg = signals["ECG_Clean"]

    epochs = ecg_segment(cleaned_ecg, rpeaks=rpeaks, sampling_rate=62.475)

    remove_list = []
    for key in epochs.keys():
        if any(ele > 0.5 or ele < -0.5 for ele in epochs[key]["Signal"]):
            remove_list.append(key)
    for key in remove_list:
        epochs.pop(key)

    return epochs


def HeartBeat_to_df(epochs, size):
    df = pd.DataFrame()
    i=0

    for patient in epochs:
        for key in epochs[patient].keys():
            i=i+1
            df[str(i)] =  epochs[patient][key]["Signal"].ffill()

    #Take a random sample of heartbeats from the dataframe to train the clustering algorithm
    df = df.T.dropna().sample(frac = 1)[0:size]

    return df