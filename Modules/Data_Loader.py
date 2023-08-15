import numpy as np
import pandas as pd
import scipy.signal as sp
import wfdb
from pathlib import Path
import re

def load_Data(database_name, max_records_to_load):

    # each subject may be associated with multiple records
    subjects = wfdb.get_record_list(database_name)

    # iterate the subjects to get a list of records
    records = []
    for subject in subjects:
        studies = wfdb.get_record_list(f'{database_name}/{subject}')
        for study in studies:
            records.append(Path(f'{subject}{study}'))
            # stop if we've loaded enough records
            if len(records) >= max_records_to_load:
                break

    return records

def filter_Data(database_name, required_sigs, req_seg_duration, records):
    matching_recs = {'dir':[], 'seg_name':[], 'length':[]}

    for record in records:
        record_dir = f'{database_name}/{record.parent}'
        record_dir = re.sub("\\\\", "/", record_dir)
        record_name = record.name

        record_data = wfdb.rdheader(record_name,
                                    pn_dir=record_dir,
                                    rd_segments=True)

        # Check whether the required signals are present in the record
        sigs_present = record_data.sig_name
        if not all(x in sigs_present for x in required_sigs):
            continue

        # Get the segments for the record
        segments = record_data.seg_name

        # Check to see if the segment is long enough
        # If not, move to the next one
        gen = (segment for segment in segments if segment != '~')
        for segment in gen:
            segment_metadata = wfdb.rdheader(record_name=segment,
                                             pn_dir=record_dir)
            seg_length = segment_metadata.sig_len/(segment_metadata.fs)
            if seg_length < req_seg_duration:
                continue

            # Next check that all required signals are present in the segment
            sigs_present = segment_metadata.sig_name

            if all(x in sigs_present for x in required_sigs):
                matching_recs['dir'].append(record_dir)
                matching_recs['seg_name'].append(segment)
                matching_recs['length'].append(seg_length)
                # Since we only need one segment per record break out of loop
                break

    print(f"A total of {len(matching_recs['dir'])} records met the requirements:")
    return matching_recs

def extract_Signal(records, signal_name, num_records):
    sig = {}

    #extract signals for ecg, blood pressure and ppg
    for rel_segment_n in range(num_records):
        rel_segment_name = records['seg_name'][rel_segment_n]
        rel_segment_dir = records['dir'][rel_segment_n]
        segment_data = wfdb.rdrecord(record_name=rel_segment_name,pn_dir=rel_segment_dir)

        inilist = [m.start() for m in re.finditer(r"/", rel_segment_dir)]
        subject_id = rel_segment_dir[inilist[3]+2:inilist[4]]

        for sig_no in range(0,len(segment_data.sig_name)):
            if signal_name in segment_data.sig_name[sig_no]:
                sig_col = sig_no

        sig[subject_id] = segment_data.p_signal[:,sig_col]

    return sig

def Remove_nan(signals):
    mask = np.isnan(signals)
    for i in range(len(signals)):
        if mask[i]:
            signals[i] =signals[i-1]
    return signals

def reduce_noise_butterworth(original_sig, fs):
    # filter cut-offs
    lpf_cutoff = 0.7 # Hz
    hpf_cutoff = 10 # Hz

    # create filter
    sos_filter = sp.butter(10, [lpf_cutoff, hpf_cutoff],
                           btype = 'bp',
                           analog = False,
                           output = 'sos',
                           fs = fs)
    filtered_sig = {}
    for rel_segment_n in original_sig:
        filtered_sig[rel_segment_n]=sp.sosfiltfilt(sos_filter, original_sig[rel_segment_n])

    return filtered_sig