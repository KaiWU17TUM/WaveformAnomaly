import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

def pulse_detect(x,fs,w,alg):
    """
    Description: Pulse detection and correction from pulsatile signals
    Inputs:  x, array with pulsatile signal [user defined units]
             fs, sampling rate of signal [Hz]
             w, window length for analysis [s]
             alg, string with the name of the algorithm to apply ['heartpy','d2max','upslopes','delineator']
    Outputs: ibis, location of cardiac cycles as detected by the selected algorithm [number of samples]
    """

    # Check selected algorithm
    pos_alg = ['heartpy','d2max','upslopes','delineator']
    if not(alg in pos_alg):
        print('Unknown algorithm determined. Using D2max as default')
        alg = 'd2max'

    # Pre-processing of signal
    x_d = sp.detrend(x)
    sos = sp.butter(10, [0.5, 10], btype = 'bp', analog = False, output = 'sos', fs = fs)
    x_f = sp.sosfiltfilt(sos, x_d)

    # Peak detection in windows of length w
    n_int = np.floor(len(x_f)/(w*fs))
    for i in range(int(n_int)):
        start = i*fs*w
        stop = (i + 1)*fs*w - 1
        # print('Start: ' + str(start) + ', stop: ' + str(stop) + ', fs: ' + str(fs))
        aux = x_f[range(start,stop)]
        if alg == 'heartpy':
            locs = heartpy(aux,fs,40,180,5)
        elif alg == 'd2max':
            locs = d2max(aux,fs)
        elif alg == 'upslopes':
            locs = upslopes(aux)
        elif alg == 'delineator':
            locs = delineator(aux,fs)
        locs = locs + start
        if i == 0:
            ibis = locs
        else:
            ibis = np.append(ibis,locs)
    if n_int*fs*w != len(x_f):
        start = stop + 1
        stop = len(x_f)
        aux = x_f[range(start,stop)]
        if len(aux) > 20:
            if alg == 'heartpy':
                locs = heartpy(aux,fs,40,180,5)
            elif alg == 'd2max':
                locs = d2max(aux,fs)
            elif alg == 'upslopes':
                locs = upslopes(aux)
            elif alg == 'delineator':
                locs = delineator(aux,fs)
            locs = locs + start
            ibis = np.append(ibis,locs)
    ind, = np.where(ibis <= len(x_f))
    ibis = ibis[ind]

    #ibis = peak_correction(x,ibis,fs,20,5,[0.5, 1.5])

    return ibis

def peak_correction(x,locs,fs,t,stride,th_len):
    """
    Correction of peaks detected from pulsatile signals

    Inputs:   x, pulsatile signal [user defined units]
              locs, location of the detected interbeat intervals [number of samples]
              fs, sampling rate [Hz]
              t, duration of intervals for the correction [s]
              stride, stride between consecutive intervals for the correction [s]
              th_len, array with the percentage of lower and higher thresholds for comparing the duration of IBIs
              [proportions]
    Outputs:  ibis, array with the corrected points related to the start of the inter-beat intervals [number of samples]
    """

    # Correction of long and short IBIs
    len_window = np.round(t*fs)
    #print('Window length: ' + str(len_window))
    first_i = 0
    second_i = len_window - 1
    while second_i < len(x):
        ind1, = np.where(locs >= first_i)
        ind2, = np.where(locs <= second_i)
        ind = np.intersect1d(ind1, ind2)

        win = locs[ind]
        dif = np.diff(win)
        #print('Indices: ' + str(ind) + ', locs: ' + str(locs[ind]) + ', dif: ' + str(dif))

        th_dif = np.zeros(2)
        th_dif[0] = th_len[0]*np.median(dif)
        th_dif[1] = th_len[1]*np.median(dif)

        th_amp = np.zeros(2)
        th_amp[0] = 0.75*np.median(x[win])
        th_amp[1] = 1.25*np.median(x[win])
        #print('Length thresholds: ' + str(th_dif) + ', amplitude thresholds: ' + str(th_amp))

        j = 0
        while j < len(dif):
            if dif[j] <= th_dif[0]:
                if j == 0:
                    opt = np.append(win[j], win[j + 1])
                else:
                    opt = np.append(win[j], win[j + 1]) - win[j - 1]
                print('Optional: ' + str(opt))
                dif_abs = np.abs(opt - np.median(dif))
                min_val = np.min(dif_abs)
                ind_min, = np.where(dif_abs == min_val)
                print('Minimum: ' + str(min_val) + ', index: ' + str(ind_min))
                if ind_min == 0:
                    print('Original window: ' + str(win), end = '')
                    win = np.delete(win, win[j + 1])
                    print(', modified window: ' + str(win))
                else:
                    print('Original window: ' + str(win), end = '')
                    win = np.delete(win, win[j])
                    print(', modified window: ' + str(win))
                dif = np.diff(win)
            elif dif[j] >= th_dif[1]:
                aux_x = x[win[j]:win[j + 1]]
                locs_pks, _ = sp.find_peaks(aux_x)

                locs_pks = locs_pks + win[j]
                ind1, = np.where(x[locs_pks] >= th_amp[0])
                ind2, = np.where(x[locs_pks] <= th_amp[1])
                ind = np.intersect1d(ind1, ind2)
                locs_pks = locs_pks[ind]

                if len(locs_pks) != 0:
                    opt = locs_pks - win[j]

                    dif_abs = np.abs(opt - np.median(dif))
                    min_val = np.min(dif_abs)
                    ind_min, = np.where(dif_abs == min_val)

                    win = np.append(win, locs_pks[ind_min])
                    win = np.sort(win)
                    dif = np.diff(win)
                    j = j + 1
                else:
                    opt = np.round(win[j] + np.median(dif))
                    if opt < win[j + 1]:
                        win = np.append(win, locs_pks[ind_min])
                        win = np.sort(win)
                        dif = np.diff(win)
                        j = j + 1
                    else:
                        j = j + 1
            else:
                j = j + 1

        locs = np.append(win, locs)
        locs = np.sort(locs)

        first_i = first_i + stride*fs - 1
        second_i = second_i + stride*fs - 1

    dif = np.diff(locs)
    dif = np.append(0, dif)
    ind, = np.where(dif != 0)
    locs = locs[ind]

    # Correction of points that are not peaks
    i = 0
    pre_loc = 0
    while i < len(locs):
        if locs[i] == 0:
            locs = np.delete(locs, locs[i])
        elif locs[i] == len(x):
            locs = np.delete(locs, locs[i])
        else:
            cond = (x[locs[i]] >= x[locs[i] - 1]) and (x[locs[i]] >= x[locs[i] + 1])
            if cond:
                i = i + 1
            else:
                if locs[i] == pre_loc:
                    i = i + 1
                else:
                    if i == 0:
                        aux = x[0:locs[i + 1] - 1]
                        aux_loc = locs[i] - 1
                        aux_start = 0
                    elif i == len(locs) - 1:
                        aux = x[locs[i - 1]:len(x) - 1]
                        aux_loc = locs[i] - locs[i - 1]
                        aux_start = locs[i - 1]
                    else:
                        aux = x[locs[i - 1]:locs[i + 1]]
                        aux_loc = locs[i] - locs[i - 1]
                        aux_start = locs[i - 1]

                    pre = find_closest_peak(aux, aux_loc, 'backward')
                    pos = find_closest_peak(aux, aux_loc, 'forward')

                    ibi_pre = np.append(pre - 1, len(aux) - pre)
                    ibi_pos = np.append(pos - 1, len(aux) - pos)
                    ibi_act = np.append(aux_loc - 1, len(aux) - aux_loc)

                    dif_pre = np.abs(ibi_pre - np.mean(np.diff(locs)))
                    dif_pos = np.abs(ibi_pos - np.mean(np.diff(locs)))
                    dif_act = np.abs(ibi_act - np.mean(np.diff(locs)))

                    avgs = [np.mean(dif_pre), np.mean(dif_pos), np.mean(dif_act)]
                    min_avg = np.min(avgs)
                    ind, = np.where(min_avg == avgs)
                    if len(ind) != 0:
                        ind = ind[0]

                    if ind == 0:
                        locs[i] = pre + aux_start - 1
                    elif ind == 1:
                        locs[i] = pos + aux_start - 1
                    elif ind == 2:
                        locs[i] = aux_loc + aux_start - 1
                    i = i + 1


    # Correction of peaks according to amplitude
    len_window = np.round(t*fs)
    #print('Window length: ' + str(len_window))
    keep = np.empty(0)
    first_i = 0
    second_i = len_window - 1
    while second_i < len(x):
        ind1, = np.where(locs >= first_i)
        ind2, = np.where(locs <= second_i)
        ind = np.intersect1d(ind1, ind2)
        win = locs[ind]
        if np.median(x[win]) > 0:
            th_amp_low = 0.5*np.median(x[win])
            th_amp_high = 3*np.median(x[win])
        else:
            th_amp_low = -3*np.median(x[win])
            th_amp_high = 1.5*np.median(x[win])
        ind1, = np.where(x[win] >= th_amp_low)
        ind2, = np.where(x[win] <= th_amp_high)
        aux_keep = np.intersect1d(ind1,ind2)
        keep = np.append(keep, aux_keep)

        first_i = second_i + 1
        second_i = second_i + stride*fs - 1

    if len(keep) != 0:
        keep = np.unique(keep)
        locs = locs[keep.astype(int)]


    return locs

def find_closest_peak(x, loc, dir_search):
    """
    Finds the closest peak to the initial location in x

    Inputs:   x, signal of interest [user defined units]
              loc, initial location [number of samples]
              dir_search, direction of search ['backward','forward']
    Outputs:  pos, location of the first peak detected in specified direction [number of samples]
    """

    pos = -1
    if dir_search == 'backward':
        i = loc - 2
        while i > 0:
            if (x[i] > x[i - 1]) and (x[i] > x[i + 1]):
                pos = i
                i = 0
            else:
                i = i - 1
        if pos == -1:
            pos = loc
    elif dir_search == 'forward':
        i = loc + 1
        while i < len(x) - 1:
            if (x[i] > x[i - 1]) and (x[i] > x[i + 1]):
                pos = i
                i = len(x)
            else:
                i = i + 1
        if pos == -1:
            pos = loc

    return pos

def seek_local(x, start, end):
    val_min = x[start]
    val_max = x[start]

    ind_min = start
    ind_max = start

    for j in range(start, end):
        if x[j] > val_max:
            val_max = x[j]
            ind_max = j
        elif x[j] < val_min:
            val_min = x[j]
            ind_min = j

    return val_min, ind_min, val_max, ind_max

def heartpy(x, fs, min_ihr, max_ihr, w):
    """
    Detects inter-beat intervals using HeartPy

    Inputs:   x, pulsatile signal [user defined units]
              fs, sampling rate [Hz]
              min_ihr, minimum value of instantaneous heart rate to be accepted [bpm]
              max_ihr, maximum value of instantaneous heart rate to be accepted [bpm]
              w, length of segments for correction of peaks [s]
    Outputs:  ibis, position of the starting points of inter-beat intervals [number of samples]

    """

    # Identification of peaks
    is_roi = 0
    n_rois = 0
    pos_pks = np.empty(0).astype(int)
    locs = np.empty(0).astype(int)

    len_ma = int(np.round(0.75*fs))
    #print(len_ma)
    sig = np.append(x[0]*np.ones(len_ma), x)
    sig = np.append(sig, x[-1]*np.ones(len_ma))

    i = len_ma
    while i < len(sig) - len_ma:
        ma = np.mean(sig[i - len_ma:i + len_ma - 1])

        # If it is the beginning of a new ROI:
        if is_roi == 0 and sig[i] >= ma:
            is_roi = 1
            n_rois = n_rois + 1

            # If it is a peak:
            if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
                pos_pks = np.append(pos_pks, int(i))

        # If it is part of a ROI which is not over:
        elif is_roi == 1 and sig[i] > ma:
            # If it is a peak:
            if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
                pos_pks = np.append(pos_pks, int(i))

        # If the ROI is over or the end of the signal has been reached:
        elif is_roi == 1 and (sig[i] < ma or i == (len(sig) - len_ma)):
            #print('End of ROI ---' + str(n_rois) + ' @ ' + str(i) + '. Pos pks: ' + str(pos_pks))
            is_roi = 0 # Lowers flag

            # If it is the end of the first ROI:
            if n_rois == 1:
                # If at least one peak has been found:
                if len(pos_pks) != 0:
                    # Determines the location of the maximum peak:
                    max_pk = np.max(sig[pos_pks])
                    ind, = np.where(max_pk == np.max(sig[pos_pks]))
                    # The maximum peak is added to the list:
                    locs = np.append(locs, pos_pks[ind])

                # If no peak was found:
                else:
                    # Counter for ROIs is reset to previous value:
                    n_rois = n_rois - 1

            # If it is the end of the second ROI:
            elif n_rois == 2:
                # If at least one peak has been found:
                if len(pos_pks) != 0:
                    # Measures instantantaneous HR of found peaks with respect to the previous peak:
                    ihr = 60/((pos_pks - locs[-1])/fs)
                    good_ihr, = np.where(ihr <= max_ihr and ihr >= min_ihr)
                    pos_pks = pos_pks[good_ihr].astype(int)

                    # If at least one peak is between HR limits:
                    if len(pos_pks) != 0:
                        # Determines the location of the maximum peak:
                        max_pk = np.max(sig[pos_pks])
                        ind, = np.where(max_pk == np.max(sig[pos_pks]))
                        # The maximum peak is added to the list:
                        locs = np.append(locs, pos_pks[ind])
                        #print('Locations: ' + str(locs))
                # If no peak was found:
                else:
                    # Counter for ROIs is reset to previous value:
                    n_rois = n_rois - 1

            # If it is the end of the any further ROI:
            else:
                # If at least one peak has been found:
                if len(pos_pks) != 0:
                    # Measures instantantaneous HR of found peaks with respect to the previous peak:
                    ihr = 60/((pos_pks - locs[-1])/fs)
                    good_ihr, = np.where(ihr <= max_ihr and ihr >= min_ihr)
                    pos_pks = pos_pks[good_ihr].astype(int)

                    # If at least one peak is between HR limits:
                    if len(pos_pks) != 0:
                        # Calculates SDNN with the possible peaks on the ROI:
                        sdnn = np.zeros(len(pos_pks))
                        for j in range(len(pos_pks)):
                            sdnn[j] = np.std(np.append(locs/fs, pos_pks[j]/fs))
                        # Determines the new peak as that one with the lowest SDNN:
                        min_pk = np.min(sdnn)
                        ind, = np.where(min_pk == np.min(sdnn))
                        locs = np.append(locs, pos_pks[ind])
                        #print('Locations: ' + str(locs))
                # If no peak was found:
                else:
                    # Counter for ROIs is reset to previous value:
                    n_rois = n_rois - 1

            # Resets possible peaks for next ROI:
            pos_pks = np.empty(0)

        i = i + 1;

    locs = locs - len_ma

    # Correction of peaks
    c_locs = np.empty(0)
    n_int = np.floor(len(x)/(w*fs))
    for i in range(int(n_int)):
        ind1, = np.where(locs >= i*w*fs)
        #print('Locs >= ' + str((i)*w*fs) + ': ' + str(locs[ind1]))
        ind2, = np.where(locs < (i + 1)*w*fs)
        #print('Locs < ' + str((i + 1)*w*fs) + ': ' + str(locs[ind2]))
        ind = np.intersect1d(ind1, ind2)
        #print('Larger and lower than locs: ' + str(locs[ind]))
        int_locs = locs[ind]

        if i == 0:
            aux_ibis = np.diff(int_locs)
        else:
            ind, = np.where(locs >= i*w*fs)
            last = locs[ind[0] - 1]
            aux_ibis = np.diff(np.append(last, int_locs))
        avg_ibis = np.mean(aux_ibis)
        th = np.append((avg_ibis - 0.3*avg_ibis), (avg_ibis + 0.3*avg_ibis))
        ind1, = np.where(aux_ibis > th[0])
        #print('Ind1: ' + str(ind1))
        ind2, = np.where(aux_ibis < th[1])
        #print('Ind2: ' + str(ind2))
        ind = np.intersect1d(ind1, ind2)
        #print('Ind: ' + str(ind))

        c_locs = np.append(c_locs, int_locs[ind]).astype(int)
        print(c_locs)

    #fig = plt.figure()
    #plt.plot(x)
    #plt.plot(sig)
    #plt.scatter(locs,x[locs],marker = 'o',color = 'red')
    #if len(c_locs) != 0:
    #plt.scatter(c_locs,x[c_locs],marker = 'o',color = 'blue')

    if len(c_locs) != 0:
        ibis = c_locs
    else:
        ibis = locs

    return ibis

def d2max(x, fs):
    """
    Detects inter-beat intervals using D2Max
    Inputs:   x, pulsatile signal [user defined units]
              fs, sampling rate [Hz]
    Outputs:  ibis, position of the starting points of inter-beat intervals [number of samples]

    """

    # Bandpass filter
    if len(x) < 4098:
        z_fill = np.zeros(4098 - len(x) + 1)
        x_z = np.append(x, z_fill)
    sos = sp.butter(10, [0.5, 8], btype = 'bp', analog = False, output = 'sos', fs = fs)
    x_f = sp.sosfiltfilt(sos, x_z)

    # Signal clipping
    ind, = np.where(x_f < 0)
    x_c = x_f
    x_c[ind] = 0

    # Signal squaring
    x_s = x_c**2

    # Blocks of interest
    w1 = (111e-3)*fs
    w1 = int(2*np.floor(w1/2) + 1)
    b = (1/w1)*np.ones(w1)
    ma_pk = sp.filtfilt(b,1,x_s)

    w2 = (667e-3)*fs
    w2 = int(2*np.floor(w2/2) + 1)
    b = (1/w2)*np.ones(w1)
    ma_bpm = sp.filtfilt(b,1,x_s)
    # Thresholding
    alpha = 0.02*np.mean(ma_pk)
    th_1 = ma_bpm + alpha
    th_2 = w1
    boi = (ma_pk > th_1).astype(int)

    blocks_init, = np.where(np.diff(boi) > 0)
    blocks_init = blocks_init + 1
    blocks_end, = np.where(np.diff(boi) < 0)
    blocks_end = blocks_end + 1
    if blocks_init[0] > blocks_end[0]:
        blocks_init = np.append(1, blocks_init)
    if blocks_init[-1] > blocks_end[-1]:
        blocks_end = np.append(blocks_end, len(x_s))

    # Search for peaks inside BOIs
    len_blks = np.zeros(len(blocks_init))
    ibis = np.zeros(len(blocks_init))
    for i in range(len(blocks_init)):
        ind, = np.where(blocks_end > blocks_init[i])
        ind = ind[0]
        len_blks[i] = blocks_end[ind] - blocks_init[i]
        if len_blks[i] >= th_2:
            aux = x[blocks_init[i]:blocks_end[ind]]
            if len(aux) != 0:
                max_val = np.max(aux)
                max_ind, = np.where(max_val == aux)
                ibis[i] = max_ind + blocks_init[i] - 1

    ind, = np.where(len_blks < th_2)
    if len(ind) != 0:
        for i in range(len(ind)):
            boi[blocks_init[i]:blocks_end[i]] = 0
    ind, = np.where(ibis == 0)
    ibis = (np.delete(ibis, ind)).astype(int)

    return ibis

def upslopes(x):
    """
    Detects inter-beat intervals using Upslopes

    Inputs:   x, pulsatile signal [user defined units]
    Outputs:  ibis, position of the starting points of inter-beat intervals [number of samples]

    """

    # Peak detection
    th = 6
    pks = np.empty(0)
    pos_pk = np.empty(0)
    pos_pk_b = 0
    n_pos_pk = 0
    n_up = 0

    for i in range(1, len(x)):
        if x[i] > x[i - 1]:
            n_up = n_up + 1
        else:
            if n_up > th:
                pos_pk = np.append(pos_pk, i)
                pos_pk_b = 1
                n_pos_pk = n_pos_pk + 1
                n_up_pre = n_up
            else:
                pos_pk = pos_pk.astype(int)
                #print('Possible peaks: ' + str(pos_pk) + ', number of peaks: ' + str(n_pos_pk))
                if pos_pk_b == 1:
                    if x[i - 1] > x[pos_pk[n_pos_pk - 1]]:
                        pos_pk[n_pos_pk - 1] = i - 1
                    else:
                        pks = np.append(pks, pos_pk[n_pos_pk - 1])
                    th = 0.6*n_up_pre
                    pos_pk_b = 0
            n_up = 0
    ibis = pks.astype(int)

    return ibis

def delineator(x, fs):
    """
    Detects inter-beat intervals using Delineator

    Inputs:   x, pulsatile signal [user defined units]
              fs, sampling rate [Hz]
    Outputs:  ibis, position of the starting points of inter-beat intervals [number of samples]

    """

    # Lowpass filter
    od = 3
    sos = sp.butter(od, 25, btype = 'low', analog = False, output = 'sos', fs = fs)
    x_f = sp.sosfiltfilt(sos, x)
    x_m = 1000*x_f

    # Moving average
    n = 5
    b = (1/n)*np.ones(n)
    x_ma = sp.filtfilt(b,1,x_m)

    # Compute differentials
    dif = np.diff(x_ma)
    dif = 100*np.append(dif[0], dif)
    dif_ma = sp.filtfilt(b,1,dif)

    # Average thresholds in original signal
    x_len = len(x)
    if x_len > 12*fs:
        n = 10
    elif x_len > 7*fs:
        n = 5
    elif x_len > 4*fs:
        n = 2
    else:
        n = 1
    #print(n)

    max_min = np.empty(0)
    if n > 1:

        n_int = np.floor(x_len/(n + 2))
        #print('Length of intervals: ' + str(n_int))
        for j in range(n):
            # Searches for max and min in 1 s intervals
            amp_min, ind_min, amp_max, ind_max = seek_local(x_ma, int(j*n_int), int(j*n_int + fs))
            max_min = np.append(max_min, (amp_max - amp_min))
        max_min_avg = np.mean(max_min)

    else:
        amp_min, ind_min , amp_max, ind_max = seek_local(x_ma, int(close_win), int(x_len))

        max_min_avg = amp_max - amp_min

    max_min_lt = 0.4*max_min_avg

    # Seek pulse beats by min-max method
    step_win = 2*fs       # Window length to look for peaks/onsets
    close_win = np.floor(0.1*fs)
    # Value of what is considered too close

    pks = np.empty(0)     # Location of peaks
    ons = np.empty(0)     # Location of onsets
    dic = np.empty(0)     # Location of dicrotic notches

    pk_index = -1          # Number of peaks found
    on_index = -1          # Number of onsets found
    dn_index = -1          # Number of dicrotic notches found

    i = int(close_win)    # Initializes counter
    while i < x_len:      # Iterates through the signal
        #print('i: ' + str(i))
        amp_min = x_ma[i] # Gets the initial value for the minimum amplitude
        amp_max = x_ma[i] # Gets the initial value for the maximum amplitude

        ind = i           # Initializes the temporal location of the index
        aux_pks = i       # Initializes the temporal location of the peak
        aux_ons = i       # Initializes the temporal location of the onset

        # Iterates while ind is lower than the length of the signal
        while ind < x_len - 1:
            #print('Ind: ' + str(ind))
            # Verifies if no peak has been found in 2 seconds
            if (ind - i) > step_win:
                #print('Peak not found in 2 s')
                ind = i   # Refreshes the temporal location of the index
                max_min_avg = 0.6*max_min_avg  # Refreshes the threshold for the amplitude
                # Verifies if the threshold is lower than the lower limit
                if max_min_avg <= max_min_lt:
                    max_min_avg = 2.5*max_min_lt # Refreshes the threshold
                break

            # Verifies if the location is a candidate peak
            if (dif_ma[ind - 1]*dif_ma[ind + 1]) <= 0:
                #print('There is a candidate peak')
                # Determines initial and end points of a window to search for local peaks and onsets
                if (ind + 5) < x_len:
                    i_stop = ind + 5
                else:
                    i_stop = x_len - 1
                if (ind - 5) >= 0:
                    i_start = ind - 5
                else:
                    i_start = 0

                # Checks for artifacts of saturated or signal loss
                if (i_stop - ind) >= 5:
                    for j in range(ind, i_stop):
                        if dif_ma[j] != 0:
                            break
                    if j == i_stop:
                        #print('Artifact')
                        break

                # Candidate onset
                #print('Looking for candidate onsets...')
                #plt.figure()
                #plt.plot(x_ma)
                if dif_ma[i_start] < 0:
                    if dif_ma[i_stop] > 0:
                        aux_min, ind_min, _, _ = seek_local(x_ma, int(i_start), int(i_stop))
                        #plt.scatter(ind_min, aux_min, marker = 'o', color = 'red')
                        if np.abs(ind_min - ind) <= 2:
                            amp_min = aux_min
                            aux_ons = ind_min
                #print('Candidate onset: ' + str([ind_min, amp_min]))
                # Candidate peak
                #print('Looking for candidate peaks...')
                if dif_ma[i_start] > 0:
                    if dif_ma[i_stop] < 0:
                        _, _, aux_max, ind_max = seek_local(x_ma, int(i_start), int(i_stop))
                        #plt.scatter(ind_max, aux_max, marker = 'o', color = 'green')
                        if np.abs(ind_max - ind) <= 2:
                            amp_max = aux_max
                            aux_pks = ind_max
                #print('Candidate peak: ' + str([ind_max, amp_max]))
                # Verifies if the amplitude of the pulse is larger than 0.4 times the mean value:
                #print('Pulse amplitude: ' + str(amp_max - amp_min) + ', thresholds: ' +
                #      str([0.4*max_min_avg, 2*max_min_avg]))
                if (amp_max - amp_min) > 0.4*max_min_avg:
                    #print('Expected amplitude of pulse')
                    # Verifies if the amplitude of the pulse is lower than 2 times the mean value:
                    if (amp_max - amp_min) < 2*max_min_avg:
                        #print('Expected duration of pulse')
                        if aux_pks > aux_ons:
                            #print('Refining onsets...')
                            # Refine onsets:
                            aux_min = x_ma[aux_ons]
                            temp_ons = aux_ons
                            for j in range(aux_pks, aux_ons + 1, -1):
                                if x_ma[j] < aux_min:
                                    aux_min = x_ma[j]
                                    temp_ons = j
                            amp_min = aux_min
                            aux_ons = temp_ons

                            # If there is at least one peak found before:
                            #print('Number of previous peaks: ' + str(pk_index + 1))
                            if pk_index >= 0:
                                #print('There were previous peaks')
                                #print('Duration of ons to peak interval: ' + str(aux_ons - pks[pk_index]) +
                                #     ', threshold: ' + str([3*close_win, step_win]))
                                # If the duration of the pulse is too short:
                                if (aux_ons - pks[pk_index]) < 3*close_win:
                                    #print('Too short interbeat interval')
                                    ind = i
                                    max_min_avg = 2.5*max_min_lt
                                    break
                                # If the time difference between consecutive peaks is longer:
                                if (aux_pks - pks[pk_index]) > step_win:
                                    #print('Too long interbeat interval')
                                    pk_index = pk_index - 1
                                    on_index = on_index - 1
                                    #if dn_index > 0:
                                    #    dn_index = dn_index - 1
                                # If there are still peaks, add the new peak:
                                if pk_index >= 0:
                                    #print('There are still previous peaks')
                                    pk_index = pk_index + 1
                                    on_index = on_index + 1
                                    pks = np.append(pks, aux_pks)
                                    ons = np.append(ons, aux_ons)
                                    #print('Peaks: ' + str(pks))
                                    #print('Onsets: ' + str(ons))

                                    tf = ons[pk_index] - ons[pk_index - 1]

                                    to = np.floor(fs/20)
                                    tff = np.floor(0.1*tf)
                                    if tff < to:
                                        to = tff
                                    to = pks[pk_index - 1] + to

                                    te = np.floor(fs/20)
                                    tff = np.floor(0.5*tf)
                                    if tff < te:
                                        te = tff
                                    te = pks[pk_index - 1] + te

                                    #tff = seek_dicrotic(dif_ma[to:te])
                                    #if tff == 0:
                                    #    tff = te - pks[pk_index - 1]
                                    #    tff = np.floor(tff/3)
                                    #dn_index = dn_index + 1
                                    #dic[dn_index] = to + tff

                                    ind = ind + close_win
                                    break
                            # If it is the first peak:
                            if pk_index < 0:
                                #print('There were no previous peaks')
                                pk_index = pk_index + 1
                                on_index = on_index + 1
                                pks = np.append(pks, aux_pks)
                                ons = np.append(ons, aux_ons)
                                #print('Peaks: ' + str(pks))
                                #print('Onsets: ' + str(ons))
                                ind = ind + close_win
                                break

            ind = ind + 1
        i = int(ind + 1)

    if len(pks) == 0:
        return -1
    else:
        x_len = len(pks)
        temp_p = np.empty(0)
        for i in range(x_len):
            temp_p = np.append(temp_p, pks[i] - od)
        ttk = temp_p[0]
        if ttk < 0:
            temp_p[0] = 0
        pks = temp_p

        x_len = len(ons)
        temp_o = np.empty(0)
        for i in range(x_len):
            temp_o = np.append(temp_o, ons[i] - od)
        ttk = temp_o[0]
        if ttk < 0:
            temp_o[0] = 0
        ons = temp_o

    pks = pks + 5
    ibis = pks.astype(int)

    return ibis



def fiducial_points2(x,pks,fs,vis):
    """
    Description: Pulse detection and correction from pulsatile signals
    Inputs:  x, array with pulsatile signal [user defined units]
             pks, array with the position of the peaks [number of samples]
             fs, sampling rate of signal [Hz]
             vis, visualisation option [True, False]
    Outputs: fidp, dictionary with the positions of several fiducial points for the cardiac cycles [number of samples]

    Fiducial points:  1: Systolic peak (pks)
                      2: Onset, as the minimum before the systolic peak (ons)
                      3: Onset, using the tangent intersection method (ti)
                      4: Diastolic peak (dpk)
                      5: Maximum slope (m1d)
                      6: a point from second derivative PPG (a2d)
                      7: b point from second derivative PPG (b2d)
                      8: c point from second derivative PPG (c2d)
                      9: d point from second derivative PPG (d2d)
                      10: e point from second derivative PPG (e2d)
                      11: p1 from the third derivative PPG (p1)
                      12: p2 from the third derivative PPG (p2)

    Libraries: NumPy (as np), SciPy (Signal, as sp), Matplotlib (PyPlot, as plt)

    Version: 1.0 - June 2022

    Developed by: Elisa Mejía-Mejía
                   City, University of London

    Edited by: Peter Charlton (see "Added by PC")

    """
    # First, second and third derivatives
    d1x = sp.savgol_filter(x, 9, 5, deriv = 1)
    d2x = sp.savgol_filter(x, 9, 5, deriv = 2)
    d3x = sp.savgol_filter(x, 9, 5, deriv = 3)

    # Search in time series: Onsets between consecutive peaks
    ons = np.empty(0)
    for i in range(len(pks) - 1):
        start = pks[i]
        stop = pks[i + 1]
        ibi = x[start:stop]
        #plt.figure()
        #plt.plot(ibi, color = 'black')
        aux_ons, = np.where(ibi == np.min(ibi))
        ind_ons = aux_ons.astype(int)
        ons = np.append(ons, ind_ons + start)
        #plt.plot(ind_ons, ibi[ind_ons], marker = 'o', color = 'red')
    ons = ons.astype(int)

    # Search in time series: Diastolic peak and dicrotic notch between consecutive onsets
    dia = np.empty(0)
    dic = np.empty(0)
    for i in range(len(ons) - 1):
        start = ons[i]
        stop = ons[i + 1]
        ind_pks, = np.intersect1d(np.where(pks < stop), np.where(pks > start))
        ind_pks = pks[ind_pks]
        ibi_portion = x[ind_pks:stop]
        ibi_2d_portion = d2x[ind_pks:stop]
        aux_dic, _ = sp.find_peaks(ibi_2d_portion)
        aux_dic = aux_dic.astype(int)
        aux_dia, _ = sp.find_peaks(-ibi_2d_portion)
        aux_dia = aux_dia.astype(int)
        if len(aux_dic) != 0:
            ind_max, = np.where(ibi_2d_portion[aux_dic] == np.max(ibi_2d_portion[aux_dic]))
            aux_dic_max = aux_dic[ind_max]
            if len(aux_dia) != 0:
                nearest = aux_dia - aux_dic_max
                aux_dic = aux_dic_max
                dic = np.append(dic, (aux_dic + ind_pks).astype(int))
                #plt.scatter(aux_dic, ibi_portion[aux_dic]/np.max(ibi_portion), marker = 'o')
                ind_dia, = np.where(nearest > 0)
                aux_dia = aux_dia[ind_dia]
                nearest = nearest[ind_dia]
                if len(nearest) != 0:
                    ind_nearest, = np.where(nearest == np.min(nearest))
                    aux_dia = aux_dia[ind_nearest]
                    dia = np.append(dia, (aux_dia + ind_pks).astype(int))
                    #plt.scatter(aux_dia, ibi_portion[aux_dia]/np.max(ibi_portion), marker = 'o')
                    #break
            else:
                dic = np.append(dic, (aux_dic_max + ind_pks).astype(int))
                #plt.scatter(aux_dia, ibi_portion[aux_dia]/np.max(ibi_portion), marker = 'o')
    dia = dia.astype(int)
    dic = dic.astype(int)

    # Search in D1: Maximum slope point
    m1d = np.empty(0)
    for i in range(len(ons) - 1):
        start = ons[i]
        stop = ons[i + 1]
        ind_pks, = np.intersect1d(np.where(pks < stop), np.where(pks > start))
        ind_pks = pks[ind_pks]
        ibi_portion = x[start:ind_pks]
        ibi_1d_portion = d1x[start:ind_pks]

        aux_m1d, _ = sp.find_peaks(ibi_1d_portion)
        aux_m1d = aux_m1d.astype(int)
        if len(aux_m1d) != 0:
            ind_max, = np.where(ibi_1d_portion[aux_m1d] == np.max(ibi_1d_portion[aux_m1d]))
            aux_m1d_max = aux_m1d[ind_max]
            if len(aux_m1d_max) > 1:
                aux_m1d_max = aux_m1d_max[0]
            m1d = np.append(m1d, (aux_m1d_max + start).astype(int))

    m1d = m1d.astype(int)


    # Search in time series: Tangent intersection points
    tip = np.empty(0)
    for i in range(len(ons) - 1):
        start = ons[i]
        stop = ons[i + 1]
        ibi_portion = x[start:stop]
        ibi_1d_portion = d1x[start:stop]
        ind_m1d, = np.intersect1d(np.where(m1d < stop), np.where(m1d > start))
        ind_m1d = m1d[ind_m1d] - start

        aux_tip = np.round(((ibi_portion[0] - ibi_portion[ind_m1d])/ibi_1d_portion[ind_m1d]) + ind_m1d)
        aux_tip = aux_tip.astype(int)
        tip = np.append(tip, (aux_tip + start).astype(int))

    tip = tip.astype(int)


    # Search in D2: A, B, C, D and E points
    a2d = np.empty(0)
    b2d = np.empty(0)
    c2d = np.empty(0)
    d2d = np.empty(0)
    e2d = np.empty(0)
    for i in range(len(ons) - 1):
        start = ons[i]
        stop = ons[i + 1]
        ibi_portion = x[start:stop]
        ibi_1d_portion = d1x[start:stop]
        ibi_2d_portion = d2x[start:stop]
        ind_m1d = np.intersect1d(np.where(m1d > start),np.where(m1d < stop))
        ind_m1d = m1d[ind_m1d]

        aux_m2d_pks, _ = sp.find_peaks(ibi_2d_portion)
        aux_m2d_ons, _ = sp.find_peaks(-ibi_2d_portion)
        # a point:
        ind_a, = np.where(ibi_2d_portion[aux_m2d_pks] == np.max(ibi_2d_portion[aux_m2d_pks]))
        ind_a = aux_m2d_pks[ind_a]
        if (ind_a < ind_m1d):
            a2d = np.append(a2d, ind_a + start)

            # b point:
            ind_b = np.where(ibi_2d_portion[aux_m2d_ons] == np.min(ibi_2d_portion[aux_m2d_ons]))
            ind_b = aux_m2d_ons[ind_b]
            if (ind_b > ind_a) and (ind_b < len(ibi_2d_portion)):
                b2d = np.append(b2d, ind_b + start)

        # e point:
        ind_e, = np.where(aux_m2d_pks > ind_m1d - start)
        aux_m2d_pks = aux_m2d_pks[ind_e]
        ind_e, = np.where(aux_m2d_pks < 0.6*len(ibi_2d_portion))
        ind_e = aux_m2d_pks[ind_e]
        if len(ind_e) >= 1:
            if len(ind_e) >= 2:
                ind_e = ind_e[1]
            e2d = np.append(e2d, ind_e + start)

            # c point:
            ind_c, = np.where(aux_m2d_pks < ind_e)
            if len(ind_c) != 0:
                ind_c_aux = aux_m2d_pks[ind_c]
                ind_c, = np.where(ibi_2d_portion[ind_c_aux] == np.max(ibi_2d_portion[ind_c_aux]))
                ind_c = ind_c_aux[ind_c]
                if len(ind_c) != 0:
                    c2d = np.append(c2d, ind_c + start)

            else:
                aux_m1d_ons, _ = sp.find_peaks(-ibi_1d_portion)
                ind_c, = np.where(aux_m1d_ons < ind_e)
                ind_c_aux = aux_m1d_ons[ind_c]
                if len(ind_c) != 0:
                    ind_c, = np.where(ind_c_aux > ind_b)
                    ind_c = ind_c_aux[ind_c]
                    if len(ind_c) > 1:
                        ind_c = ind_c[0]
                    c2d = np.append(c2d, ind_c + start)

            # d point:
            if len(ind_c) != 0:
                ind_d = np.intersect1d(np.where(aux_m2d_ons < ind_e), np.where(aux_m2d_ons > ind_c))
                if len(ind_d) != 0:
                    ind_d_aux = aux_m2d_ons[ind_d]
                    ind_d, = np.where(ibi_2d_portion[ind_d_aux] == np.min(ibi_2d_portion[ind_d_aux]))
                    ind_d = ind_d_aux[ind_d]
                    if len(ind_d) != 0:
                        d2d = np.append(d2d, ind_d + start)

                else:
                    ind_d = ind_c
                    d2d = np.append(d2d, ind_d + start)

    a2d = a2d.astype(int)
    b2d = b2d.astype(int)
    c2d = c2d.astype(int)
    d2d = d2d.astype(int)
    e2d = e2d.astype(int)


    # Search in D3: P1 and P2 points
    p1p = np.empty(0)
    p2p = np.empty(0)
    for i in range(len(ons) - 1):
        start = ons[i]
        stop = ons[i + 1]
        ibi_portion = x[start:stop]
        ibi_1d_portion = d1x[start:stop]
        ibi_2d_portion = d2x[start:stop]
        ibi_3d_portion = d3x[start:stop]
        ind_b = np.intersect1d(np.where(b2d > start),np.where(b2d < stop))
        ind_b = b2d[ind_b]
        ind_c = np.intersect1d(np.where(c2d > start),np.where(c2d < stop))
        ind_c = c2d[ind_c]
        ind_d = np.intersect1d(np.where(d2d > start),np.where(d2d < stop))
        ind_d = d2d[ind_d]
        ind_dic = np.intersect1d(np.where(dic > start),np.where(dic < stop))
        ind_dic = dic[ind_dic]

        aux_p3d_pks, _ = sp.find_peaks(ibi_3d_portion)
        aux_p3d_ons, _ = sp.find_peaks(-ibi_3d_portion)
        # P1:
        if (len(aux_p3d_pks) != 0 and len(ind_b) != 0):
            ind_p1, = np.where(aux_p3d_pks > ind_b - start)
            if len(ind_p1) != 0:
                ind_p1 = aux_p3d_pks[ind_p1[0]]
                p1p = np.append(p1p, ind_p1 + start)

        # P2:
        if (len(aux_p3d_ons) != 0 and len(ind_c) != 0 and len(ind_d) != 0):
            if ind_c == ind_d:
                ind_p2, = np.where(aux_p3d_ons > ind_d - start)
                ind_p2 = aux_p3d_ons[ind_p2[0]]
            else:
                ind_p2, = np.where(aux_p3d_ons < ind_d - start)
                ind_p2 = aux_p3d_ons[ind_p2[-1]]
            if len(ind_dic) != 0:
                aux_x_pks, _ = sp.find_peaks(ibi_portion)
                if ind_p2 > ind_dic - start:
                    ind_between = np.intersect1d(np.where(aux_x_pks < ind_p2), np.where(aux_x_pks > ind_dic - start))
                else:
                    ind_between = np.intersect1d(np.where(aux_x_pks > ind_p2), np.where(aux_x_pks < ind_dic - start))
                if len(ind_between) != 0:
                    ind_p2 = aux_x_pks[ind_between[0]]
            p2p = np.append(p2p, ind_p2 + start)

    p1p = p1p.astype(int)
    p2p = p2p.astype(int)


    # Added by PC: Magnitudes of second derivative points
    bmag2d = np.zeros(len(b2d))
    cmag2d = np.zeros(len(b2d))
    dmag2d = np.zeros(len(b2d))
    emag2d = np.zeros(len(b2d))
    for beat_no in range(0,len(d2d)-3):
        bmag2d[beat_no] = d2x[b2d[beat_no]]/d2x[a2d[beat_no]]
        cmag2d[beat_no] = d2x[c2d[beat_no]]/d2x[a2d[beat_no]]
        dmag2d[beat_no] = d2x[d2d[beat_no]]/d2x[a2d[beat_no]]
        emag2d[beat_no] = d2x[e2d[beat_no]]/d2x[a2d[beat_no]]

        # Added by PC: Refine the list of fiducial points to only include those corresponding to beats for which a full set of points is available
    off = ons[1:]
    ons = ons[:-1]
    if pks[0] < ons[0]:
        pks = pks[1:]
    if pks[-1] > off[-1]:
        pks = pks[:-1]

    # Visualise results
    if vis == True:
        fig, (ax1) = plt.subplots(1, 1, sharex = True, sharey = False, figsize=(10,5))
        fig.suptitle('Fiducial points')

        ax1.plot(x, color = 'black')
        #print([x[i] for i in pks])  #[ppg_filt[patient_num][i] for i in ppg_beats[patient_num]]
        ax1.scatter(pks, [x[i] for i in pks], color = 'orange', label = 'pks')
        ax1.scatter(ons, [x[i] for i in ons], color = 'green', label = 'ons')
        ax1.scatter(off, [x[i] for i in off], marker = '*', color = 'green', label = 'off')
        ax1.scatter(dia, [x[i] for i in dia], color = 'yellow', label = 'dia')
        ax1.scatter(dic, [x[i] for i in dic], color = 'blue', label = 'dic')
        ax1.scatter(tip, [x[i] for i in tip], color = 'purple', label = 'dic')
        ax1.legend()
        ax1.set_ylabel('x')

    # Creation of dictionary
    fidp = {'pks': pks.astype(int),
            'ons': ons.astype(int),
            'off': off.astype(int),  # Added by PC
            'tip': tip.astype(int),
            'dia': dia.astype(int),
            'dic': dic.astype(int),
            'm1d': m1d.astype(int),
            'a2d': a2d.astype(int),
            'b2d': b2d.astype(int),
            'c2d': c2d.astype(int),
            'd2d': d2d.astype(int),
            'e2d': e2d.astype(int),
            'bmag2d': bmag2d,
            'cmag2d': cmag2d,
            'dmag2d': dmag2d,
            'emag2d': emag2d,
            'p1p': p1p.astype(int),
            'p2p': p2p.astype(int)
            }
    return fidp



# QRS detection function
def qrs_detect(ecg,fs,w):
    """

    Description: QRS peak detection and correction
    Inputs:  ecg, array with ECG signal [user defined units]
             fs, sampling rate of signal [Hz]
             w, window length for analysis [s]
    Outputs: qrs, positions of R peaks [number of samples]
             n_int, number of intervals of length w in the signal

    Libraries: NumPy (as np), SciPy (Signal, as sp), Matplotlib (PyPlot, as plt)

    Version: 1.0 - June 2022

    Developed by: Elisa Mejía-Mejía
                   City, University of London

    """

    # Normalization of ECG
    ecg_n = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg))
    ecg_n = ecg_n - np.mean(ecg_n)

    # Peak detection in windows of length w
    n_int = np.floor(len(ecg)/(w*fs))
    for i in range(int(n_int)):
        start = i*fs*w
        stop = (i + 1)*fs*w - 1
        #print('Start: ' + str(start) + ', stop: ' + str(stop))
        aux = ecg_n[range(start,stop)]
        locs, amps, nlocs = rpeakdetect(aux,fs,0.2)
        locs = locs + start
        if i == 0:
            qrs = locs
        else:
            qrs = np.append(qrs,locs)
    if n_int*fs*w != len(ecg_n):
        start = stop + 1
        stop = len(ecg_n)
        aux = ecg_n[range(start,stop)]
        if len(aux) > 20:
            locs, amps, nlocs = rpeakdetect(aux,fs,0.2)
            locs = locs + start
            qrs = np.append(qrs,locs)

    return qrs, n_int




# R peak detect
def rpeakdetect(ecg, fs, th):
    """

    Description: QRS peak detection based on rpeakdetect2.m by G. Clifford
                 A batch QRS detector based upon that of Pan, Hamilton and Tompkins:
                 J. Pan & W. Tompkins. A real-time QRS detection algorithm. IEEE Trans Biomed Eng, vol. BME-32 NO. 3. 1985.
                 P. Hamilton & W. Tompkins. Quantitative Investigation of QRS Detection Rules Using the MIT/BIH Arrythmia
                 Database. IEEE Trans Biomed Eng, vol. BME-33, NO. 12.1986.
    Inputs:  ecg, array with ECG signal [user defined units]
             fs, sampling rate of signal [Hz]
             th, threshold for peaks in integrated waveform - Default: 0.2
    Outputs: locs, positions of R peaks [number of samples]
             amps, amplitudes of R peaks [user defined units]
             nlocs, number of R peaks found

    Libraries: NumPy (as np), SciPy (Signal, as sp), Matplotlib (PyPlot, as plt)

    Version: 1.0 - June 2022

    Developed by: Elisa Mejía-Mejía
                  City, University of London

    """

    # Create time array
    t = np.divide(range(0,len(ecg) - 1), fs)

    # Preprocessing:
    # (1) Filter data
    sos = sp.butter(6, 40, btype = 'low', analog = False, output = 'sos', fs = fs)
    ecg_f = sp.sosfiltfilt(sos, ecg)
    # (2) Differentiate ECG
    ecg_d = np.diff(ecg_f)
    # (3) Square ECG
    ecg_s = ecg_d*ecg_d
    # (4) Integrate data
    if fs >= 256:
        d = np.ones(int(np.round(7*fs/256)))
    else:
        d = np.ones(21)
    ecg_fir = sp.lfilter(d, 1, ecg_s)
    ecg_med = sp.medfilt(ecg_fir, kernel_size = 9)

    # Remove filter delay:
    delay = np.ceil(len(d)/2)
    ecg_med = ecg_med[int(delay):len(ecg_med)]

    # Find peaks:
    # (1) Find highest bumps in data
    start_int = round((len(ecg) - 1)/4)
    stop_int = round(3*(len(ecg) - 1)/4)
    max_h = np.max(ecg_med[start_int:stop_int])
    # (2) Determine left and right boundaries
    ecg_med = np.insert(ecg_med,0,0)
    ecg_med = np.append(ecg_med,0)
    n_left = 0
    n_right = 0
    for i in range(len(ecg_med)):
        if i > 0 and i < len(ecg_med):
            if ecg_med[i] > (th*max_h) and ecg_med[i - 1] < (th*max_h): # left boundary
                if n_left == 0:
                    left_bound = i - 1
                    n_left = 1
                else:
                    left_bound = np.append(left_bound,i - 1)
            if ecg_med[i] > (th*max_h) and ecg_med[i + 1] < (th*max_h): # right boundary
                if n_right == 0:
                    right_bound = i - 1
                    n_right = 1
                else:
                    right_bound = np.append(right_bound,i - 1)
                    #print(left_bound)
    #print(right_bound)
    # (4) Look through all possibilities
    if left_bound[0] > right_bound[0]:
        right_bound = np.delete(right_bound,0)
    if left_bound[-1] > right_bound[-1]:
        left_bound = np.delete(left_bound,0)
    nlocs = 0
    for i in range(len(left_bound)):
        #print(i)
        lb = left_bound[i]
        rb, = np.where(np.array(right_bound) > lb)
        if len(rb) != 0:
            rb = right_bound[rb[0]]
            #print(lb, rb)

            amp = np.max(ecg[lb:rb])
            pos, = np.where(np.array(ecg[lb:rb]) == np.amax(ecg[lb:rb]))
            pos = pos[0]
            pos = int(pos + lb)
            #print(pos, amp)
            if nlocs == 0:
                locs = pos
                amps = ecg[pos]
            else:
                locs = np.append(locs, pos)
                amps = np.append(amps, ecg[pos])
            nlocs = nlocs + 1


    return locs, amps, nlocs,