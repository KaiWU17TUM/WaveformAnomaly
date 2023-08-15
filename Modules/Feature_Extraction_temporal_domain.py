import numpy as np
from numpy.typing import ArrayLike
import neurokit2 as nk
import math


def create_timestamp_signal(resolution: str, length: float, start: float, rate: float) -> ArrayLike:
    """Generates a timestamp array.

    Args:
        resolution (str): Timestamp resolution. It can be 'ns', 'ms', 's' or 'min'.
        length (float): Length of timestamp array to be generated.
        start (float): Starting time.
        rate (float): Rate of increment.

    Raises:
        ValueError: If starting time is less then zero.
        ValueError: If resolution is undefined.

    Returns:
        ArrayLike: Timestamp array.
    """

    if start < 0:
        raise ValueError("Timestamp start must be greater than 0")

    if resolution == "ns":
        timestamp_factor = 1 / 1e-9
    elif resolution == "ms":
        timestamp_factor = 1 / 0.001
    elif resolution == "s":
        timestamp_factor = 1
    elif resolution == "min":
        timestamp_factor = 60
    else:
        raise ValueError('resolution must be "ns","ms","s","min"')

    timestamp = (np.arange(length) / rate) * timestamp_factor
    timestamp = timestamp + start

    return timestamp


def check_timestamp(timestamp, timestamp_resolution):

    possible_timestamp_resolution = ["ns", "ms", "s", "min"]

    if timestamp_resolution in possible_timestamp_resolution:
        pass
    else:
        raise ValueError('timestamp_resolution must be "ns","ms","s","min"')

    if np.any(np.diff(timestamp) < 0):
        raise ValueError("Timestamp must be monotonic")

    return True


import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from plotly_resampler import register_plotly_resampler

def create_signal_plot_matplotlib(
        ax: plt.Axes,
        signal: ArrayLike = None,
        x_values=None,
        show_peaks: bool = False,
        peaks: dict = None,
        plot_title: str = "Signal Plot",
        signal_name: str = "Signal",
        x_label: str = "Sample",
):
    """Generates plots for given signals using Matplotlib.

    Args:
        ax (plt.Axes): Axes to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (_type_, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
    """
    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    # Check if there is existing legend
    if isinstance(ax.get_legend(), type(None)):
        legend = []
    else:
        legend = [x.get_text() for x in ax.get_legend().texts]

    ax.plot(x_values, signal)
    legend.append(signal_name)

    if show_peaks:
        for peak_type, peak_loc in peaks.items():
            peak_amp = [signal[i] for i in peak_loc]
            ax.scatter([x_values[i] for i in peak_loc], peak_amp, label=peak_type)
            legend.append(signal_name + " " + peak_type)

    ax.set_title(plot_title)
    ax.set_xlim([0, max(x_values)])
    # ax.set_xlabel(x_label)
    # ax.set_ylabel('Amplitude')

    ax.legend(legend, loc="center left", bbox_to_anchor=(1.0, 0.5))


def create_signal_plot_plotly(
        fig: go.Figure,
        signal: ArrayLike = None,
        x_values: ArrayLike = None,
        show_peaks: bool = False,
        peaks: dict = None,
        plot_title: str = "Signal Plot",
        signal_name: str = "Signal",
        x_label: str = "Sample",
        width: float = 1050,
        height: float = 600,
        location: tuple = None,
):
    """Generates plots for given signals using Plotly.

    Args:
        fig (go.Figure): Figure to plot signal.
        signal (ArrayLike, optional): Array of y-axis values. Defaults to None.
        x_values (ArrayLike, optional): Array of x-axis values. Defaults to None.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to False.
        peaks (dict, optional): Dictionary of peaks to be plotted. Defaults to None.
        plot_title (str, optional): Plot title. Defaults to "Signal Plot".
        signal_name (str, optional): Name of signal to be plotted. Defaults to "Signal".
        x_label (str, optional): Label of x-axis. Defaults to 'Sample'.
        width (float, optional): Figure width. Defaults to 1050.
        height (float, optional): Figure height. Defaults to 600.
        location (tuple, optional): Subplot location. Defaults to None.

    Raises:
        ValueError: If location is not provided.
    """
    # adjust it
    limit = 200000

    if len(signal) > limit:
        Warning("Signal is too large and will be resampled. Consider using create_signal_plot instead")
        register_plotly_resampler(mode="auto")

    if x_values is None:
        x_values = np.linspace(0, len(signal), len(signal))

    if location is None:
        raise ValueError("Location must be specified")

    fig.append_trace(go.Scatter(x=x_values, y=signal, name=signal_name), row=location[0], col=location[1])

    if show_peaks:

        for peak_type, peak_loc in peaks.items():
            peak_amp = signal[peak_loc]
            fig.append_trace(
                go.Scatter(x=x_values[peak_loc], y=peak_amp, name=signal_name + " " + peak_type, mode="markers"),
                row=location[0],
                col=location[1],
            )

    fig.update_layout({"xaxis": {"range": [0, x_values.max()]}}, title=plot_title, width=width, height=height)
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=1.2,
        showarrow=False,
        text=plot_title,
        row=location[0],
        col=location[1],
    )


import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


def plot_ecg(
        signals: dict,
        peaks: dict = None,
        sampling_rate: float = None,
        timestamps: ArrayLike = None,
        timestamp_resolution: str = None,
        method: str = "matplotlib",
        show_peaks: bool = True,
        figsize: tuple = (18.5, 10.5),
        width: float = 800,
        height: float = 440,
):
    """Generates plots for ECG signal.

    Args:
        signals (dict): The dictionary of signals to be plotted.
        peaks (dict, optional): The dictionary of peaks to be plotted. Defaults to None.
        sampling_rate (float, optional): Sampling rate of the signal. Defaults to None.
        timestamps (ArrayLike, optional): Timestamp array. Defaults to None.
        timestamp_resolution (str, optional): Timestamp resolution. Defaults to None.
        method (str, optional): Package to generate plots. Defaults to 'matplotlib'.
        show_peaks (bool, optional): If True, peaks are plotted. Defaults to True.
        figsize (tuple, optional): Figure size for matplotlib. Defaults to (18.5, 10.5).
        width (float, optional): Figure width for Plotly. Defaults to 800.
        height (float, optional): Figure height for Plotly. Defaults to 440.

    Raises:
        ValueError: If timestamps is not None and timestamp resolution is not provided.
        ValueError: If timestamps array and ECG signal have different lengths.
        ValueError: If method is not 'matplotlib' or 'plotly'.
    """
    ecg_raw = signals.get("Raw")

    if timestamps is not None:
        if len(timestamps) != len(ecg_raw):
            raise ValueError("Timestamps and ECG signal must have the same length!")

        if timestamp_resolution is None:
            raise ValueError("Timestamp resolution must be provided if timestamps are provided!")
        else:
            timestamp_resolution = timestamp_resolution

        x_values = timestamps
        x_label = "Time (" + timestamp_resolution + ")"

    else:
        if sampling_rate is not None:
            if timestamp_resolution is None:
                timestamp_resolution = "s"

            x_values = create_timestamp_signal(
                resolution=timestamp_resolution, length=len(ecg_raw), rate=sampling_rate, start=0
            )
            x_label = "Time (" + timestamp_resolution + ")"

        else:
            x_values = np.linspace(0, len(ecg_raw), len(ecg_raw))
            x_label = "Sample"

    if peaks is None:
        if show_peaks:
            raise ValueError("Peaks must be specified if show_peaks is True.")
        else:
            peaks = {}

    if method == "matplotlib":
        _plot_ecg_matplotlib(
            signals=signals, peaks=peaks, x_values=x_values, x_label=x_label, figsize=figsize, show_peaks=show_peaks
        )
    elif method == "plotly":
        _plot_ecg_plotly(
            signals=signals,
            peaks=peaks,
            x_values=x_values,
            x_label=x_label,
            width=width,
            height=height,
            show_peaks=show_peaks,
        )
    else:
        raise ValueError("Undefined method.")


def _plot_ecg_matplotlib(
        signals: dict,
        peaks: dict = None,
        x_values: ArrayLike = None,
        x_label: str = "Sample",
        figsize=(18.5, 10.5),
        show_peaks=True,
):
    """Generates plots for ECG signal using Matplotlib."""
    # Create figure
    fig, axs = plt.subplots(figsize=figsize)

    # Plot raw ECG, filtered ECG and peaks
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_matplotlib(
            ax=axs,
            signal=signal,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=" ",
            signal_name=signal_name + " ECG",
            x_label=x_label,
        )

    fig.supxlabel(x_label)
    fig.supylabel("Amplitude")
    plt.title("ECG Signal")

    fig.tight_layout()
    plt.show()


def _plot_ecg_plotly(
        signals: dict,
        peaks: dict = None,
        x_values: ArrayLike = None,
        x_label: str = "Sample",
        width=800,
        height=440,
        show_peaks=True,
):
    """Generates plots for ECG signal using Plotly."""
    # Create figure
    fig = make_subplots(rows=1, cols=1)

    # Plot raw ECG, filtered ECG and peaks
    for signal_name, signal in signals.items():

        if signal_name not in peaks.keys():
            peaks[signal_name] = {}

        create_signal_plot_plotly(
            fig,
            signal=signal,
            x_values=x_values,
            show_peaks=show_peaks,
            peaks=peaks[signal_name],
            plot_title=" ",
            signal_name=signal_name + " ECG",
            width=width,
            height=height,
            location=(1, 1),
        )

    fig.update_layout(
        {"title": {"text": "ECG Signal", "x": 0.45, "y": 0.9}}, xaxis_title=x_label, yaxis_title="Amplitude"
    )
    fig.show()


from ecgdetectors import Detectors
from numpy.typing import ArrayLike


def ecg_detectpeaks(sig: ArrayLike, sampling_rate: float, method: str = "pantompkins") -> ArrayLike:
    """Detects R peaks from ECG signal.
    Uses py-ecg-detectors package(https://github.com/berndporr/py-ecg-detectors/).

    Args:
        sig (ArrayLike): ECG signal.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        method (str, optional): Peak detection method. Should be 'pantompkins', 'hamilton' or 'elgendi'. Defaults to 'pantompkins'.
        'pantompkins': "Pan, J. & Tompkins, W. J.,(1985). 'A real-time QRS detection algorithm'. IEEE transactions
                        on biomedical engineering, (3), 230-236."
        'hamilton': "Hamilton, P.S. (2002), 'Open Source ECG Analysis Software Documentation', E.P.Limited."
        'elgendi': "Elgendi, M. & Jonkman, M. & De Boer, F. (2010). 'Frequency Bands Effects on QRS Detection',
                    The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If method is not 'pantompkins', 'hamilton' or 'elgendi'.

    Returns:
        ArrayLike: R-peak locations
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()
    detectors = Detectors(sampling_rate)

    if method == "pantompkins":
        r_peaks = detectors.pan_tompkins_detector(sig)
    elif method == "hamilton":
        r_peaks = detectors.hamilton_detector(sig)
    elif method == "elgendi":
        r_peaks = detectors.two_average_detector(sig)
    else:
        raise ValueError(f"Undefined method: {method}")

    return r_peaks

import numpy as np
from numpy.typing import ArrayLike

# Morphological features from R-peak locations
FEATURES_RPEAKS = {
    "a_R": lambda sig, _0, peaks_locs, beatno: sig[peaks_locs[beatno]],
    "RR0": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1),
    "RR1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR2": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1),
    "RRm": lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno),
    "RR_0_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, -1)
                                                            / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR_2_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_RR_interval(peaks_locs, sampling_rate, beatno, 1)
                                                            / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
    "RR_m_1": lambda _0, sampling_rate, peaks_locs, beatno: _get_mean_RR(peaks_locs, sampling_rate, beatno)
                                                            / _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
}
# Morphological features from all fiducials
FEATURES_WAVES = {
    "t_PR": lambda sig, sampling_rate, locs_P, _0, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_R, sampling_rate, beatno, False
    ),
    "t_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_Q, locs_R, sampling_rate, beatno, False
    ),
    "t_RS": lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(
        sig, locs_S, locs_R, sampling_rate, beatno, False
    ),
    "t_RT": lambda sig, sampling_rate, _0, _1, locs_R, _2, locs_T, beatno: _get_diff(
        sig, locs_T, locs_R, sampling_rate, beatno, False
    ),
    "t_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, False
    ),
    "t_PS": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_P, locs_S, sampling_rate, beatno, False
    ),
    "t_PT": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, False
    ),
    "t_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_Q, locs_S, sampling_rate, beatno, False
    ),
    "t_QT": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, False
    ),
    "t_ST": lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, False
    ),
    "t_PT_QS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, False
    )
                                                                                      / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
    "t_QT_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, False
    )
                                                                                  / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, False),
    "a_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, _2, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    ),
    "a_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, _1, _2, beatno: _get_diff(
        sig, locs_Q, locs_R, sampling_rate, beatno, True
    ),
    "a_RS": lambda sig, sampling_rate, _0, _1, locs_R, locs_S, _2, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    ),
    "a_ST": lambda sig, sampling_rate, _0, _1, _2, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    ),
    "a_PS": lambda sig, sampling_rate, locs_P, _0, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_P, locs_S, sampling_rate, beatno, True
    ),
    "a_PT": lambda sig, sampling_rate, locs_P, _0, _1, _2, locs_T, beatno: _get_diff(
        sig, locs_P, locs_T, sampling_rate, beatno, True
    ),
    "a_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, _2, beatno: _get_diff(
        sig, locs_Q, locs_S, sampling_rate, beatno, True
    ),
    "a_QT": lambda sig, sampling_rate, _0, locs_Q, _1, _2, locs_T, beatno: _get_diff(
        sig, locs_Q, locs_T, sampling_rate, beatno, True
    ),
    "a_ST_QS": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_RS_QR": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
    "a_PQ_QS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_PQ_QT": lambda sig, sampling_rate, locs_P, locs_Q, _0, _1, locs_T, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
    "a_PQ_PS": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_P, locs_S, sampling_rate, beatno, True),
    "a_PQ_QR": lambda sig, sampling_rate, locs_P, locs_Q, locs_R, _0, _1, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_R, sampling_rate, beatno, True),
    "a_PQ_RS": lambda sig, sampling_rate, locs_P, locs_Q, locs_R, locs_S, _0, beatno: _get_diff(
        sig, locs_P, locs_Q, sampling_rate, beatno, True
    )
                                                                                      / _get_diff(sig, locs_R, locs_S, sampling_rate, beatno, True),
    "a_RS_QS": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, _1, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_S, sampling_rate, beatno, True),
    "a_RS_QT": lambda sig, sampling_rate, _0, locs_Q, locs_R, locs_S, locs_T, beatno: _get_diff(
        sig, locs_R, locs_S, sampling_rate, beatno, True
    )
                                                                                      / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
    "a_ST_PQ": lambda sig, sampling_rate, locs_P, locs_Q, _0, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
                                                                                      / _get_diff(sig, locs_P, locs_Q, sampling_rate, beatno, True),
    "a_ST_QT": lambda sig, sampling_rate, _0, locs_Q, _1, locs_S, locs_T, beatno: _get_diff(
        sig, locs_S, locs_T, sampling_rate, beatno, True
    )
                                                                                  / _get_diff(sig, locs_Q, locs_T, sampling_rate, beatno, True),
}


def from_Rpeaks(
        sig: ArrayLike, peaks_locs: ArrayLike, sampling_rate: float, prefix: str = "ecg", average: bool = False
) -> dict:
    """Calculates R-peak-based ECG features and returns a dictionary of features for each heart beat.

        'a_R': Amplitude of R peak
        'RR0': Previous RR interval
        'RR1': Current RR interval
        'RR2': Subsequent RR interval
        'RRm': Mean of RR0, RR1 and RR2
        'RR_0_1': Ratio of RR0 to RR1
        'RR_2_1': Ratio of RR2 to RR1
        'RR_m_1': Ratio of RRm to RR1

    Args:
        sig (ArrayLike): ECG signal segment.
        peaks_locs (ArrayLike): ECG R-peak locations.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Returns:
        dict: Dictionary of ECG features.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    features_rpeaks = {}
    for m in range(1, len(peaks_locs) - 2):
        features = {}
        for key, func in FEATURES_RPEAKS.items():
            try:
                features["_".join([prefix, key])] = func(sig, sampling_rate, peaks_locs=peaks_locs, beatno=m)
            except:
                features["_".join([prefix, key])] = np.nan
        features_rpeaks[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_rpeaks.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.mean(features_[k])

        return features_avr

    else:
        return features_rpeaks


def from_waves(
        sig: ArrayLike,
        R_peaks: ArrayLike,
        fiducials: dict,
        sampling_rate: float,
        prefix: str = "ecg",
        average: bool = False,
) -> dict:
    """Calculates ECG features from the given fiducials and returns a dictionary of features.

        't_PR': Time between P and R peak locations
        't_QR': Time between Q and R peak locations
        't_RS': Time between R and S peak locations
        't_RT': Time between R and T peak locations
        't_PQ': Time between P and Q peak locations
        't_PS': Time between P and S peak locations
        't_PT': Time between P and T peak locations
        't_QS': Time between Q and S peak locations
        't_QT':Time between Q and T peak locations
        't_ST': Time between S and T peak locations
        't_PT_QS': Ratio of t_PT to t_QS
        't_QT_QS': Ratio of t_QT to t_QS
        'a_PQ': Difference of P wave and Q wave amplitudes
        'a_QR': Difference of Q wave and R wave amplitudes
        'a_RS': Difference of R wave and S wave amplitudes
        'a_ST': Difference of S wave and T wave amplitudes
        'a_PS': Difference of P wave and S wave amplitudes
        'a_PT': Difference of P wave and T wave amplitudes
        'a_QS': Difference of Q wave and S wave amplitudes
        'a_QT': Difference of Q wave and T wave amplitudes
        'a_ST_QS': Ratio of a_ST to a_QS
        'a_RS_QR': Ratio of a_RS to a_QR
        'a_PQ_QS': Ratio of a_PQ to a_QS
        'a_PQ_QT': Ratio of a_PQ to a_QT
        'a_PQ_PS': Ratio of a_PQ to a_PS
        'a_PQ_QR': Ratio of a_PQ to a_QR
        'a_PQ_RS': Ratio of a_PQ to a_RS
        'a_RS_QS': Ratio of a_RS to a_QS
        'a_RS_QT': Ratio of a_RS to a_QT
        'a_ST_PQ': Ratio of a_ST to a_PQ
        'a_ST_QT': Ratio of a_ST to a_QT

    Args:
        sig (ArrayLike): ECG signal segment.
        R_peaks (ArrayLike): ECG R-peak locations.
        fiducials (dict): Dictionary of fiducial locations (keys: "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks").
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        prefix (str, optional): Prefix for the feature. Defaults to 'ecg'.
        average (bool, optional): If True, averaged features are returned. Defaults to False.

    Raises:
        ValueError: If sampling rate is not greater than 0.

    Returns:
        dict: Dictionary of ECG features.
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    feature_list = FEATURES_WAVES.copy()

    fiducial_names = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks"]
    fiducials = {key: fiducials.get(key, []) for key in fiducial_names}

    P_peaks = fiducials["ECG_P_Peaks"]
    Q_peaks = fiducials["ECG_Q_Peaks"]
    S_peaks = fiducials["ECG_S_Peaks"]
    T_peaks = fiducials["ECG_T_Peaks"]

    if len(P_peaks) == 0:
        P_features = [
            "t_PR",
            "t_PQ",
            "t_PS",
            "t_PT",
            "t_PT_QS",
            "a_PQ",
            "a_PS",
            "a_PT",
            "a_PQ_QS",
            "a_PQ_QT",
            "a_PQ_PS",
            "a_PQ_QR",
            "a_PQ_RS",
            "a_ST_PQ",
        ]
        [feature_list.pop(key, None) for key in P_features]

    if len(Q_peaks) == 0:
        Q_features = [
            "t_QR",
            "t_PQ",
            "t_QS",
            "t_QT",
            "t_PT_QS",
            "t_QT_QS",
            "a_PQ",
            "a_QR",
            "a_QS",
            "a_QT",
            "a_ST_QS",
            "a_RS_QR",
            "a_PQ_QS",
            "a_PQ_QT",
            "a_PQ_PS",
            "a_PQ_QR",
            "a_PQ_RS",
            "a_RS_QS",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in Q_features]

    if len(S_peaks) == 0:
        S_features = [
            "t_SR",
            "t_PS",
            "t_QS",
            "t_ST",
            "t_PT_QS",
            "t_QT_QS",
            "a_RS",
            "a_ST",
            "a_PS",
            "a_QS",
            "a_ST_QS",
            "a_RS_QR",
            "a_PQ_QS",
            "a_PQ_PS",
            "a_PQ_RS",
            "a_RS_QS",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in S_features]

    if len(T_peaks) == 0:
        T_features = [
            "t_TR",
            "t_PT",
            "t_QT",
            "t_ST",
            "t_PT_QS",
            "t_QT_QS",
            "a_ST",
            "a_PT",
            "a_QT",
            "a_ST_QS",
            "a_PQ_QT",
            "a_RS_QT",
            "a_ST_PQ",
            "a_ST_QT",
        ]
        [feature_list.pop(key, None) for key in T_features]

    features_waves = {}
    for m in range(len(R_peaks)):
        features = {}
        for key, func in feature_list.items():
            try:
                features["_".join([prefix, key])] = func(
                    sig, sampling_rate, P_peaks, Q_peaks, R_peaks, S_peaks, T_peaks, beatno=m
                )
            except:
                features["_".join([prefix, key])] = np.nan
        features_waves[m] = features

    if average:
        features_avr = {}

        features_ = {}
        for subdict in features_waves.values():
            for key, value in subdict.items():
                if key not in features_:
                    features_[key] = [value]
                else:
                    features_[key].append(value)

        for k in features_.keys():
            features_avr[k] = np.mean(features_[k])

        return features_avr

    else:
        return features_waves


def _get_RR_interval(peaks_locs: ArrayLike, sampling_rate: float, beatno: int, interval: int = 0) -> float:

    rr_int = (peaks_locs[beatno + interval + 1] - peaks_locs[beatno + interval]) / sampling_rate

    return rr_int


def _get_mean_RR(peaks_locs: ArrayLike, sampling_rate: float, beatno: int) -> float:

    rr_m = np.mean(
        [
            _get_RR_interval(peaks_locs, sampling_rate, beatno, -1),
            _get_RR_interval(peaks_locs, sampling_rate, beatno, 0),
            _get_RR_interval(peaks_locs, sampling_rate, beatno, 1),
        ]
    )
    return rr_m


def _get_diff(
        sig: ArrayLike,
        loc_array1: ArrayLike,
        loc_array2: ArrayLike,
        sampling_rate: float,
        beatno: int,
        amplitude: bool = False,
) -> float:

    if amplitude:
        feature = sig[loc_array2[beatno]] - sig[loc_array1[beatno]]
    else:
        feature = abs((loc_array2[beatno] - loc_array1[beatno])) / sampling_rate

    return feature



def find_features(signal, sampling_rate):
    #Filter ECG signal by using predefined filters
    #signal = np.asarray(signal)

    #Detect peaks using 'pantompkins' method.
    locs_peaks=ecg_detectpeaks(signal,sampling_rate,'pantompkins')

    #Delineate ECG signal using 'neurokit2' package.
    _, fiducials = nk.ecg_delineate(ecg_cleaned=signal, rpeaks=locs_peaks, sampling_rate=sampling_rate, method='peak')

    p_peaks_locs = fiducials['ECG_P_Peaks']
    q_peaks_locs = fiducials['ECG_Q_Peaks']

    p_peaks_locs = [x for x in p_peaks_locs if (math.isnan(x) == False)]
    q_peaks_locs = [x for x in q_peaks_locs if (math.isnan(x) == False)]
    locs_peaks = [x for x in locs_peaks if (math.isnan(x) == False)]

    #Calculate features from R peaks
    features_rpeaks = from_Rpeaks(signal, locs_peaks, sampling_rate, average=True)
    #features_rpeaks = {k: 0 if math.isnan(v) else v for k, v in features_rpeaks.items() }

    #Calculate features from P, Q, R, S, T waves
    features_waves = from_waves(signal, locs_peaks, fiducials, sampling_rate, average = True)
    # for i in range(len(features_waves)):
    #     features_waves[i] = {k: 0 if math.isnan(v) else v for k, v in features_waves[i].items() }

    return signal, locs_peaks, p_peaks_locs, q_peaks_locs, features_rpeaks, features_waves


import neurokit2 as nk
from ecgdetectors import Detectors
from numpy.typing import ArrayLike


def ecg_detectpeaks(sig: ArrayLike, sampling_rate: float, method: str = "pantompkins") -> ArrayLike:
    """Detects R peaks from ECG signal.
    Uses py-ecg-detectors package(https://github.com/berndporr/py-ecg-detectors/).

    Args:
        sig (ArrayLike): ECG signal.
        sampling_rate (float): Sampling rate of the ECG signal (Hz).
        method (str, optional): Peak detection method. Should be 'pantompkins', 'hamilton' or 'elgendi'. Defaults to 'pantompkins'.
        'pantompkins': "Pan, J. & Tompkins, W. J.,(1985). 'A real-time QRS detection algorithm'. IEEE transactions
                        on biomedical engineering, (3), 230-236."
        'hamilton': "Hamilton, P.S. (2002), 'Open Source ECG Analysis Software Documentation', E.P.Limited."
        'elgendi': "Elgendi, M. & Jonkman, M. & De Boer, F. (2010). 'Frequency Bands Effects on QRS Detection',
                    The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.

    Raises:
        ValueError: If sampling rate is not greater than 0.
        ValueError: If method is not 'pantompkins', 'hamilton' or 'elgendi'.

    Returns:
        ArrayLike: R-peak locations
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    method = method.lower()
    detectors = Detectors(sampling_rate)

    if method == "pantompkins":
        r_peaks = detectors.pan_tompkins_detector(sig)
    elif method == "hamilton":
        r_peaks = detectors.hamilton_detector(sig)
    elif method == "elgendi":
        r_peaks = detectors.two_average_detector(sig)
    else:
        raise ValueError(f"Undefined method: {method}")

    return r_peaks


import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

# Constants to check for physiological viability and morphological features.
HR_MIN = 40
HR_MAX = 180
PP_MAX = 3
MAX_PP_RATIO = 2.2
MIN_SPD = 0.08
MAX_SPD = 0.49
SP_DP_RATIO = 1.1
MIN_PWD = 0.27
MAX_PWD = 2.4
MAX_VAR_DUR = 300
MAX_VAR_AMP = 400
CORR_TH = 0.9


def check_phys(peaks_locs: ArrayLike, sampling_rate: float) -> dict:
    """Checks for physiological viability.

    Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal.
            For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2

    Args:
        peaks_locs (ArrayLike): Array of peak locations.
        sampling_rate (float): Sampling rate of the input signal.

    Returns:
        dict: Dictionary of decisions.
    """
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than 0.")

    info = {}

    # Rule 1: Average HR should be between 40-180 bpm (up to 300 bpm in the case of exercise)
    intervals = np.diff(peaks_locs) / sampling_rate
    HR_mean = 60 / np.mean(intervals)

    if HR_mean < HR_MIN or HR_mean > HR_MAX:
        info["Rule 1"] = False
    else:
        info["Rule 1"] = True

    # Rule 2: Maximum P-P interval: 1.5 seconds. Allowing for a single missing beat, it is 3 seconds
    if np.size(np.where(intervals > PP_MAX)) > 0:
        info["Rule 2"] = False
    else:
        info["Rule 2"] = True

    # Rule 3: Maximum P-P interval / minimum P-P interval ratio: 10 of the signal length for a short signal.
    # For 10 seconds signal, it is 1.1; allowing for a single missing beat, it is 2.2
    if (intervals.max() / intervals.min()) > MAX_PP_RATIO:
        info["Rule 3"] = False
    else:
        info["Rule 3"] = True

    return info


def template_matching(sig: ArrayLike, peaks_locs: ArrayLike, corr_th: float = CORR_TH) -> Tuple[float, bool]:
    """Applies template matching method for signal quality assessment.

    Args:
        sig (ArrayLike): Signal to be analyzed.
        peaks_locs (ArrayLike): Peak locations (Systolic peaks for PPG signal, R peaks for ECG signal).
        corr_th (float, optional): Threshold for the correlation coefficient above which the signal is considered to be valid. Defaults to CORR_TH.

    Returns:
        Tuple[float,bool]: Correlation coefficient and the decision
    """
    if corr_th <= 0:
        raise ValueError("Threshold for the correlation coefficient must be greater than 0.")

    wl = np.median(np.diff(peaks_locs))
    waves = np.empty((0, 2 * math.floor(wl / 2) + 1))
    nofwaves = np.size(peaks_locs)

    for i in range((nofwaves)):
        wave_st = peaks_locs[i] - math.floor(wl / 2)
        wave_end = peaks_locs[i] + math.floor(wl / 2)
        wave = []

        if wave_st < 0:
            wave = sig[:wave_end]
            for _ in range(-wave_st + 1):
                wave = np.insert(wave, 0, wave[0])

        elif wave_end > len(sig) - 1:
            wave = sig[wave_st - 1 :]
            for _ in range(wave_end - len(sig)):
                wave = np.append(wave, wave[-1])

        else:
            wave = sig[wave_st : wave_end + 1]

        waves = np.vstack([waves, wave])

    sig_temp = np.mean(waves, axis=0)

    ps = np.array([])
    for j in range(np.size(peaks_locs)):
        p = np.corrcoef(waves[j], sig_temp, rowvar=True)
        ps = np.append(ps, p[0][1])

    if np.size(np.where(ps < corr_th)) > 0:
        result = False

    else:
        result = True

    return ps, result
