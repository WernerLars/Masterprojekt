import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord


def Filterbank(eeg_data, window_details, fs):
    # Filtering (adapted from FBCSP)
    f_trans = 2
    f_pass = np.arange(4, 40, 4)
    f_width = 4
    gpass = 3
    gstop = 30
    filter_coeff = {}
    Nyquist_freq = fs / 2
    for i, f_low_pass in enumerate(f_pass):
        f_pass = np.asarray([f_low_pass, f_low_pass + f_width])
        f_stop = np.asarray([f_pass[0] - f_trans, f_pass[1] + f_trans])
        wp = f_pass / Nyquist_freq
        ws = f_stop / Nyquist_freq
        order, wn = cheb2ord(wp, ws, gpass, gstop)
        b, a = signal.cheby2(order, gstop, ws, btype='bandpass')
        filter_coeff.update({i: {'b': b, 'a': a}})

    n_trials, n_channels, n_samples = eeg_data.get('x_data').shape
    if window_details:
        n_samples = int(fs * (window_details.get('tmax') - window_details.get('tmin'))) + 1
    filtered_data = np.zeros((len(filter_coeff), n_trials, n_channels, n_samples))
    for i, fb in filter_coeff.items():
        b = fb.get('b')
        a = fb.get('a')
        eeg_data_filtered = np.asarray([signal.lfilter(b, a, eeg_data.get('x_data')[j, :, :]) for j in range(n_trials)])
        if window_details:
            eeg_data_filtered = eeg_data_filtered[:, :, int((4.5 + window_details.get('tmin')) * fs):int(
                (4.5 + window_details.get('tmax')) * fs) + 1]
        filtered_data[i, :, :, :] = eeg_data_filtered
    return filtered_data