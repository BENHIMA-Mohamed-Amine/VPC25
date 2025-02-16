#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

import numpy as np
import librosa
import scipy
import librosa


def apply_mcadams(waveform, sample_rate, winLengthinms=20, shiftLengthinms=5, lp_order=32, mcadams=0.743):
    """
    Apply McAdams transformation on a single audio waveform.


    Parameters:
        waveform (torch.Tensor or np.ndarray): Audio signal. If multi-channel, the first channel is used.
        sample_rate (int): Sampling rate.
        winLengthinms (int): Window length in milliseconds.
        shiftLengthinms (int): Frame shift in milliseconds.
        lp_order (int): Order of LPC analysis.
        mcadams (float): McAdams coefficient.


    Returns:
        np.ndarray: Processed (anonymized) waveform (float32).
    """
    # Convert to a 1D numpy array if it’s a PyTorch tensor or multi-dimensional array.
    if hasattr(waveform, "numpy"):
        sig = waveform.squeeze().numpy()
    elif isinstance(waveform, np.ndarray):
        sig = waveform.squeeze()
    else:
        raise ValueError("waveform must be a PyTorch tensor or a NumPy array")


    eps = np.finfo(np.float32).eps
    sig = sig + eps  # avoid numerical issues


    fs = sample_rate
    winlen = int(np.floor(winLengthinms * 0.001 * fs))
    shift = int(np.floor(shiftLengthinms * 0.001 * fs))
    length_sig = len(sig)


    # Although NFFT is computed here, it is not used later.
    NFFT = 2 ** int(np.ceil(np.log2(winlen)))


    # Create analysis window and compute synthesis window satisfying perfect reconstruction constraint.
    wPR = np.hanning(winlen)
    K = np.sum(wPR) / shift
    win = np.sqrt(wPR / K)


    Nframes = 1 + int(np.floor((length_sig - winlen) / shift))
    sig_rec = np.zeros(length_sig)  # allocate output vector


    # Process each frame (starting from frame 1 as in original code)
    for m in range(1, Nframes):
        indices = np.arange(m * shift, min(m * shift + winlen, length_sig))
        frame = sig[indices] * win


        # LPC analysis using librosa
        a_lpc = librosa.core.lpc(frame + eps, order=lp_order)


        # Get poles from LPC coefficients
        _, poles, _ = scipy.signal.tf2zpk(np.array([1.0]), a_lpc)
        # Find indices of complex poles and take one from each conjugate pair.
        ind_imag = np.where(np.isreal(poles) == False)[0]
        ind_imag_con = ind_imag[::2]


        # Adjust pole angles with the McAdams coefficient.
        new_angles = np.angle(poles[ind_imag_con]) ** mcadams
        new_angles[new_angles >= np.pi] = np.pi
        new_angles[new_angles <= 0] = 0


        new_poles = poles.copy()
        for k in range(len(ind_imag_con)):
            new_poles[ind_imag_con[k]] = (np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k]))
            # Also update the conjugate pole (assumed to be the next one).
            if ind_imag_con[k] + 1 < len(new_poles):
                new_poles[ind_imag_con[k] + 1] = (np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k]))


        # Recover new LPC coefficients from modified poles.
        a_lpc_new = np.real(np.poly(new_poles))


        # Get the residual (excitation) and reconstruct the frame.
        res = scipy.signal.lfilter(a_lpc, [1.0], frame)
        frame_rec = scipy.signal.lfilter([1.0], a_lpc_new, res)
        frame_rec *= win


        out_indices = np.arange(m * shift, m * shift + len(frame_rec))
        sig_rec[out_indices] += frame_rec


    # Normalize the reconstructed signal.
    sig_rec = sig_rec / np.max(np.abs(sig_rec))
    return sig_rec.astype(np.float32), np.float32(sample_rate)


def anonymize(input_audio_path): # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    anonymization algorithm


    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.


    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32,
        which ensures compatibility with soundfile.write().
    sr : int
        The sample rate of the processed audio.
    """


    # Read the source audio file
    data, sample_rate = librosa.load(input_audio_path, sr=None)
    # Apply your anonymization algorithm
    anonymized_audio, anonymized_sample_rate = apply_mcadams(data, sample_rate)
    # Output:
    audio = anonymized_audio
    sr = anonymized_sample_rate
   
    return audio, sr
