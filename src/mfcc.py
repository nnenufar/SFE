###
# Receives: waveform, beat timestamps
# Outputs: array of shape [len(timestamps), feat_dimension]

import librosa
import numpy as np
from librosa.feature import melspectrogram, mfcc

class MFCC():
    def __init__(self, sr, n_mfcc, frame_length):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length

    def extract_mfcc(self, waveform, identifier, beat_index):
        assert self.frame_length % 2 == 0, "Frame length must be a multiple of 2"

        center = beat_index
        start = center - self.frame_length // 2
        end = center + self.frame_length // 2
        
        frame = waveform[start:end]

        try:
            S = melspectrogram(y=frame, sr=self.sr, n_fft=self.frame_length, center=False, n_mels=128) # (n_mels, n_frames: 1)
            feats = mfcc(S=librosa.power_to_db(S), n_mfcc=self.n_mfcc)
            # TODO: normalize MFCCs?
        except Exception as e:
            print(f"Error extracting MFCCs for {identifier} at beat index {beat_index}: {e}")
            feats = np.zeros((self.n_mfcc, 1))
        return feats