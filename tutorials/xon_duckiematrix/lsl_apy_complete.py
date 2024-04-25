from scipy.signal import lfilter, lfilter_zi, firwin
import numpy as np
from time import sleep


from lsl_api import BaseLSLStreamer

"""
Defines a complete LSL API class streaming and filtering EEG data +
extracting features. 


"""

class DataFilter:
    def __init__(self, sfreq: float, n_chan: int) -> None:
        # initialize variables for filtering
        self.bf: np.ndarray = firwin(
            32,
            np.array([1, 40]) / (sfreq / 2.),
            width=0.05,
            pass_zero=False,
        )
        self.af: list = [1.0]
        self.zi: np.ndarray = lfilter_zi(self.bf, self.af)
        self.filt_state: np.ndarray = np.tile(self.zi, (n_chan, 1)).transpose()

    def __call__(self, samples:np.ndarray) -> np.ndarray:
        # apply filter
        samples, self.filt_state = lfilter(self.bf, self.af, samples, axis=0, zi=self.filt_state)
        return samples
    


class RollingSTD:
    def __init__(self, n_features: int, history_chunks:int=100):
        self.recent_stddevs = []
        self.n_features = n_features
        self.history_chunks = history_chunks


    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Process batch of data of shape [timesteps_per_chunk, n_features]."""
        
        # Trim unused channels
        data = data[:, :self.n_features]
        
        # Zero-mean the data
        data -= np.nanmean(data, axis=0, keepdims=True)
        
        # Compute current stddevs
        current_stddevs = np.nanstd(data, axis=0)

        # Compute baseline (mean) stddev per channel
        self.recent_stddevs.append(current_stddevs)
        if len(self.recent_stddevs) > self.history_chunks:
            self.recent_stddevs.pop(0)
        baseline_stddevs = np.mean(self.recent_stddevs, axis=0)
        
        # Features are normalized stddevs
        features = current_stddevs / baseline_stddevs

        return features



class CompleteStreamer(BaseLSLStreamer):
    def __init__(self, stream_name:str, window:float=5.0, buffer:int=1)-> None:
        super().__init__(stream_name, window, buffer)
        self.filter = DataFilter(self.sfreq, self.n_chan)
        self.feature_extractor = RollingSTD(self.n_chan)
    
    def pull_data(self):
        window,sfreq, inlet, times, data = self.window, self.sfreq, self.inlet, self.times, self.data

        samples, timestamps = inlet.pull_chunk(
            timeout=0.01, max_samples=24)
        
        while inlet.samples_available() > 50:
            samples, timestamps = inlet.pull_chunk(
                timeout=0.01, max_samples=24)
        
        if not (isinstance(timestamps, list) and len(timestamps) > 1):
            sleep(0.01)
            return times, data 
        
        # Dejitter and append times
        num_new_samples = len(timestamps)
        timestamps = np.float64(np.arange(num_new_samples)) / sfreq
        timestamps += times[-1] + 1. / sfreq
        times = np.concatenate([times, timestamps])
        n_samples = int(sfreq * window)
        times = times[-n_samples:]
        
        # Add new data
        self.data = np.vstack([data, samples])
        self.data = self.data[-n_samples:]

    def __call__(self):
        self.pull_data()
        self.data = self.filter(self.data)
        features = self.feature_extractor(self.data)
        return features
        


    