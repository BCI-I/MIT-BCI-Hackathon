"""LSL API and entrypoint to view data and feature streams.

To view data and feature streams, run
$ python3 lsl_api.py

You may want to modify the _TARGET_STREAM, _WRITE_FILE, and _BASELINE_STDDEVS
parameters. See documentation below for descriptions of those.

This file is loosely based on the LSL viewer in BciPy:
https://github.com/CAMBI-tech/BciPy/blob/72053718855a602d146d5dadc235b9f0fa14b94a/bcipy/signal/viewer/lsl_viewer.py
"""

import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_stream
import seaborn as sns

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)
sns.set(style="whitegrid")

# Target stream name for EEG headset
_TARGET_STREAM = "X.on-102801-0070"
# Filename to write stddev features to
_WRITE_FILE = './data/stddevs_nw_v0.csv'
# Baseline stddev for each channel. Set to None to infer this from data
_BASELINE_STDDEVS = (21.7, 19.7, 13.9, 12.5, 26.9, 14.0, 13.9)
# How long to wait (in seconds) between re-trying samples.
_TIMEOUT = 0.01


class StddevFeatures():
    """Feature extractor that takes local stddev of each channel."""
    
    def __init__(self,
                 write_file=None,
                 timesteps_per_chunk=200,
                 history_chunks=100,
                 baseline_stddevs=_BASELINE_STDDEVS,
                 n_channels=7):
        """Constructor.
        
        Args:
            write_file: None or string. If string, path to a csv file to write
                stddev of each channel to. This is useful for figuring out the
                baseline stddev of each channel.
            timesteps_per_chunk: Number of streaming steps to use for each
                feature computation. This is only set as a property here, but is
                used in LSLAPI().
            history_chunks: Number of recent data chunks to use for computing
                the baseline stddev per channel. If baseline_stddevs is not
                None, this is not used. In practice the EEG streaming sample
                rate is ~400Hz, so 200 corresponds to half a second, which works
                pretty well.
            baseline_stddevs: None or iterable of length n_channels. If None,
                the baseline (mean) stddev for each channel is computed on a
                rolling basis using the last history_chunks data chunks. If not
                None, specifies the baseline stddev for each channel. This
                baseline stddev is used to normalize the momentary stddev for
                each channel.
            n_channels: Number of channels. Only the first n_channels channels
                of the input data are used for feature computation. Default is 7
                because the EEG data has only 7 EEG channels (the rest of the
                channels are accelerometer data and stuff we do not want).
        """
        self._n_channels = n_channels
        self._timesteps_per_chunk = timesteps_per_chunk
        self._history_chunks = history_chunks
        self._recent_stddevs = []
        
        if write_file is not None:
            self._writer = csv.writer(open(write_file, 'w'))
        else:
            self._writer = None
        
        if baseline_stddevs is None:
            self._baseline_stddevs = None
        else:
            self._baseline_stddevs = np.array(baseline_stddevs)
    
    def __call__(self, data):
        """Process batch of data of shape [timesteps_per_chunk, n_channels]."""
        
        # Trim unused channels
        data = data[:, :self._n_channels]
        
        # Zero-mean the data
        data -= np.nanmean(data, axis=0, keepdims=True)
        
        # Compute current stddevs
        current_stddevs = np.nanstd(data, axis=0)
        
        # write stddevs to csv if necessary
        if self._writer is not None:
            to_write = [str(x) for x in current_stddevs]
            self._writer.writerow(to_write)
        
        # Compute baseline (mean) stddev per channel
        if self._baseline_stddevs is None:
            self._recent_stddevs.append(current_stddevs)
            if len(self._recent_stddevs) > self._history_chunks:
                self._recent_stddevs.pop(0)
            baseline_stddevs = np.mean(self._recent_stddevs, axis=0)
        else:
            baseline_stddevs = self._baseline_stddevs
        
        # Features are normalized stddevs
        features = current_stddevs / baseline_stddevs
        
        return features
    
    @property
    def n_features(self):
        return self._n_channels
    
    @property
    def timesteps_per_chunk(self):
        return self._timesteps_per_chunk
    

class LSLAPI():
    """LSL API for streaming and plotting EEG data."""

    def __init__(self,
                 stream,
                 feature_extractor,
                 window=5.,
                 buf=1,
                 subsample_plot=3,
                 plot=True):
        """Constructor.
        
        Args:
            stream: LSL data stream.
            feature_extractor: Callable mapping data -> features.
            window: Scalar. Window size in seconds for data caching and
                plotting.
            buf: Int. Number of second for the data stream buffer.
            subsample_plot: Int. Stride for subsampling data when plotting.
            plot: Bool. Whether to plot data stream real-time.
        """
        self._feature_extractor = feature_extractor
        self._n_feat = self._feature_extractor.n_features
        self._window = window
        self._inlet = StreamInlet(stream, max_buflen=buf, max_chunklen=buf)
        self._subsample_plot = subsample_plot
        self._plot = plot
        
        # General metadata about inlet stream
        info = self._inlet.info()
        self._sfreq = info.nominal_srate()
        self._n_samples = int(self._sfreq * self._window)
        self._n_chan = info.channel_count()

        # Initialize variable for data, features, and times
        self._times = np.arange(-self._window, 0, 1. / self._sfreq)
        self._data = np.zeros((self._n_samples, self._n_chan))
        self._features = np.zeros((self._n_samples, self._n_feat))
        
        # Raw data filter
        self._display_every = int(0.2 / (12 / self._sfreq))
        self._bf = firwin(
            32,
            np.array([1, 40]) / (self._sfreq / 2.),
            width=0.05,
            pass_zero=False,
        )
        self._af = [1.0]
        zi = lfilter_zi(self._bf, self._af)
        self._filt_state = np.tile(zi, (self._n_chan, 1)).transpose()
        
    def __call__(self):
        """Return current vector of features.
        
        Returns:
            features: Numpy array of shape [n_features]. Features for the most
                recent chunk of streamed data.
        """
        
        samples, timestamps = self._inlet.pull_chunk(
                timeout=0.01, max_samples=24)
        while self._inlet.samples_available() > 50:
            samples, timestamps = self._inlet.pull_chunk(
                timeout=0.01, max_samples=24)
        
        if not (isinstance(timestamps, list) and len(timestamps) > 1):
            sleep(_TIMEOUT)
            return self()
        
        # Dejitter and append times
        num_new_samples = len(timestamps)
        timestamps = np.float64(np.arange(num_new_samples)) / self._sfreq
        timestamps += self._times[-1] + 1. / self._sfreq
        self._times = np.concatenate([self._times, timestamps])
        self._n_samples = int(self._sfreq * self._window)
        self._times = self._times[-self._n_samples:]
        
        # Add new data
        filt_samples, self._filt_state = lfilter(
            self._bf, self._af, samples, axis=0, zi=self._filt_state)
        self._data = np.vstack([self._data, filt_samples])
        self._data = self._data[-self._n_samples:]
        
        # Return new features
        features = self._feature_extractor(
            self._data[-self._feature_extractor.timesteps_per_chunk:])
        
        # Update self._features by the tiled features
        if self._plot:
            features_tiled = features[None] * np.ones((num_new_samples, 1))
            self._features = np.vstack([self._features, features_tiled])
            self._features = self._features[-self._n_samples:]
        
        return features
        
    def update_plot(self, update):
        """Update plot of data and features."""
        
        # Call to read and cache new samples from the inlet stream.
        _ = self.__call__()
        
        # Plot if necessary
        if self._plot and update:
            # Update data plot
            plot_data = np.copy(self._data[::self._subsample_plot])
            plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
            plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
            for chan in range(self._n_chan):
                self.lines_data[chan].set_xdata(
                    self._times[::self._subsample_plot] - self._times[-1])
                self.lines_data[chan].set_ydata(plot_data[:, chan] - chan)
            self._ax_raw.set_xlim(-self._window, 0)
            
            # Update features plot
            plot_data = np.copy(self._features[::self._subsample_plot])
            plot_data = 0.5 * (plot_data - 0.5)
            for chan in range(self._n_feat):
                self.lines_feat[chan].set_xdata(
                    self._times[::self._subsample_plot] - self._times[-1])
                self.lines_feat[chan].set_ydata(plot_data[:, chan] - chan)
            self._ax_feat.set_xlim(-self._window, 0)
            
            self._fig.canvas.draw()
            plt.pause(_TIMEOUT)
                
    def run_plot_loop(self):
        """Run a plotting loop to show streamed data and features."""
        
        sns.despine(left=True)
        self._fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        self._ax_raw, self._ax_feat = axes
        times_plot = self._times[::self._subsample_plot]
    
        # Data lines
        lines_data = []
        for channel in range(self._n_chan):
            line, = self._ax_raw.plot(
                times_plot, np.zeros_like(times_plot) - channel, lw=1)
            lines_data.append(line)
        self.lines_data = lines_data
        self._ax_raw.set_ylim(-self._n_chan - 0.5, 0.5)
        self._ax_raw.set_xlabel('Time (s)')
        self._ax_raw.xaxis.grid(False)
        self._ax_raw.set_yticks(np.arange(0, -self._n_chan, -1))
    
        # Feature lines
        lines_feat = []
        for channel in range(self._n_feat):
            line, = self._ax_feat.plot(
                times_plot, np.zeros_like(times_plot) - channel, lw=1)
            lines_feat.append(line)
        self.lines_feat = lines_feat
        self._ax_feat.set_ylim(-self._n_feat - 0.5, 0.5)
        self._ax_feat.set_xlabel('Time (s)')
        self._ax_feat.xaxis.grid(False)
        self._ax_feat.set_yticks(np.arange(0, -self._n_feat, -1))
        
        update_index = 0
        while True:
            update = update_index % self._display_every == 0
            self.update_plot(update=update)
            update_index += 1
            
    @property
    def n_features(self):
        return self._feature_extractor.n_features
            
            
def get_lsl_api(target_stream_name=_TARGET_STREAM, write_file=None):
    """Get LSL API for given EEG stream name.
    
    Args:
        target_stream_name: String. Name of target stream.
        write_file: None or string. If string, path to file to write stddev
            features.
    """
    
    logging.debug("looking for an EEG stream...")
    # Loop for a while to look for the EEG stream, because sometimes the stream
    # is not found by resolve_stream() even when it exists
    for _ in range(100):
        streams = resolve_stream('type', 'EEG')
        target_stream = None
        for stream in streams:
            if stream.name() == target_stream_name:
                target_stream = stream
                break
            
        if target_stream:
            logging.debug("Start aquiring data")
            feature_extractor = StddevFeatures(write_file=write_file)
            lsl_api = LSLAPI(target_stream, feature_extractor=feature_extractor)
            return lsl_api
    
    print(f'Cannot find stream {target_stream_name}')
    return None


class NoiseAPI():
    """Noise API mimicking LSL API except returning noise features."""
    
    def __init__(self, n_features=7):
        self._n_features = n_features
        
    def __call__(self):
        return np.random.uniform(-1, 1, size=self._n_features)
        
    @property
    def n_features(self):
        return self._n_features


def get_noise_api():
    return NoiseAPI()


if __name__ == '__main__':
    """Run streaming and plotting."""
    lsl_api = get_lsl_api(write_file=_WRITE_FILE)
    if lsl_api is None:
        exit()
    lsl_api.run_plot_loop()
    