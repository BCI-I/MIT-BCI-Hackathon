from pylsl import StreamInlet, resolve_stream, StreamInfo, resolve_streams
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, List
from IPython.display import display, clear_output


"""
Link to the original winning team code: https://github.com/nwatters01/mit_bci_hackathon
"""

def print_lsl_streams():
    # Resolve EEG streams
    streams = resolve_streams()

    # Iterate through streams
    print(f"Found {len(streams)} streams")
    print("---------------")

    for stream in streams:
        print("Stream Name:", stream.name())
        print("Stream Type:", stream.type())
        print("Stream ID:", stream.source_id())   # this should match your X.on Serial Number
        print("Stream Unique Identifier:", stream.uid())
        print("---------------")

def get_lsl_stream(target_stream_name:str, n_attempts:int=100,) -> StreamInfo:
    """Get LSL API for given EEG stream name.
    
    Args:
        target_stream_name: String. Name of target stream.
        n_attempts: Int. Number of attempts to look for the stream.
    """
    
    print("looking for an EEG stream...")
    # Loop for a while to look for the EEG stream, because sometimes the stream
    # is not found by resolve_stream() even when it exists
    for _ in range(n_attempts):
        streams = resolve_stream('type', 'EEG')
        for stream in streams:
            if stream.name() == target_stream_name:
                print(f"Found {target_stream_name} stream")
                return stream
                break
            

    raise ValueError(f'Cannot find stream {target_stream_name}')


class BaseLSLStreamer:
    subsample_plot: int = 5

    def __init__(self, stream_name:str, window:float=5.0, buffer:int=1)-> None:
        stream: StreamInfo = get_lsl_stream(stream_name)
        self.inlet = StreamInlet(stream, max_buflen=buffer, max_chunklen=buffer)
        self.stream_name = stream_name
        self.window = window

        # Get stream info
        info = self.inlet.info()
        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * window)
        self.n_chan = info.channel_count()

        # initialize variables
        self.times = np.arange(-window, 0, 1. / self.sfreq)
        self.data = np.zeros((self.n_samples, self.n_chan))

        print(f"""Starting streamer for {stream_name} stream
    Sampling rate: {self.sfreq}
    Number of channels: {self.n_chan}
    Window: {window} seconds
              """)

    def initialize_plotter(self):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))
        self.lines = self.ax.plot(self.times, self.data)
        # self.ax.set_ylim(-self.n_chan - 0.5, 0.5)
        self.ax.set_xlabel('Time (s)')
        self.ax.xaxis.grid(False)
        # self.ax.set_yticks(np.arange(0, -self.n_chan, -1))
        plt.ion()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        
        # Update data plot
        plot_data = np.copy(self.data[::self.subsample_plot])
        plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
        plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
        for chan in range(self.n_chan):
            self.ax.plot(
                self.times[::self.subsample_plot] - self.times[-1],
                plot_data[:, chan] - chan,
                color="C0",
                alpha=0.5,
            )
        self.ax.set_xlim(-self.window, 0)
        self.ax.set(title="Real-time EEG")
        

        display(self.fig)
        clear_output(wait = True)
        plt.pause(0.01)

        


    def stream(self, n_steps:int, update_plot_every_s:float=0.2, callbacks:Optional[List[Callable]]=None):
        update_plot_every_n = int(update_plot_every_s / (12 / self.sfreq))

        self.initialize_plotter()

        update_idx = 0 
        for i in range(n_steps):
            self.pull_data()
            if callbacks is not None:
                for cb in callbacks:
                    self.data = cb(self.data)
            if update_idx % update_plot_every_n == 0:
                self.update_plot()
            update_idx += 1
            
