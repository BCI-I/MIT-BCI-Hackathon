"""Calibration GUI based on Tkinter."""

from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk


class CalibrationGUI():
    """Calibration GUI class.
    
    The calibration learns the mapping between the EEG space and the
    (rotation, speed) action space for the duckie. The action space is
    2-dimensional and is represented by a rectangular visual display. To
    calibrate, a red ball travels via a serpentine pattern in this display. The
    subject should simultaneously track the ball. After completing it's tour, a
    model is trained and saved, and the process repeats. The model trains after
    each trial, and the model-decoded ball is displayed as a green ball.
    
    This is a tkinter GUI, which uses .after() and .mainloop() to loop the
    display. Due to tkinter's API, in order to insert other computations into
    this loop we need to use callback functions. That in combination with the
    design of calibration.Agent() that runs trials one-by-one is why this class
    has the ugly set_callback function for computing the agent action.
    """
    
    def __init__(self,
                 n_levels=5,
                 step_size=4,
                 ball_size=20,
                 canvas_width=800,
                 canvas_height=800,
                 buffer=10):
        """Constructor.
        
        Args:
            n_levels: Int. Number of horizontal rows in the serpentine pattern.
            step_size: Int. Ball speed.
            ball_size: Int. Ball size.
            canvas_width: Int. Width of the display.
            canvas_height: Int. Height of the display.
            buffer: Int. Border on the display to prevent the ball from
                appearing out of frame.
        """
        self._n_levels = n_levels
        self._step_size = step_size
        self._ball_size = ball_size
        self._canvas_width=canvas_width
        self._canvas_height=canvas_height
        self._buffer = buffer
        
        # Create canvas.
        self.root = tk.Tk()
        self.canvas = tk.Canvas(
            self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        
        # Create target and agent balls
        self.target = self.canvas.create_oval(
            self._buffer,
            self._canvas_height - self._ball_size - self._buffer,
            self._ball_size + self._buffer,
            self._canvas_height - self._buffer,
            fill='red',
        )
        self.agent = self.canvas.create_oval(
            -3 * self._buffer,
            self._canvas_height - self._ball_size + 3 * self._buffer,
            self._ball_size - 3 * self._buffer,
            self._canvas_height + 3 * self._buffer,
            fill='green',
        )
        
        # Setup sequence of steps for the target ball to execute.
        row_steps = (
            (self._canvas_width - 2 * (self._ball_size + self._buffer)) //
            self._step_size
        )
        column_steps = (
            (self._canvas_height - 2 * (self._buffer + self._ball_size)) //
            (self._step_size * (self._n_levels - 1))
        )
        deltas = []
        moving_right = 1
        for level in range(self._n_levels):
            # Add horizontal leg
            h_delta = moving_right * self._step_size
            for _ in range(row_steps):
                deltas.append((h_delta, 0))
            moving_right *= -1
            
            if level < self._n_levels - 1:
                # Add vertical leg
                for _ in range(column_steps):
                    deltas.append((0, -self._step_size))
        self._deltas = deltas
        self.reset()
        
    def _coords_to_pos(self, coords):
        """Convert canvas coordinates to position in [-1, 1] x [0, 1]."""
        w = float(coords[0]) / self._canvas_width
        w = 2 * w - 1
        h = 1. - float(coords[1]) / self._canvas_width
        return np.array([w, h])
    
    def _pos_to_coords(self, pos):
        """Convert position in [-1, 1] x [0, 1] to canvas coordinates."""
        w = self._canvas_width * 0.5 * (1 + pos[0])
        h = self._canvas_height * (1. - pos[1])
        return (w, h)
    
    def _move_object(self, obj, target_coords):
        """Move object to target canvas coordinates."""
        current_coords = self.canvas.coords(obj)
        delta = (
            target_coords[0] - current_coords[0],
            target_coords[1] - current_coords[1],
        )
        self.canvas.move(obj, delta[0], delta[1])
        
    def set_callback(self, callback):
        """Set callback function (target, finished) --> agent_pos."""
        self._callback = callback
        
    def reset(self):
        """Reset function to initialize a new trial."""
        self._step_index = 0
        target_coords = (
            self._buffer,
            self._canvas_height - self._ball_size - self._buffer,
        )
        self._move_object(self.target, target_coords)
        
    def step(self):
        """Take a step of calibration."""
        
        # If new trial, pause to let the subject get ready.
        if self._step_index == 0:
            plt.pause(2)
        
        # Move target ball
        delta = self._deltas[self._step_index]
        self._step_index += 1
        self.canvas.move(self.target, delta[0], delta[1])
        
        # Move agent ball
        finished = self._step_index >= len(self._deltas)
        target_pos = self._coords_to_pos(self.canvas.coords(self.target))
        agent_pos = self._callback(target_pos, finished)
        agent_coords = self._pos_to_coords(agent_pos)
        self._move_object(self.agent, agent_coords)
        
        # Reset if finished trial
        if finished:
            self.reset()
            
        # Loop next step
        self.root.after(10, self.step)
        
    @property
    def n_features(self):
        return 2
        

if __name__ == '__main__':
    """Run calibration with dummy agent callback that does nothing."""
    gui = CalibrationGUI()
    callback = lambda target, finished: np.array([0, 0])
    gui.set_callback(callback)
    gui.root.after(10, gui.step)
    gui.root.mainloop()
    