"""Entrypoint to run calibration GUI.

Run a calibration protocol, using either noise features or EEG features.

To run using noise features, run:
    $ python3 calibration.py name=$YOUR_NAME stream=noise

To run using EEG features, run:
    $ python3 calibration.py name=$YOUR_NAME stream=lsl_api
    
To fine-tune a previously run snapshot, run:
    $ python3 calibration.py name=$NEW_MODEL_NAME stream=lsl_api \
        snapshot_name=$SNAPSHOT_NAME
    
The string $YOUR_NAME is an identified to label this calibration, and points to
the file in the ./snapshots/ directory that contains the parameters fit by this
calibration. If you want to avoid overriding a calibration, use a different name
for future calibration runs.

The calibration learns the mapping between the input space (EEG or noise) and
the (rotation, speed) action space for the duckie. The action space is
2-dimensional and is represented by a rectangular visual display. To calibrate,
a red ball travels via a serpentine pattern in this display. The user should
simultaneously track the ball. After completing it's tour, a model is trained
and saved, and the process repeats. The model trains after each trial, and the
model-decoded ball is displayed as a green ball. These trials will loop (and the
calibration will improve) indefinitely, so terminate the program whenever you
think it is good enough. When you terminate, it may take a couple seconds to
navigate to the terminal to do so, during which your EEG input will not be good
for calibration, so when you want to terminate we recommend focusing on the
calibration through the end of a trial, waiting for the next trial to start
(indicating training for the previous trial has completed), then terminate
before the new trial ends.

In practice, for the duckie controller in duckie.py, only the rotation component
of the action space is used. So to calibrate you can ignore the vertical
(height) axis and just focus on left/right. Clenching your jaw on the left and
right side to track the left/right position of the red ball leads to pretty good
calibration.
"""

import gui as gui_lib
import numpy as np
import lsl_api
from pathlib import Path
import sys
import torch

_SNAPSHOT_DIR = Path('./snapshots')


class MLP(torch.nn.Module):
    """MLP model."""
    
    def __init__(self, in_features, layer_features, activation=None):
        """Constructor.
        
        Args:
            in_features: Int. Number of features for input to MLP.
            layer_featuers: Iterable of ints. Number of features for each layer
                after input.
            activation: None or torch activation function. Defaults to
                torch.nn.Sigmoid().
        """
        super(MLP, self).__init__()
        
        self._in_features = in_features
        self._layer_features = layer_features
        if activation is None:
            activation = torch.nn.Sigmoid()
        self.activation = activation

        features_list = [in_features] + list(layer_features)
        module_list = []
        for i in range(len(features_list) - 1):
            if i > 0:
                module_list.append(activation)
            layer = torch.nn.Linear(
                in_features=features_list[i],
                out_features=features_list[i + 1]
            )
            module_list.append(layer)
        
        self.net = torch.nn.Sequential(*module_list)

    def forward(self, x):
        return self.net(x)

    @property
    def in_features(self):
        return self._in_features

    @property
    def layer_features(self):
        return self._layer_features
        
    @property
    def out_features(self):
        return self._layer_features[-1]
    
    
class Agent(torch.nn.Module):
    """Agent class."""
    
    def __init__(self,
                 name,
                 in_features,
                 hidden_layer_sizes=(256, 256),
                 out_features=2,
                 snapshot_name=None):
        """Constructor.
        
        Args:
            in_features: Int. Number of features of input to agent. In practice
                this is the number of features of the EEG feature extractor.
            out_features: Int Number of features for the output action space.
            hidden_layer_sizes: Iterable of ints. Number of features for hidden
                layers of MLP.
            snapshot_name: None or string. If string, must name a snapshot in
                ./snapshots/ directory, in which case the model parameters are
                initialized from that snapshot. Otherwise randomly initialized
                model parameters.
        """
        super(Agent, self).__init__()
        self._name = name
        self._snapshot_path = _SNAPSHOT_DIR / name
        
        self._net = MLP(
            in_features=in_features,
            layer_features=tuple(list(hidden_layer_sizes) + [out_features]),
        )
        if snapshot_name is not None:
            # Load agent from snapshot
            state_dict_path = _SNAPSHOT_DIR / snapshot_name
            self._net.load_state_dict(torch.load(state_dict_path))
            print(f'Loaded from snapshot {state_dict_path}')
            
    def __call__(self, features, as_numpy=True):
        """Convert features to action.
        
        Input features if a numpy array and may be either batched of size
        [batch_size, in_features] or un-batched of size [in_features].
        """
        no_batch = len(features.shape) == 1
        
        if no_batch:
            features = features[None]
        action = self._net(torch.from_numpy(features.astype(np.float32)))
        if no_batch:
            action = action[0]
            
        # Convert to numpy is necessary
        if as_numpy:
            action = action.detach().numpy()
            
        return action
    
    def snapshot(self):
        """Save model parameters."""
        torch.save(self._net.state_dict(), self._snapshot_path)
        print(f'Saved snapshot to {self._snapshot_path}')


def _sample_batch(*arrays, batch_size):
    """Sample a batch of data from arrays."""
    num_samples = arrays[0].shape[0]
    indices = np.random.choice(num_samples, size=batch_size)
    batch_arrays = [x[indices] for x in arrays]
    return batch_arrays
    
    
class Calibrator():
    """Action space calibrator to learn mapping from EEG features to actions."""
    
    def __init__(self,
                 name,
                 feature_stream,
                 gui,
                 snapshot_name=None,
                 batch_size=128,
                 training_steps=2000,
                 optimizer=torch.optim.SGD,
                 lr=0.01,
                 grad_clip=1):
        """Constructor.
        
        Args:
            name: String. Name for this calibration. The model will be saved to
                ./snapshots/$name after each trial through the gui.
            features_stream: Callable that returns current features. See
                lsl_api.py.
            gui: TKinter gui for the calibration. See gui.py.
            snapshot_name: None or string. If string, must point to a snapshot
                in ./snapshots/ directory and model parameters will be
                initialized from that snapshot. If None, model parameters are
                randomly initialized.
            batch_size: Int. Batch size for model training.
            training_steps: Int. Number of model training steps to perform after
                each calibration trial.
            optimized: Torch optimizer for training.
            lr: Float. Learning rate for optimizer.
            grad_clip: Scalar. Gradient clipping.
        """
        self._feature_stream = feature_stream
        self._gui = gui
        self._name = name
        self._agent = Agent(
            in_features=feature_stream.n_features,
            out_features=gui.n_features,
            name=name,
            snapshot_name=snapshot_name,
        )
        
        # Optimization
        self._batch_size = batch_size
        self._training_steps = training_steps
        self._optimizer = optimizer(self._agent.parameters(), lr=lr)
        self._grad_clip = grad_clip
        
    def _train(self):
        """Run training loop and save model."""
        print(f'Training')
        all_inputs = np.array(self._all_inputs)
        all_targets = np.array(self._all_targets)
        for _ in range(self._training_steps):
            self._optimizer.zero_grad()
            
            # Sample batch
            batch_inputs, batch_targets = _sample_batch(
                all_inputs, all_targets, batch_size=self._batch_size,
            )
            batch_outputs = self._agent(batch_inputs, as_numpy=False)
            batch_targets = torch.from_numpy(batch_targets.astype(np.float32))
            
            # Evaluate loss
            loss = torch.mean(torch.sum(torch.square(
                batch_targets - batch_outputs), axis=1))
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._agent.parameters(), self._grad_clip)
            self._optimizer.step()
            
        # Save model
        print(f'Saving')
        self._agent.snapshot()
    
    def __call__(self):
        """Run calibration loop."""
        self._gui.reset()
        self._all_inputs = []
        self._all_targets = []
        
        # Create callback function for the gui
        def _callback(target, fin):
            if fin:
                print('Finished calibration trial')
                self._train()
                self._gui.reset()
                self._all_inputs = []
                self._all_targets = []
                
            features = self._feature_stream()
            agent_pos = self._agent(features)
            self._all_inputs.append(features)
            self._all_targets.append(target)
            return agent_pos

        # Set gui callback and run main loop
        self._gui.set_callback(_callback)
        self._gui.root.after(3, self._gui.step)
        self._gui.root.mainloop()
        
        # Save inputs and targets for training
        self._all_inputs = np.array(self._all_inputs)
        self._all_targets = np.array(self._all_targets)
        
        
def main(name, snapshot_name=None, stream='lsl'):
    """Main function."""
    
    # Get feature stream
    if stream == 'lsl':
        feature_stream = lsl_api.get_lsl_api()
    elif stream == 'noise':
        feature_stream = lsl_api.get_noise_api()
    else:
        raise ValueError(f'Invalid stream {stream}')
    
    # Create gui and calibrator 
    gui = gui_lib.CalibrationGUI()
    calibrator = Calibrator(
        name=name,
        feature_stream=feature_stream,
        gui=gui,
        snapshot_name=snapshot_name,
    )
    
    # Run calibration loop
    calibrator()


if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:]))
    