"""Main entrypoint for duckie control."""

import time
from duckietown.sdk.robots.duckiebot import DB21J
import calibration
import lsl_api
import numpy as np
import sys

# Whether or not we are playing in duckiematrix simulation or with a real duckie
# robot
SIMULATION = False

# Parameters for converting actions to wheels. See
# DuckieController._action_to_wheels() for documentation
_ACTION_TO_WHEELS_PARAMS = dict(
    rotation_bias=-0.15,
    rotation_gain=0.24,
    max_speed=0.73,
    turn_slowdown=0.75,
)
# Parameters for rotation kernel. See _get_kernel() for documentation.
_ROTATION_KERNEL_PARAMS = dict(
    window=7,
    rebound_magnitude=0.2,
    rebound_delay=25,
)    

if SIMULATION:
    ROBOT: DB21J = DB21J("map_0/vehicle_0", simulated=True)
else:
    ROBOT: DB21J = DB21J("perseverance", simulated=False)
    

def _get_kernel(window, rebound_magnitude=0, rebound_delay=None):
    """Create smoothing kernel.
    
    The smoothing kernel is a half-triangular window, so recent events have
    highest weight.
    
    With just smoothing alone, it is difficult to prevent over-turning.
    Specifically, since we use right/left jaw-clenching as our EMG control, it
    is difficult for the driver to rapidly switch between left and right turns.
    As a result it is much easier to rapidly correct for under-turning (where
    correcting requires clenching harder on the same side) than over-turning
    (where correcting requires clenching on the other side). Consequently, we
    include an automatic correction for over-turning, namely a "rebound"
    feature in the smoothing kernel.
    
    If rebound_delay is not None, then the kernel has a "rebound" which is a
    negative triangular window component for more distant history.
    
    Args:
        window: Int. Window size of the positive component of the kernel. If
            rebound_delay is None, then this is the size of the resulting
            kernel. Otherwise, the resulting kernel has length
            2 * window + rebound_delay to include the negative and positive
            components and the delay between them.
        rebound_magnitude: Scalar. Weight of the negative "rebound" component of
            the kernel relative to the positive component.
        rebound_delay: None or int. If None, no rebound is used. If int, adds a
            negative "rebound" component with given delay before the positive
            component.
            
    Returns:
        kernel: Numpy array. Kernel, normalized to integrate to 1.
    """
    kernel = np.linspace(0, 1, window + 1)[1:]

    if rebound_delay is not None:
        kernel_bounceback = (
            -rebound_magnitude * np.linspace(0, 1, window + 1)[1:])
        kernel = np.concatenate([
            kernel_bounceback[::-1], np.zeros(rebound_delay), kernel
        ])

    kernel /= np.sum(kernel)
    return kernel


def _smooth(history, new_sample, kernel):
    """Smooths sample based on history with kernel.
    
    Also appends new_sample to history vector and trims history to not be longer
    than necessary.
    
    Args:
        history: List. Running list of samples.
        new_sample: Scalar. New sample to append to history.
        kernel: Array. Smoothing kernel.
        
    Returns:
        output: Float. Dot product of kernel with tail of history
    """
    # Add new sample and truncate history to kernel length
    history.append(new_sample)
    while len(history) > len(kernel):
        history.pop(0)

    # Pad history with zeros if it is shorter than kernel. This is only
    # necessary at the beginning of a run.
    history = (len(kernel) - len(history)) * [0] + history

    # Apply kernel
    output = np.dot(kernel, np.array(history))

    return output

    
class DuckieControllerEEG():
    """Controller class for duckie control with EEG."""
    
    def __init__(self,
                 snapshot_name=None,
                 action_to_wheels_params=_ACTION_TO_WHEELS_PARAMS,
                 rotation_kernel_params=_ROTATION_KERNEL_PARAMS):
        """Constructor.
        
        Args:
            snapshot_name: String. Name of the model snapshot to use for
                converting EEG features into (rotation, speed) actions.
            action_to_wheels_params: Dict. Parameters for converting
                (rotation, speed) actions into wheel speeds. See arguments of
                self._action_to_wheels().
            rotation_kernel_params: Dict. Parameters for computing the smoothing
                kernel for rotation actions.
        """
        self._action_to_wheels_params = action_to_wheels_params
        self._rotation_kernel = _get_kernel(**rotation_kernel_params)
        self._feature_stream = lsl_api.get_lsl_api()
        self._agent = calibration.Agent(
            in_features=self._feature_stream.n_features,
            out_features=2,
            name='',
            snapshot_name=snapshot_name,
        )
        self._rotation_history = []
        self._rotation_log = []
        
    def _action_to_wheels(self,
                          action, 
                          rotation_bias,
                          rotation_gain,
                          max_speed,
                          turn_slowdown):
        """Convert action to wheel speeds.
        
        Input action is a 2-iterable (rotation, speed). In practice we ignore
        the speed component of the action and instead the pilot only controls
        the rotation of the duckie bot. The speed of the duckie bot is
        determined based on the rotation.
        
        Args:
            action: 2-tuple of scalars. (rotation, speed) action.
            rotation_bias: Scalar. Bias of the rotation from the action space.
                This is subtracted from the rotation action. Positive means
                turning right too much, negative means turning left too much.
            rotation_gain: Scalar. Gain on rotation. Higher means more sensitive
                turning.
            max_speed: Scalar. Maximum speed of the duckie wheels.
            turn_slowdown: Scalar. Linear scaling factor by which speed is
                reduced as a function of rotation magnitude. This slows the
                duckie down during turns, which helps for better control.
        """
        rotation, _ = action
        rotation -= rotation_bias
        
        # Smooth the rotation action
        rotation = _smooth(
            self._rotation_history, rotation, self._rotation_kernel)
        
        # Log the rotation and print the mean rotation to the console. This
        # helps easily identify bias of the robot so we can quickly adjust the
        # rotation_bias parameter.
        self._rotation_log.append(rotation)
        mean_recent_rotation = np.mean(self._rotation_log[-20:])
        print(f'Average rotation = {mean_recent_rotation}')
        
        # Compute the speed
        max_speed *= (1 - turn_slowdown * np.abs(rotation))
        
        # Convert to wheel coordinates
        wheel_left = max_speed + rotation_gain * rotation
        wheel_right = max_speed - rotation_gain * rotation
        
        return [wheel_left, wheel_right]
    
    def __call__(self):
        """Sample current wheel speeds for the duckie robot."""
        features, _ = self._feature_stream()
        action = self._agent(features)
        wheels = self._action_to_wheels(action, **self._action_to_wheels_params)
        return wheels


def main():
    """Main loop to control robot."""
    controller = DuckieControllerEEG(snapshot_name='cal_nw_1_v2')

    # Wait until console input to start
    input('Press ENTER to start')
    
    ROBOT.motors.start()
    while True:
        wheel_left, wheel_right = controller()
        speeds = (wheel_left, wheel_right)
        ROBOT.motors.publish(speeds)
        time.sleep(0.001)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # Stop motors if program was terminated so duckie robot doesn't continue
        # to move
        ROBOT.motors.stop()
        sys.exit()