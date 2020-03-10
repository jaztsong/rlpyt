
import math
import numpy as np


from rlpyt.replays.base import BaseReplayBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.buffer import torchify_buffer

class ModelBasedBuffer(BaseReplayBuffer):
    """ A simple sequential buffer that stores pairs of dynamics/states transitions.
    It helps extract individual episodes.

    """

    def __init__(self, example, size, B):
        self.T = T = math.ceil(size / B)
        self.B = B
        self.size = T * B
        self.t = 0  # Cursor (in T dimension).
        self.samples = buffer_from_example(example, (T, B),
            share_memory=self.async_)
        self.samples_return_ = self.samples.reward
        self.samples_done_n = self.samples.done
        self._buffer_full = False
        self.off_backward = 1  # Current invalid samples.
        self.off_forward = 1  # i.e. current cursor, prev_action overwritten.

    def append_samples(self, samples):
        """Write the samples into the buffer and advance the time cursor.
        Handle wrapping of the cursor if necessary (boundary doesn't need to
        align with length of ``samples``).  Compute and store returns with
        newly available rewards."""
        # filter out the invalid states
        after_done = samples.done.squeeze().roll(1)
        #fill the very first element as valid
        after_done[0] = False 
        # Extract all the valid samples
        samples = samples[(after_done == False).nonzero().squeeze()]
        T, B = get_leading_dims(samples, n_dim=2)  # samples.env.reward.shape[:2]
        assert B == self.B
        t = self.t
        if t + T > self.T:  # Wrap.
            idxs = np.arange(t, t + T) % self.T
        else:
            idxs = slice(t, t + T)
        self.samples[idxs] = samples
        if not self._buffer_full and t + T >= self.T:
            self._buffer_full = True  # Only changes on first around.
        self.t = (t + T) % self.T
        return T, idxs  # Pass these on to subclass.

    def sample_batch(self, batch_T):
        """Can dynamically input length of sequences to return, by ``batch_T``,
        else if ``None`` will use interanlly set value.  Returns batch with
        leading dimensions ``[batch_T, batch_B]``.
        """
        if self.t > batch_T:
            return torchify_buffer(self.samples[:-batch_T]) 
        else:
            return torchify_buffer(self.samples[:self.t+1])
