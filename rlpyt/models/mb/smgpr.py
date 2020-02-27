
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.smgpr import MultitaskGPModel


class GPDynamicsModel(torch.nn.Module):
    """Gaussian Process Model for learning environment transition."""

    def __init__(
            self,
            observation_shape,
            action_size,
            num_inducing_pts
            ):
        """Instantiate neural net according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.gp = MultitaskGPModel(
            input_size=int(np.prod(observation_shape)) + action_size,
            output_size=int(np.prod(observation_shape)),
            num_inducing_pts=num_inducing_pts,
        )
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=int(np.prod(observation_shape)))
        self.mml = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=int(np.prod(observation_shape)))
    def forward(self, observation, prev_action, prev_reward, action):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        gp_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        gp = self.likelihood(self.gp(gp_input)).rsample().squeeze(-1)
        gp = restore_leading_dims(gp, lead_dim, T, B)
        return gp

