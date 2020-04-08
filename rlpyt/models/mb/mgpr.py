
import numpy as np
import torch
import gpytorch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mgpr import ExactGPModel


class GPDynamicsModel(torch.nn.Module):
    """Gaussian Process Model for learning environment transition."""

    def __init__(
            self,
            observation_shape,
            action_size,
            ):
        """Instantiate GP according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=int(np.prod(observation_shape)))
        # get 10 random points as placeholder
        self.gp = ExactGPModel(
            torch.rand(10, int(np.prod(observation_shape) + action_size)), torch.rand(
                10, int(np.prod(observation_shape))), self.likelihood
        )

    def set_train_data(self, observation, action, delta):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        X = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        Y = delta.view(T * B, -1)
        self.gp.set_train_data(X, Y, strict=False)

    def forward(self, observation, prev_action, prev_reward, action, train=False):
        lead_dim, T, B, _ = infer_leading_dims(observation,
            self._obs_ndim)
        gp_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        if train:
            gp = self.gp(gp_input)
            return gp

        with gpytorch.settings.fast_pred_var():
            gp = self.likelihood(self.gp(gp_input)).mean.squeeze(-1)

        gp = restore_leading_dims(gp, lead_dim, T, B)
        return gp

    def randomize(self):
        mean = 0; sigma = 1
        with torch.no_grad():
            self.gp.covar_module.base_kernel._set_lengthscale(0)
            self.gp.covar_module._set_outputscale(0)
            self.likelihood._set_noise(0.1)