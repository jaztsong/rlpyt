
import math
import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_out = train_y.shape[1]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_out]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1],
                    # lengthscale_prior = gpytorch.priors.GammaPrior(1,10),
                    batch_shape=torch.Size([self.num_out])),
                batch_shape=torch.Size([self.num_out]),
                # outputscale_prior = gpytorch.priors.GammaPrior(1.5,2),
                )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    @property
    def output_size(self):
        """Returns the final output size."""
        return self._output_size


