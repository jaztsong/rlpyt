
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.qpg.mlp import MuMlpModel
from rlpyt.models.mb.smgpr import GPDynamicsModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", ["mu"])


class GP_MlpAgent(BaseAgent):
    """Agent for deep deterministic policy gradient algorithm."""

    shared_mu_model = None

    def __init__(
            self,
            ModelCls=MuMlpModel,  # Mu model.
            DModelCls=GPDynamicsModel,
            model_kwargs=None,  # Mu model.
            d_model_kwargs=None,
            initial_model_state_dict=None,  # Mu model.
            initial_d_model_state_dict=None,
            action_std=0.01,
            action_noise_clip=None,
            ):
        """Saves input arguments; default network sizes saved here."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[20], output_max=1)
        if d_model_kwargs is None:
            d_model_kwargs = dict(num_inducing_pts=50)
        save__init__args(locals())
        super().__init__()  # For async setup.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """Instantiates mu and gp, and target_mu."""
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.d_model = self.DModelCls(**self.env_model_kwargs,
            **self.d_model_kwargs)
        if self.initial_d_model_state_dict is not None:
            self.d_model.load_state_dict(self.initial_d_model_state_dict)
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        if self.initial_model_state_dict is not None:
            self.target_model.load_state_dict(self.model.state_dict())
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            std=self.action_std,
            noise_clip=self.action_noise_clip,
            clip=env_spaces.action.high[0],  # Assume symmetric low=-high.
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)  # Takes care of self.model.
        self.target_model.to(self.device)
        self.d_model.to(self.device)

    def data_parallel(self):
        super().data_parallel()  # Takes care of self.model.
        if self.device.type == "cpu":
            self.d_model = DDPC(self.d_model)
        else:
            self.d_model = DDP(self.d_model)

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def predict_obs_delta(self, observation, prev_action, prev_reward, action):
        """Compute the next state for input state/observation and action (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        predict_obs_delta = self.d_model(*model_inputs, train=True)
        # Warning: Ideally, the output of the agent should always be on cpu.
        # But due to the complexity to migrate the GP output from gpu to cpu,
        # I decide to just leave it on device and defer to data sync in algo
        return predict_obs_delta

    def predict_next_obs_at_mu(self, observation, prev_action, prev_reward):
        """Compute Q-value for input state/observation, through the mu_model
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.model(*model_inputs)
        next_obs = self.d_model(
            *model_inputs, mu) + model_inputs[0] # model_inputs[0] is the observation
        return next_obs.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes distribution parameters (mu) for state/observation,
        returns (gaussian) sampled action."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.model(*model_inputs)
        action = self.distribution.sample(DistInfo(mean=mu))
        agent_info = AgentInfo(mu=mu)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.model.state_dict(), tau)

    def d_parameters(self):
        # FIXME: What should be the parameters: gp + likelihood
        return self.d_model.parameters()

    def mu_parameters(self):
        return self.model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.d_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.d_model.eval()
        self.distribution.set_std(self.action_std)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.d_model.eval()
        self.distribution.set_std(0.)  # Deterministic.

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            d_model=self.d_model.state_dict(),
            target_model=self.target_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.d_model.load_state_dict(state_dict["d_model"])
        self.target_model.load_state_dict(state_dict["target_model"])
