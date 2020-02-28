import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
    AsyncTlUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.visom import VisdomLinePlotter
from rlpyt.algos.utils import valid_from_done

OptInfo = namedtuple("OptInfo",
    ["muLoss", "dLoss", "muGradNorm", "dGradNorm"])


class GP_Mlp(RlAlgorithm):
    """Model-based algorithm that uses Gaussian Process to predict model and a deep
    neural network to control."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=64,
            min_steps_learn=int(1e1), # very efficient
            target_update_tau=0.01,
            target_update_interval=10,
            policy_update_interval=10,
            learning_rate=1e-4,
            d_learning_rate=1e-2,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=1e8,
            d_target_clip=1e6,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only.
            bootstrap_timelimit=True,
            obs_cost_fn=None
            ):
        """Saves input arguments."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes  optimizer.
        Use in non-async runners."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = 20
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        # Agent give min itr learn.?
        self.optim_initialize(rank)
        self.initialize_visom()

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.mu_optimizer = self.OptimCls(self.agent.mu_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.d_optimizer = self.OptimCls(self.agent.d_parameters(),
            lr=self.d_learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.d_optimizer.load_state_dict(self.initial_optim_state_dict["d"])
            self.mu_optimizer.load_state_dict(self.initial_optim_state_dict["mu"])

    def initialize_visom(self):
        self.plotter = VisdomLinePlotter(env_name='main')


    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            if self.mid_batch_reset and not self.agent.recurrent:
                valid = torch.ones_like(samples.env.done, dtype=torch.float)
            else:
                valid = valid_from_done(samples.done)
            if self.bootstrap_timelimit:
                # To avoid non-use of bootstrap when environment is 'done' due to
                # time-limit, turn off training on these samples.
                valid *= (1 - getattr(samples.env.env_info, "timeout", None).float())
            optimizing_model_iter = 20
            for _ in range(optimizing_model_iter):
                self.d_optimizer.zero_grad()
                d_loss = self.d_loss(samples, valid)
                d_loss.backward()
                d_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.d_parameters(), self.clip_grad_norm)
                self.d_optimizer.step()
                opt_info.dLoss.append(d_loss.item())
                opt_info.dGradNorm.append(d_grad_norm)

            self.update_counter += 1
            if self.update_counter % self.policy_update_interval == 0:
                self.mu_optimizer.zero_grad()
                mu_loss = self.mu_loss(samples, valid)
                mu_loss.backward()
                mu_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.mu_parameters(), self.clip_grad_norm)
                self.mu_optimizer.step()
                opt_info.muLoss.append(mu_loss.item())
                opt_info.muGradNorm.append(mu_grad_norm)
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        return opt_info


    def mu_loss(self, samples, valid):
        """Computes the mu_loss by rolling out from s_0."""
        T = 40
        mu_loss = 0
        n_obs_dim = samples.env.observation.size(-1)
        n_samples = samples.env.observation.size(0)
        for dim in range(n_obs_dim):
            self.plotter.plot('dim='+str(dim), 'true',
                              'Model Multi-Step Prediction dim='+str(dim),
                              range(n_samples-1),
                              samples.env.observation[1:, 0, dim].data.numpy(), update='replace')
            self.plotter.plot('dim='+str(dim), 'predict',
                              'Model Multi-Step Prediction dim='+str(dim), [0], [0], update='replace')
        next_obs = samples.env.observation[0, :, :]
        prev_action = samples.agent.prev_action[0, :, :]
        prev_reward = samples.env.prev_reward[0, :]
        for t in range(T):
            next_obs = self.agent.predict_next_obs_at_mu(
                next_obs, prev_action, prev_reward)
            mu_loss += self.obs_cost_fn(next_obs)

            for dim in range(n_obs_dim):
                self.plotter.plot('dim='+str(dim), 'predict',
                                  'Model Multi-Step Prediction dim='+str(dim), [t], next_obs[:, dim].data.numpy(), update='append')

        return mu_loss

    def d_loss(self, samples, valid):
        """Constructs the prediction loss Input samples have leading batch dimension [B,..] (but not time)."""
        assert(samples.env.observation.size(0) > 1)
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )

        # only predict 1:n-1
        pred_obs_delta = self.agent.predict_obs_delta(
            agent_inputs.observation[:-1, :, :],
            agent_inputs.prev_action[:-1, :, :],
            agent_inputs.prev_reward[:-1, :],
            samples.agent.action[:-1, :, :])
        obs_delta = agent_inputs.observation[1:,:,:] - agent_inputs.observation[:-1,:,:]
        # next_obs = torch.clamp(
        #     next_obs, -samples.env.observation_space.high, samples.env.observation_space.high)
        d_losses = -self.agent.d_model.mml(pred_obs_delta, obs_delta)
        d_loss = valid_mean(d_losses, valid)
        return d_loss

    def optim_state_dict(self):
        return dict(d=self.d_optimizer.state_dict(),
            mu=self.mu_optimizer.state_dict())

    def load_optim_state_dict(self, state_dict):
        self.d_optimizer.load_state_dict(state_dict["d"])
        self.mu_optimizer.load_state_dict(state_dict["mu"])
