
"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.mb.gp_mlp import GP_Mlp
from rlpyt.agents.mb.gp_mlp_agent import GP_MlpAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
import torch


def build_and_train(env_id="Hopper-v3", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=50,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=2,
        eval_max_steps=int(51e3),
        eval_max_trajectories=200,
    )
    # The cost function for InvertedPendulumBulletEnv
    def obs_cost_fn(x):
        target = torch.FloatTensor([0,0,1,0,0])
        c = (x - target)**2
        c = -c.sum(dim=1)
        return -c.exp()
    algo = GP_Mlp(obs_cost_fn=obs_cost_fn)  # Run with defaults.
    agent = GP_MlpAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=1e6,
        log_interval_steps=200,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    name = "gp_mlp_" + env_id
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID',
                        default='InvertedPendulumBulletEnv-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
