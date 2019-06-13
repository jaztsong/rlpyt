
import copy

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        batch_size=128,
        learning_rate=2.5e-4,
        clip_grad_norm=10.,
        min_steps_learn=int(1e3),  # DEBUG
        double_dqn=False,
        prioritized_replay=False,
        n_step_return=1,
    ),
    env=dict(
        game="pong",
        episodic_lives=True,
    ),
    eval_env=dict(
        game="pong",  # NOTE: update in train script!
        episodic_lives=False,
        horizon=int(27e3),
    ),
    model=dict(dueling=False),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=2,
        batch_B=32,
        max_decorrelation_steps=1000,
        eval_n_envs=8,
        eval_max_steps=int(25e3),
        eval_max_trajectories=None,
        eval_min_envs_reset=2,
    ),
)

configs["0"] = config

config = copy.deepcopy(config)
config["algo"]["double_dqn"] = True
config["algo"]["prioritized_replay"] = True
config["model"]["dueling"] = True

configs["double_pri_duel"] = config

config = copy.deepcopy(config)
config["algo"]["n_step_return"] = 3
configs["ernbw"] = config