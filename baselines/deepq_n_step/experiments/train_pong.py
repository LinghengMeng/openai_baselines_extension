from baselines import deepq_n_step
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari
import os.path as osp
import datetime
import argparse

def main(env_name, seed, n_step, exp_name):
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'dqn_Atari', datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S_")+exp_name)
    logger.configure(dir=data_dir)
    env = make_atari(env_name)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq_n_step.wrap_atari_dqn(env)

    model = deepq_n_step.learn(
        env,
        "conv_only",
        seed=seed,
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        total_timesteps=int(1e7),
        n_step=n_step,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
    )

    model.save('pong_model.pkl')
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='dqn_n_step')
    parser.add_argument('--n_step', type=int, default=1)
    args = parser.parse_args()
    main(env_name=args.env_name, seed=args.seed, n_step=args.n_step, exp_name=args.exp_name)
