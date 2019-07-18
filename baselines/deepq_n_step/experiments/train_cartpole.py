import gym
import os.path as osp
import datetime
import argparse
from baselines import logger

from baselines import deepq_n_step


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main(env_name, seed, n_step, buffer_size, exp_name):
    data_dir = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))),
                        'openai_dqn_classic', datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S_") + exp_name)
    logger.configure(dir=data_dir)
    env = gym.make(env_name)
    act = deepq_n_step.learn(
        env,
        network='mlp',
        seed=seed,
        lr=1e-3,
        total_timesteps=100000,
        n_step=n_step,
        buffer_size=buffer_size,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default="CartPole-v0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='dqn_n_step')
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--buffer_size', type=int, default=50000)
    args = parser.parse_args()
    main(env_name=args.env_name, seed=args.seed,
         n_step=args.n_step, buffer_size=args.buffer_size,
         exp_name=args.exp_name)
