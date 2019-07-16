from baselines.deepq_n_step import models  # noqa
from baselines.deepq_n_step.build_graph import build_act, build_train  # noqa
from baselines.deepq_n_step.deepq import learn, load_act  # noqa
from baselines.deepq_n_step.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
