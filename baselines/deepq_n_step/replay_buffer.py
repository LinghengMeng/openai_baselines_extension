import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from baselines import logger
import csv
import os

# class ReplayBuffer(object):
#     def __init__(self, size, obs_dim):
#         """Create Replay buffer.
#
#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0
#         self.obs_dim = obs_dim
#
#     def __len__(self):
#         return len(self._storage)
#
#     def add(self, obs_t, action, reward, obs_tp1, done):
#         data = (obs_t, action, reward, obs_tp1, done)
#
#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize
#
#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#             dones.append(done)
#         return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
#
#     def sample(self, batch_size):
#         """Sample a batch of experiences.
#
#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.
#
#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         """
#         idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes)
#
#     def sample_batch_n_step(self, batch_size=32, n_step=1, task_type='Atari'):
#         """
#         return training batch for n-step experiences
#         :param batch_size:
#         :param n_step:
#         :return: dict:
#             'obs1': batch_size x n_step x obs_dim
#             'obs2': batch_size x n_step x obs_dim
#             'acts': batch_size x n_step x act_dim
#             'rews': batch_size x n_step
#             'done': batch_size x n_step
#         """
#         idxes = np.random.randint(0, len(self._storage) -(n_step-1), size=batch_size)
#
#         if len(np.asarray(self._storage[0][0]).shape)>2:
#             task_type = 'Atari'
#         else:
#             task_type = 'Classic'
#         if task_type == 'Atari':
#             batch_obs1 = np.zeros([batch_size, n_step, 84, 84, 4])
#             batch_obs2 = np.zeros([batch_size, n_step, 84, 84, 4])
#             batch_acts = np.zeros([batch_size, n_step])
#             batch_rews = np.zeros([batch_size, n_step])
#             batch_done = np.zeros([batch_size, n_step])
#             for i in range(n_step):
#                 obs, act, rew, next_obs, done = self._encode_sample(idxes+i)
#                 batch_obs1[:, i, :, :, :] = obs
#                 batch_obs2[:, i, :, :, :] = next_obs
#                 batch_acts[:, i] = act
#                 batch_rews[:, i] = rew
#                 batch_done[:, i] = done
#             # Set all done after the fist met one to 1
#             done_index = np.asarray(np.where(batch_done==1))
#             for d_i in range(done_index.shape[1]):
#                 x, y=done_index[:, d_i]
#                 batch_done[x, y:] = 1
#
#             return batch_obs1[:,0,:,:,:], batch_acts[:,0], batch_rews, batch_obs2[:,-1,:,:,:], batch_done
#         else:
#             batch_obs1 = np.zeros([batch_size, n_step, self.obs_dim])
#             batch_obs2 = np.zeros([batch_size, n_step, self.obs_dim])
#             batch_acts = np.zeros([batch_size, n_step])
#             batch_rews = np.zeros([batch_size, n_step])
#             batch_done = np.zeros([batch_size, n_step])
#             for i in range(n_step):
#                 obs, act, rew, next_obs, done = self._encode_sample(idxes + i)
#                 batch_obs1[:, i, :] = obs
#                 batch_obs2[:, i, :] = next_obs
#                 batch_acts[:, i] = act
#                 batch_rews[:, i] = rew
#                 batch_done[:, i] = done
#             # Set all done after the fist met one to 1
#             done_index = np.asarray(np.where(batch_done == 1))
#             for d_i in range(done_index.shape[1]):
#                 x, y = done_index[:, d_i]
#                 batch_done[x, y:] = 1
#
#             return batch_obs1[:, 0, :], batch_acts[:, 0], batch_rews, batch_obs2[:, -1, :], batch_done

class experience_logger(object):
    def __init__(self, obs_dim, act_dim, file_name='experiences.csv'):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.file_name = file_name
        self.file_path = os.path.join(logger.get_dir(), self.file_name)
        with open('{}'.format(self.file_path), 'a', newline='\n') as csvfile:
            self.field_names = []
            for o_i in range(obs_dim):
                self.field_names.append('obs_{}'.format(o_i))
            for a_i in range(act_dim):
                self.field_names.append('act_{}'.format(a_i))
            self.field_names.append('reward')
            for o_i in range(obs_dim):
                self.field_names.append('obs_new_{}'.format(o_i))
            self.field_names.append('done')
            self.exp_writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            self.exp_writer.writeheader()

    def log_experience(self, obs, act, reward, obs_new, done):
        with open('{}'.format(self.file_path), 'a', newline='\n') as csvfile:
            self.exp_writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            experience = {}
            for obs_i, obs_v in enumerate(np.array(obs).flatten()):
                experience['obs_{}'.format(obs_i)] = obs_v
            for act_i, act_v in enumerate(np.array(act).flatten()):
                experience['act_{}'.format(act_i)] = act_v
            experience['reward'] = reward
            for obs_new_i, obs_new_v in enumerate(np.array(obs_new).flatten()):
                experience['obs_new_{}'.format(obs_new_i)] = obs_new_v
            experience['done'] = done
            self.exp_writer.writerow(experience)

class ReplayBuffer(object):
    def __init__(self, size, obs_dim, act_dim):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.experience_logger = experience_logger(self.obs_dim, self.act_dim)

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        self.experience_logger.log_experience(obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def sample_batch_n_step(self, batch_size=32, n_step=1, task_type='Atari'):
        """
        return training batch for n-step experiences
        :param batch_size:
        :param n_step:
        :return: dict:
            'obs1': batch_size x n_step x obs_dim
            'obs2': batch_size x n_step x obs_dim
            'acts': batch_size x n_step x act_dim
            'rews': batch_size x n_step
            'done': batch_size x n_step
        """
        idxes = np.random.randint(0, len(self._storage) -(n_step-1), size=batch_size)

        if len(np.asarray(self._storage[0][0]).shape)>2:
            task_type = 'Atari'
        else:
            task_type = 'Classic'
        if task_type == 'Atari':
            batch_obs1 = np.zeros([batch_size, n_step, 84, 84, 4])
            batch_obs2 = np.zeros([batch_size, n_step, 84, 84, 4])
            batch_acts = np.zeros([batch_size, n_step])
            batch_rews = np.zeros([batch_size, n_step])
            batch_done = np.zeros([batch_size, n_step])
            for i in range(n_step):
                obs, act, rew, next_obs, done = self._encode_sample(idxes+i)
                batch_obs1[:, i, :, :, :] = obs
                batch_obs2[:, i, :, :, :] = next_obs
                batch_acts[:, i] = act
                batch_rews[:, i] = rew
                batch_done[:, i] = done
            # Set all done after the fist met one to 1
            done_index = np.asarray(np.where(batch_done==1))
            for d_i in range(done_index.shape[1]):
                x, y=done_index[:, d_i]
                batch_done[x, y:] = 1

            return batch_obs1[:,0,:,:,:], batch_acts[:,0], batch_rews, batch_obs2[:,-1,:,:,:], batch_done
        else:
            batch_obs1 = np.zeros([batch_size, n_step, self.obs_dim])
            batch_obs2 = np.zeros([batch_size, n_step, self.obs_dim])
            batch_acts = np.zeros([batch_size, n_step])
            batch_rews = np.zeros([batch_size, n_step])
            batch_done = np.zeros([batch_size, n_step])
            for i in range(n_step):
                obs, act, rew, next_obs, done = self._encode_sample(idxes + i)
                batch_obs1[:, i, :] = obs
                batch_obs2[:, i, :] = next_obs
                batch_acts[:, i] = act
                batch_rews[:, i] = rew
                batch_done[:, i] = done
            # Set all done after the fist met one to 1
            done_index = np.asarray(np.where(batch_done == 1))
            for d_i in range(done_index.shape[1]):
                x, y = done_index[:, d_i]
                batch_done[x, y:] = 1

            return batch_obs1[:, 0, :], batch_acts[:, 0], batch_rews, batch_obs2[:, -1, :], batch_done

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
