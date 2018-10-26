import argparse
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import SingleDeviceEnv

import dqn
from dqn_utils import *

def model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def device_optimizer():
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )

def stop_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return env.done or t >= num_timesteps 
    return stopping_criterion

def exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def kwargs():
    return {
        'optimizer_spec': device_optimizer(),
        'q_func': model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def learn(env, session, num_timesteps, seed):

    optimizer = device_optimizer()
    stopping_criterion = stop_criterion(num_timesteps)
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

    dqn.learn(
        env=env,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stop_criterion(num_timesteps),
        double_q=False,
        **kwargs()
    )

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count={'GPU': 0})
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    return session

def get_env(seed):
    env = SingleDeviceEnv.get_random_env()
    set_global_seeds(seed)

    return env

def main():
    # Run training
    seed = 4565 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(seed)
    session = get_session()
    set_global_seeds(seed)
    learn(env, session, num_timesteps=1e4, seed=seed)

if __name__ == "__main__":
    main()
