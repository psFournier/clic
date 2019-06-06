import tensorflow as tf
from gym import wrappers
import pkg_resources
import keras.backend as K
import numpy as np
import datetime
import os
import json
from utils.logger import Logger


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def wrap_gym(env,render,dir):
    if not render:
        env = wrappers.Monitor(
            env, dir, video_callable=False, force=True)
    else:
        env = wrappers.Monitor(env, dir, force=True)
    return env

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

def build_logger(args):
    param_strings = [args['--agent'], args['--env']]
    now = datetime.datetime.now()
    log_dir = os.path.join(args['--log_dir'], '_'.join(param_strings), now.strftime("%Y%m%d%H%M%S_%f"))
    os.makedirs(log_dir, exist_ok=True)
    args['--time'] = now
    print(args)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args, default=str))
    logger = Logger(dir=os.path.join(log_dir,'log_steps'),
                         format_strs=['json', 'tensorboard_{}'.format(int(args['--eval_freq'])),
                                      'stdout'.format(int(args['--eval_freq']))])
    return logger

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)

def kl_divergence(x, y):
    x = (x+1)/2
    y = (y+1)/2
    x = np.clip(x, K.epsilon(), 1)
    y = np.clip(y, K.epsilon(), 1)
    aux = x * np.log(x / y)
    return np.sum(aux, axis=0)

def huber_loss(y_true, y_pred, delta_clip):
    err = y_true - y_pred
    L2 = 0.5 * K.square(err)

    # Deal separately with infinite delta (=no clipping)
    if np.isinf(delta_clip):
        return K.mean(L2)

    cond = K.abs(err) < delta_clip
    L1 = delta_clip * (K.abs(err) - 0.5 * delta_clip)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p