from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
import numpy as np
from keras.losses import mse
import tensorflow as tf
from utils.util import softmax
import time

class Critic2(object):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.gamma = 0.99
        self.lr_imit = float(args['--lrimit'])
        self.w_i = float(args['--wimit'])
        self.margin = float(args['--margin'])
        self.network = float(args['--network'])
        self.filter = int(args['--filter'])
        self.layers = [int(l) for l in args['--layers'].split(',')]
        self.num_actions = env.action_dim
        self.num_tasks = env.N
        self.stats = {}
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        ### Inputs
        S = Input(shape=self.s_dim)
        TASKS = Input(shape=(1,), dtype='int32')
        ACTIONS = Input(shape=(1,), dtype='int32')
        TARGETS = Input(shape=(1,))
        # MCR = Input(shape=(1,), dtype='float32')

        ### Q values model
        qvals = self.create_critic_network(S)
        self.model = Model([S], qvals)

        batch_size = K.shape(qvals)[0]
        full_shape = tf.stack([batch_size, self.num_tasks, self.num_actions])
        qvals_reshaped = K.reshape(qvals, full_shape)
        self.qvals = K.function(inputs=[S], outputs=[qvals_reshaped], updates=None)
        # prepare row indices
        row_indices = tf.range(batch_size)[...,tf.newaxis]
        # zip row indices with column indices
        full_indices = tf.concat([row_indices, TASKS, ACTIONS], axis=1)
        # retrieve values by indices
        qval = tf.gather_nd(qvals_reshaped, full_indices)[...,tf.newaxis]

        self.qval = K.function(inputs=[S, TASKS, ACTIONS], outputs=[qval], updates=None)

        ### DQN Training
        td_errors = qval - TARGETS
        l2errors = K.square(td_errors)
        loss_dqn = K.mean(l2errors, axis=0)
        inputs_dqn = [S, TASKS, ACTIONS, TARGETS]
        updates_dqn = Adam(lr=0.001).get_updates(loss_dqn, self.model.trainable_weights)
        self.metrics_dqn_names = ['loss_dqn', 'qval']
        metrics_dqn = [loss_dqn, qval, td_errors]
        self.train = K.function(inputs_dqn, metrics_dqn, updates_dqn)

        ### Large margin loss
        task_indices = tf.concat([row_indices, TASKS], axis=1)
        qvals_per_task = tf.gather_nd(qvals_reshaped, task_indices)
        # qvalWidth = K.max(qvals, axis=1, keepdims=True) - K.min(qvals, axis=1, keepdims=True)
        onehot = 1 - K.squeeze(K.one_hot(ACTIONS, self.num_actions), axis=1)
        # onehotMargin = K.repeat_elements(self.margin, self.num_actions, axis=1) * onehot
        imit = (K.max(qvals_per_task + self.margin * onehot, axis=1, keepdims=True) - qval)

        ### Suboptimal demos
        # val = K.max(qvals_per_task, axis=1, keepdims=True)
        # advClip = K.cast(K.greater(MCR, val), dtype='float32')
        # goodexp = K.sum(advClip)

        ### Imitation
        # if self.filter == 0:
        loss_imit = K.mean(imit, axis=0)
        loss_dqn_imit = K.mean(l2errors, axis=0)
        # elif self.filter == 1:
        #     loss_imit = K.mean(imit * advClip, axis=0)
        #     loss_dqn_imit = K.mean(l2errors, axis=0)
        # else:
        #     loss_imit = K.mean(imit * advClip, axis=0)
        #     loss_dqn_imit = K.mean(l2errors * advClip, axis=0)
        loss = loss_dqn_imit + self.w_i * loss_imit
        inputs_imit = [S, TASKS, ACTIONS, TARGETS]
        self.metrics_imit_names = ['loss_imit', 'loss_dqn_imit', 'qvals', 'goodexp']
        metrics_imit = [loss_imit, loss_dqn_imit, qvals]
        updates_imit = Adam(lr=self.lr_imit).get_updates(loss, self.model.trainable_weights)
        self.imit = K.function(inputs_imit, metrics_imit, updates_imit)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        targetqvals = self.create_critic_network(S)
        self.targetmodel = Model([S], targetqvals)

        TASKS = Input(shape=(1,), dtype='int32')
        ACTIONS = Input(shape=(1,), dtype='int32')
        batch_size = K.shape(targetqvals)[0]
        full_shape = tf.stack([batch_size, self.num_tasks, self.num_actions])
        targetqvals_reshaped = K.reshape(targetqvals, full_shape)
        row_indices = tf.range(batch_size)[..., tf.newaxis]
        full_indices = tf.concat([row_indices, TASKS, ACTIONS], axis=1)
        targetqval = tf.gather_nd(targetqvals_reshaped, full_indices)[..., tf.newaxis]
        self.targetqval = K.function(inputs=[S, TASKS, ACTIONS], outputs=[targetqval], updates=None)

        # self.target_train()

    # def target_train(self):
    #     weights = self.model.get_weights()
    #     target_weights = self.targetmodel.get_weights()
    #     for i in range(len(weights)):
    #         target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
    #     self.targetmodel.set_weights(target_weights)

    def target_train(self):
        self.targetmodel.set_weights(self.model.get_weights())

    def get_targets_dqn(self, s, task, r, t):
        qvals = self.qvals([s])[0]
        batchsize, _, numactions = qvals.shape
        qvals_for_task = qvals[np.arange(batchsize)[:, np.newaxis], task, np.arange(numactions)]
        actions = np.argmax(qvals_for_task, axis=1)
        # probs = softmax(qvals_for_task, theta=1, axis=1)
        # t1 = time.time()
        # self.stats['traintime2'] += t1 - t0
        # t0 = t1
        #
        # actions = [np.random.choice(range(self.env.action_dim), p=prob) for prob in probs]
        a1 = np.expand_dims(np.array(actions), axis=1)
        q = self.targetqval([s, task, a1])[0]
        targets = r + (1 - t) * self.gamma * q.squeeze()
        targets = np.clip(targets, self.env.rNotTerm / (1 - self.env.gamma), self.env.rTerm)
        return np.expand_dims(targets, axis=1)

    def create_critic_network(self, S):
        h = S
        for l in self.layers:
            h = Dense(l, activation="relu",
                      kernel_initializer=lecun_uniform(),
                      kernel_regularizer=l2(0.01))(h)
        Q_values = Dense(self.num_actions * self.num_tasks,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h)
        return Q_values