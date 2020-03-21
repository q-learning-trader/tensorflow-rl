import shutil
from collections import deque

import tensorflow as tf
import numpy as np
from IPython.display import clear_output

import base

def output(x, num):
    x = tf.keras.layers.Dense(512, "elu")(x)
    out = [tf.keras.layers.Dense(2)(x) for _ in range(num)]
    out = tf.keras.layers.concatenate(out)

    return tf.reshape(out, (-1, 2, num))


def build_model(n=200, dim=(30, 4)):
    inputs = tf.keras.layers.Input(dim)

    x = base.bese_net(inputs)

    out = output(x, n)

    return tf.keras.Model(inputs, out)

class Agent(base.Base_Agent):
    def build(self):
        self.types = "DQN"
        self.gamma = 0.2
        self.epsilon = 0.05

        n = 200
        self.tau = np.array([i / n for i in range(n)])

        if self.restore:
            self.i = np.load("qrdqn_epoch.npy")
            self.model = tf.keras.models.load_model("qrdqn.h5")
            self.target_model = tf.keras.models.load_model("qrdqn.h5")
        else:
            self.model = build_model(n)
            self.model.compile("nadam", "mse")
            self.target_model = build_model(n)
            self.target_model.set_weights(self.model.get_weights())

    def loss(self, states, new_states, rewards, actions):
        q = self.model.predict_on_batch(states)
        target_q = self.target_model.predict_on_batch(new_states).numpy()
        arg_q = np.sum(self.model.predict_on_batch(new_states), -1).reshape((-1,2))
        arg_q = [np.argmax(i) if 0.05 < np.random.rand() else np.random.randint(2) for i in arg_q]

        q_backup = q.numpy()

        for i in range(q.shape[0]):
            q_backup[i, actions[i]] = rewards[i] + self.gamma * target_q[i, arg_q[i]]


        error = q_backup - q
        q_error = tf.maximum(self.tau * error, (self.tau - 1) * error)
        loss = tf.where(q_error < 2, q_error ** 2 * .5, 2 * q_error - 0.5 * 2 ** 2)

        return tf.reduce_mean(tf.reduce_sum(loss, 2), 1)

    def sample(self, memory):
        states = np.array([a[0] for a in memory], np.float32)
        new_states = np.array([a[3] for a in memory], np.float32)
        actions = np.array([a[1] for a in memory]).reshape((-1, 1))
        rewards = np.array([a[2] for a in memory], np.float32).reshape((-1, 1))

        return self.loss(states, new_states, rewards, actions).numpy()

    def train(self):
        tree_idx, replay = self.memory.sample(128)

        states = np.array([a[0][0] for a in replay], np.float32)
        new_states = np.array([a[0][3] for a in replay], np.float32)
        actions = np.array([a[0][1] for a in replay]).reshape((-1, 1))
        rewards = np.array([a[0][2] for a in replay], np.float32).reshape((-1, 1))

        with tf.GradientTape() as tape:
            error = self.loss(states, new_states, rewards, actions)
            loss = tf.reduce_mean(error)

        ae = error.numpy().reshape((-1,)) + 1e-10
        self.memory.batch_update(tree_idx, ae)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [(tf.clip_by_value(grad, -10.0, 10.0))
        #              for grad in gradients]
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def lr_decay(self, i):
        lr = self.lr * 0.0001 ** (i / 10000000)
        self.model.optimizer.lr.assign(lr)

    def policy(self, state, i):
        epsilon = self.epsilon + (1 - self.epsilon) * (np.exp(-0.0001 * i))
        q = np.sum(self.model.predict_on_batch(state), -1)
        q = np.abs(q) / np.sum(np.abs(q), 1).reshape((-1, 1)) * (np.abs(q) / q)

        if (i + 1) % 5 != 0:
            epsilon = epsilon if self.random % 5 != 0 else 1.
            q += epsilon * np.random.randn(q.shape[0], q.shape[1])
            action = np.argmax(q, 1)
            self.random += 1
        else:
            action = np.argmax(q, -1)

        return action

    def save(self, i):
        self.restore = True
        self.i = i
        self.model.save("qrdqn.h5")
        np.save("qrdqn_epoch", i)
        _ = shutil.copy("/content/qrdqn.h5", "/content/drive/My Drive")
        _ = shutil.copy("/content/qrdqn_epoch.npy", "/content/drive/My Drive")