import tensorflow as tf
import numpy as np
from six.moves import xrange

def discount_episode_rewards(rewards=[], gamma=0.99, mode=0):

    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(xrange(0, rewards.size)):
        if mode == 0:
            if rewards[t] != 0: running_add = 0

        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add
    return discounted_r


'''
def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    try: # TF 1.0
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=actions, logits=logits, name=name)
        cross_entropy = tf.losses.log_loss(labels=actions,predictions=logits)
    except:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, targets=actions)
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, actions)

    try: ## TF1.0
        loss = tf.reduce_sum(tf.multiply(cross_entropy, rewards))
    except: ## TF0.12
        loss = tf.reduce_sum(tf.mul(cross_entropy, rewards))   # element-wise mul
    return loss
'''

def cross_entropy_reward_loss(logits, actions, rewards, name=None):
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=actions)
    cross_entropy = actions*tf.log(logits)+(1-actions)*tf.log(1-logits)
    loss = 0-tf.reduce_sum(tf.multiply(cross_entropy, rewards))
    return loss

def loss(logits, actions):
    loss = 0-tf.reduce_sum(actions * tf.log(logits) + (1 - actions) * tf.log(1 - logits))
    return loss
