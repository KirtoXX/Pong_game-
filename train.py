import tensorflow as tf
import tensorlayer as tl
import rein
import gym
import gym_ple
import numpy as np
import AI

observation = None
prev_x = None
running_reward = None

#---------define the placeholder--------
# image shape is 512,288,3
#convert shape is 64,36,1

states_batch_pl = tf.placeholder(tf.float32, shape=[None,64,36,1])
actions_batch_pl = tf.placeholder(tf.float32, shape=[None,1])
discount_rewards_batch_pl = tf.placeholder(tf.float32, shape=[None,1])


#---------define the loss and train_op--------
gamma = 0.9
decay_rate = 0.9

#network = AI.policy_network(states_batch_pl)
sampling_prob = AI.policy_network(states_batch_pl)
loss = rein.cross_entropy_reward_loss(sampling_prob, actions_batch_pl,discount_rewards_batch_pl)
logloss = rein.loss(logits=sampling_prob,actions=actions_batch_pl)

train_op = tf.train.RMSPropOptimizer(learning_rate=0.1,decay=0.9).minimize(loss)

#---------train---------
env = gym.make('FlappyBird-v0')

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    update_batch = 70   #when times > update_batch update my weights

    play_times = 1500
    i = 0

    for i in range(play_times):
        # trunk
        xs, ys, rs = [], [], []
        # init env
        observation = env.reset()
        reward_sum = 0
        done = False
        times = 0        #total live times in one game

        #-------play game-------
        while done==False:
            x = AI.preprocessing(observation)
            prob = sess.run(sampling_prob,feed_dict={states_batch_pl: x})  # do action

            #action = np.random.choice([0,1],prob)
            action = 1 if np.random.uniform()<prob else 0

            observation, reward, done, _ = env.step(action)
            xs.append(x)
            ys.append(action)
            rs.append(reward)
            times+=1

        #--------update weights------
        if times > update_batch:
            #-------buid the input-------
            epx = np.vstack(xs)
            epy = np.asarray(ys)
            epr = np.asarray(rs)
            disR = tl.rein.discount_episode_rewards(epr, gamma)
            disR -= np.mean(disR)
            disR /= np.std(disR)
            epy = epy.reshape(-1,1)
            disR = disR.reshape(-1,1)

            # --------loss and reward---------
            reward_sum = np.sum(epr)
            loss1 = sess.run(logloss,feed_dict={states_batch_pl: epx, actions_batch_pl: epy})

            #-------updata the weight-------
            sess.run(train_op,feed_dict={states_batch_pl: epx,actions_batch_pl: epy,discount_rewards_batch_pl: disR})
            print('game:',i,' times: ',times,' total loss is:',loss1,' total reward is: ',reward_sum)


#tl.files.save_npz(network.all_params , name='weight/20170407.npz')























