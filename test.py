import gym
import numpy as np

'''
observation 是 numpy 结构 512*288*3的图像，unit8
action 的值 为0或者1,0是上，1是不动

'''


env = gym.make("Pong-v0")
observation = env.reset()

done = False
play_time = 0

while done==False:
    env.render()
    #action = ai.action(observation)
    action = 1
    observation, reward, done, _ = env.step(action)
    play_time += 1

print(play_time)

