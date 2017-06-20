import gym
import gym_ple
import numpy as np
from Yuanba_li import yunba

env = gym.make('FlappyBird-v0')

observation = env.reset()
ai = yunba()
ai.build()

done = False
play_time = 0

while done==False:
    env.render()
    action = ai.action(observation)
    observation, reward, done, _ = env.step(action)
    play_time+=1

print(play_time)

