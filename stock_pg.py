import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger

import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL


#from parl.algorithms import PolicyGradient


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm

        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_prob = self.model(obs)  # 获取输出动作概率
        # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        log_prob = layers.reduce_sum(
            -1.0 * layers.log(act_prob) * layers.one_hot(
                action, act_prob.shape[ 1 ]),
            dim=1)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost

LEARNING_RATE = 1e-4

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hidden_dim_1 = hidden_dim_2 = 128

        self.fc1 = layers.fc(size=hidden_dim_1, act='tanh')
        self.fc2 = layers.fc(size=hidden_dim_2, act='tanh')
        self.fc3 = layers.fc(size=act_dim, act="softmax")

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
        #print(act_prob)
        act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作

        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        #print(obs)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        #print(act_prob)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = data_process(obs)
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 1 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(1):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = data_process(obs)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

# 数据预处理
def data_process(obs):
    # 缩放数据尺度
    obs[:,0]/=100
    return obs.astype(np.float).ravel()


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.9):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    #reward_arr -= np.mean(reward_arr)
    #reward_arr /= np.std(reward_arr)
    return reward_arr


# 创建环境

env_test = gym.make('stocks-v0', frame_bound=(1800, 2150), window_size=10)
obs_dim = 20
act_dim = 2
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)


# 加载模型
agent.restore('./stock_pg_v1_2.ckpt')

test_flag = 0 # 是否直接测试


if(test_flag==1):
    for i in range(5000):

        # 每次跟新训练环境
        start = np.random.randint(10,1900)
        env_train = gym.make('stocks-v0', frame_bound=(start, start+100), window_size=10)

        # 每次都是单个环境
        #env_train = gym.make('stocks-v0', frame_bound=(10, 2000), window_size=10)

        obs_list, action_list, reward_list = run_episode(env_train, agent)

        if i % 50 == 0:
            logger.info("Train Episode {}, Reward Sum {}.".format(i,
                                                sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        cost = agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env_test, agent)
            logger.info('Episode {}, Test reward: {}'.format(i + 1,
                                                total_reward))
            # 保存模型
            ckpt = 'stock_pg_v1/steps_{}.ckpt'.format(i)
            agent.save(ckpt)

            #plt.cla()
            #env_test.render_all()
            #plt.show()

    # save the parameters to ./model.ckpt
    agent.save('./stock_pg_v1_2.ckpt')

else:
    # 加载模型
    agent.restore('./stock_pg_v1/steps_4899.ckpt')
    total_reward = evaluate(env_test, agent,render=True)
    logger.info('Test reward: {}'.format(total_reward))
    plt.cla()
    env_test.render_all()
    plt.show()
