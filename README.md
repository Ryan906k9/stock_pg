# 强化学习算法实现自动炒股

本文利用强化学习算法 PG，来对股票市场的指数进行交易研究

感兴趣的朋友可以在这个基础上导入其他市场的数据，添加 observation 的维度（本文仅使用了“当天收盘价”和“与前一天收盘价的差值”两个维度）

添加多个维度的数据，再对多个股票进行算法训练，一定会使得该模型更具有鲁棒性，希望大家多多尝试

百度深度学习算法库 PARL ，以及搭建股票测试环境的 gym-anytrading 网址附在下面，用起来感觉不错的可以去 github 上给她们点 star

另外附上本文的 github 地址：https://github.com/Ryan906k9/stock_pg

# 环境依赖

paddlepaddle

parl，网址：https://github.com/PaddlePaddle/PARL

gym

gym-anytrading，网址：https://github.com/AminHP/gym-anytrading

# 提醒：
数据和方法皆来源于网络，无法保证有效性，仅供学习和研究！
