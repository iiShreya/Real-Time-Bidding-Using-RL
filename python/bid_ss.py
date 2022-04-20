import config
from ss_mdp import SS_MDP
from mcpc import Mcpc
from lin_bid import Lin_Bid
from rlb_dp_i import RLB_DP_I

import os
import _pickle as pickle

import time
import numpy as np
import tensorflow as tf
import math


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def getTime():
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def write_log(log_path, line, echo=False):
	with open(log_path, "a") as log_in:
		log_in.write(line + "\n")
		if echo:
			print(line)


def activate(act_func, x):
	if act_func == 'tanh':
		return tf.tanh(x)
	elif act_func == 'relu':
		return tf.nn.relu(x)
	else:
		return tf.sigmoid(x)


def activate_calc(act_func, x):
	if act_func == "tanh":
		return np.tanh(x)
	elif act_func == "relu":
		return max(0, x)
	else:
		return sigmoid(x)


def init_var_map(init_path, _vars):
	if init_path:
		var_map = pickle.load(open(init_path, "rb"))
	else:
		var_map = {}

	for i in range(len(_vars)):
		key, shape, init_method, init_argv = _vars[i]
		if key not in var_map.keys():
			if init_method == "normal":
				mean, dev, seed = init_argv
				var_map[key] = tf.random_normal(shape, mean, dev, seed=seed)
			elif init_method == "uniform":
				min_val, max_val, seed = init_argv
				var_map[key] = tf.random_uniform(shape, min_val, max_val, seed=seed)
			else:
				var_map[key] = tf.zeros(shape)

	return var_map


def build_optimizer(opt_argv, loss):
	opt_method = opt_argv[0]
	if opt_method == 'adam':
		_learning_rate, _epsilon = opt_argv[1:3]
		opt = tf.train.AdamOptimizer(learning_rate=_learning_rate, epsilon=_epsilon).minimize(loss)
	elif opt_method == 'ftrl':
		_learning_rate = opt_argv[1]
		opt = tf.train.FtrlOptimizer(learning_rate=_learning_rate).minimize(loss)
	else:
		_learning_rate = opt_argv[1]
		opt = tf.train.GradientDescentOptimizer(learning_rate=_learning_rate).minimize(loss)
	return opt


# obj_type: clk, profit, imp
class Opt_Obj:
	def __init__(self, obj_type="clk", clk_v=500):
		self.obj_type = obj_type
		self.clk_v = clk_v
		if obj_type == "clk":
			self.v1 = 1
			self.v0 = 0
			self.w = 0
		elif obj_type == "profit":
			self.v1 = clk_v
			self.v0 = 1
			self.w = 0
		else:
			self.v1 = 0
			self.v0 = 0
			self.w = 1

	def get_obj(self, imp, clk, cost):
		return self.v1 * clk - self.v0 * cost + self.w * imp


def calc_m_pdf(m_counter, laplace=1):
	m_pdf = [0] * len(m_counter)
	sum = 0
	for i in range(0, len(m_counter)):
		sum += m_counter[i]
	for i in range(0, len(m_counter)):
		m_pdf[i] = (m_counter[i] + laplace) / (
			sum + len(m_counter) * laplace)
	return m_pdf


def str_list2float_list(str_list):
	res = []
	for _str in str_list:
		res.append(float(_str))
	return res


obj_type = "clk"
clk_vp = 1
N = 1000
c0 = 1 / 32
gamma = 1

src = "ipinyou"

log_in = open(config.projectPath + "bid-performance/{}_N={}_c0={}_obj={}_clkvp={}.txt".format(src, N, c0, obj_type, clk_vp), "w")
print("logs in {}".format(log_in.name))
log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>9}\t {:>8}\t {:>8}"\
	.format("setting", "objective", "auction", "impression", "click", "cost", "win-rate", "CPM", "eCPC")
print(log)
log_in.write(log + "\n")

if src == "ipinyou":
	camps = config.ipinyou_camps
	data_path = config.ipinyouPath
	max_market_price = config.ipinyou_max_market_price
elif src == "vlion":
	camps = config.vlion_camps
	data_path = config.vlionPath
	max_market_price = config.vlion_max_market_price
elif src == "yoyi":
	camps = config.yoyi_camps
	data_path = config.yoyiPath
	max_market_price = config.yoyi_max_market_price

for camp in camps:
	camp_info = config.get_camp_info(camp, src)
	auction_in = open(data_path + camp + "/test.theta.txt", "r")
	opt_obj = Opt_Obj(obj_type, int(clk_vp * camp_info["cost_train"] / camp_info["clk_train"]))
	B = int(camp_info["cost_train"] / camp_info["imp_train"] * c0 * N)
	
	m_pdf = calc_m_pdf(camp_info["price_counter_train"])
	rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
	rlb_dp_i.calc_optimal_value_function_with_approximation_i(N, B, max_market_price, m_pdf, 
			data_path + camp + "/bid-model/v_nb_N={}.txt".format(N))

	# SS-MDP
	auction_in = open(data_path + camp + "/test.theta.txt", "r")
	ss_mdp = SS_MDP(camp_info)
	setting = "{}, camp={}, algo={}, N={}, c0={}" \
		.format(src, camp, "ss_mdp", N, c0)
	bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

	model_path = data_path + camp + "/bid-model/v_nb_N={}.txt".format(N)
	
	ss_mdp.load_value_function(N, B, model_path)
	(auction, imp, clk, cost) = ss_mdp.run(auction_in, bid_log_path, N, c0,
	                                       max_market_price, delimiter=" ", save_log=False)
	
	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	# Mcpc
	auction_in = open(data_path + camp + "/test.theta.txt", "r")
	mcpc = Mcpc(camp_info)
	setting = "{}, camp={}, algo={}, N={}, c0={}"\
		.format(src, camp, "mcpc", N, c0)
	bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
	(auction, imp, clk, cost) = mcpc.run(auction_in, bid_log_path, N, c0,
	                                     max_market_price, delimiter=" ", save_log=False)
	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")
	
	# Lin-Bid
	auction_in = open(data_path + camp + "/test.theta.txt", "r")
	lin_bid = Lin_Bid(camp_info)
	setting = "{}, camp={}, algo={}, N={}, c0={}" \
		.format(src, camp, "lin_bid", N, c0)
	bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)
	model_path = data_path + camp + "/bid-model/{}_{}_{}_{}_{}.pickle".format("lin-bid", N, c0, obj_type, opt_obj.clk_v)
	valid_path = data_path + camp + "/train.theta.txt"
	lin_bid.parameter_tune(opt_obj, valid_path, model_path, N, c0, max_market_price,
	                       max_market_price, delimiter=" ", load=True)
	(auction, imp, clk, cost) = lin_bid.run(auction_in, bid_log_path, N, c0,
	                                        max_market_price, delimiter=" ", save_log=False)
	
	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

	# RLB
	auction_in = open(data_path + camp + "/test.theta.txt", "r")
	rlb_dp_i = RLB_DP_I(camp_info, opt_obj, gamma)
	setting = "{}, camp={}, algo={}, N={}, c0={}" \
		.format(src, camp, "rlb", N, c0)
	bid_log_path = config.projectPath + "bid-log/{}.txt".format(setting)

	model_path = data_path + camp + "/bid-model/v_nb_N={}.txt".format(N)
	rlb_dp_i.load_value_function(N, B, model_path)

	(auction, imp, clk, cost) = rlb_dp_i.run(auction_in, bid_log_path, N, c0,
	                                         max_market_price, delimiter=" ", save_log=False)

	win_rate = imp / auction * 100
	cpm = (cost / 1000) / imp * 1000
	ecpc = (cost / 1000) / clk
	obj = opt_obj.get_obj(imp, clk, cost)
	log = "{:<55}\t {:>10}\t {:>8}\t {:>10}\t {:>8}\t {:>8}\t {:>8.2f}%\t {:>8.2f}\t {:>8.2f}" \
		.format(setting, obj, auction, imp, clk, cost, win_rate, cpm, ecpc)
	print(log)
	log_in.write(log + "\n")

log_in.flush()
log_in.close()