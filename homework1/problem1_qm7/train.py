import os
import pickle
import sys
import numpy as np
import model as nn
import copy
import scipy
import scipy.io 
import argparse
from tqdm import tqdm
from utils import rotate_3d_coordinates
import matplotlib.pyplot as plt

def train(nnsgd, nnavg, R, Z, E_true, mb, hist, augment):
	rmses = []

	for i in tqdm(range(1,100000)):

		# learning rate schedule
		if i > 0:     lr = 0.0000001  
		if i > 500:   lr = 0.00000025
		if i > 2500:  lr = 0.0000005
		if i > 12500: lr = 0.000001
		lr = lr*0.2

		#sample minibatch indices
		r = np.random.randint(0,len(R),[mb])
		# If augmentation is enabled, rotate each minibatch element independently.
		if augment:

			R_aug = np.empty_like(R[r])

			for j in range(mb):
				# Sample independent rotation angles for x, y, and z between -180 and 180 degrees.
				x_angle = np.random.uniform(-180, 180)
				y_angle = np.random.uniform(-180, 180)
				z_angle = np.random.uniform(-180, 180)
				# Rotate the j-th element.
				R_aug[j] = rotate_3d_coordinates(R[r][j], x_angle, y_angle, z_angle)
			R_batch = R_aug
		else:
			R_batch = R[r]
		E_pred = nnsgd.forward((R_batch, Z[r]))  # Forward pass
		loss = 0.5 * np.sum((E_pred - E_true[r]) ** 2)  # MSE loss

		rmse = np.square(E_pred-E_true[r]).mean(axis=0)**.5
		rmses.append(rmse)
		nnsgd.backward((E_pred- E_true[r])) #TODO: add argument to backward
		nnsgd.update(lr)
		nnavg.average(nnsgd,(1/hist)/((1/hist)+i))
		nnavg.nbiter = i

		if i % 100 == 0: 
			print(f"RMSE: {sum(rmses[-100:])/100} kcal/mol")
			pickle.dump(nnavg,open(f'nn-augment={augment}.pkl','wb'),pickle.HIGHEST_PROTOCOL)
	np.save(f'training-rmses-augment={augment}.npy', np.array(rmses))
	print("Done Training")


if __name__ == '__main__':
	'''Setup'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=123, help='Random seed')
	parser.add_argument('--augmentation', action='store_true', help='whether to perform data augmentations')
	parser.add_argument('--train_fraction', type=float, default=0.5, help='Fraction of training split to use')
	parser.add_argument('--mb', type=int, default=25, help='Batch size')
	parser.add_argument('--hist', type=float, default=0.1, help='Fraction of the history to be remembered for Exponential Moving Average')
	args = parser.parse_args()

	np.random.seed(args.seed)

	'''Load data'''
	if not os.path.exists('qm7.mat'): os.system('wget http://www.quantum-machine.org/data/qm7.mat')
	dataset = scipy.io.loadmat('qm7.mat')
	split_idx = dataset['P'][1:].flatten() #leave first split for testing
	skip_every = int(1/args.train_fraction)
	R = dataset['R'][split_idx][::skip_every]
	Z = dataset['Z'][split_idx][::skip_every]
	E_true = dataset['T'][0,split_idx][::skip_every]

	'''Create neural network'''
	#Define input and output layers
	I,O = nn.Input((R, Z)),nn.Output(E_true)
	nnsgd = nn.Sequential([
		I,
		nn.Linear(92, 400),
		nn.Tanh(),
		nn.Linear(400, 100),
		nn.Tanh(),
		nn.Linear(100, 1),
		O
	])
	nnavg = copy.deepcopy(nnsgd)

	'''Train neural network'''
	train(nnsgd, nnavg, R, Z, E_true, args.mb, args.hist, True)
