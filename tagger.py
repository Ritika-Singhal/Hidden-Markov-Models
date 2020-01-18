import numpy as np

from util import accuracy
from hmm import HMM

def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	obs_dict = {}
	state_dict = dict(zip(tags, np.where(tags)[0].tolist()))

	N = len(tags)
	pi = np.zeros((N))
	A = np.zeros((N,N))
	B = np.zeros((N,1))

	line_start = np.array([])
	obs_keys = np.array(list(obs_dict.keys()))

	for line in train_data[:100]:
		line_words = line.words
		words_list = np.unique(np.array(line_words))
		new_keys = np.setdiff1d(words_list, obs_keys)
		obs_keys = np.append(obs_keys, new_keys)
		obs_dict = dict(zip(obs_keys, np.where(obs_keys)[0].tolist()))

		if len(B[0]) == 1:
			B = np.zeros((B.shape[0],len(new_keys)))
		else:
			B = np.hstack((B,np.zeros((B.shape[0],len(new_keys)))))

		line_start = np.append(line_start, line.tags[0])
		for i in range(len(line.tags)):
			if i != len(line.tags)-1:
				A[state_dict[line.tags[i]], state_dict[line.tags[i+1]]] += 1
			B[state_dict[line.tags[i]], obs_dict[line_words[i]]] += 1

	A = (A.T/np.sum(A, axis=1)).T
	B = (B.T/np.sum(B, axis=1)).T

	line_start_tags, count = np.unique(line_start, return_counts=True)
	for i in range(len(line_start_tags)):
		pi[state_dict[line_start_tags[i]]] = count[i]/100.0

	model = HMM(pi, A, B, obs_dict, state_dict)
	return model



def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []

	obs_keys = np.array(list(model.obs_dict.keys()))
	for line in test_data:
		new_keys = np.setdiff1d(np.unique(line.words), obs_keys)
		if len(new_keys) != 0:
			obs_keys = np.append(obs_keys, new_keys)
			model.obs_dict = dict(zip(obs_keys, np.where(obs_keys)[0].tolist()))
			model.B = np.hstack((model.B, np.full((model.B.shape[0], len(new_keys)), 10**-6)))

		tagging.append(model.viterbi(line.words))

	return tagging
