from hmm import HMM
import numpy as np
import json
import time
from data_process import Dataset
from util import accuracy
from tagger import model_training, sentence_tagging

def hmm_test():

    st_time = time.time()

    model_file = "hmm_model.json"

    # load data
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    # observation symbols
    obs_dict = data['observations']
    # state symbols
    states_symbols = dict()
    for idx, item in enumerate(data['states']):
        states_symbols[item] = idx
    Osequence = np.array(data['Osequence'])
    N = len(Osequence)
    model = HMM(pi, A, B, obs_dict, states_symbols)

    delta = model.forward(Osequence)
    print("Forward function output:", delta)

    gamma = model.backward(Osequence)
    print("Backward function output:", gamma)

    prob1 = model.sequence_prob(Osequence)
    print("Sequence_prob function output:", prob1)

    prob2 = model.posterior_prob(Osequence)
    print("Posterior_prob function output:", prob2)

    prob3 = model.likelihood_prob(Osequence)
    print("Likelihood_prob function output:", prob3)

    viterbi_path = model.viterbi(Osequence)
    print('Viterbi function output: ', viterbi_path)

    en_time = time.time()
    print()
    print("hmm total time: ", en_time - st_time)


def speech_tagging_test():
    st_time = time.time()
    data = Dataset("pos_tags.txt", "pos_sentences.txt", train_test_split=0.8, seed=0)

    data.train_data = data.train_data[:100]

    data.test_data = data.test_data[:10]

    model = model_training(data.train_data, data.tags)

    tagging = sentence_tagging(data.test_data, model, data.tags)

    total_words = 0
    total_correct = 0
    for i in range(len(tagging)):
        correct, words, accur = accuracy(tagging[i], data.test_data[i].tags)
        total_words += words
        total_correct += correct
        print("accuracy: ", accur)

    print("Total accuracy: ", total_correct*1.0/total_words)

    en_time = time.time()
    print("sentence_tagging total time: ", en_time - st_time)


if __name__ == "__main__":

    hmm_test()

    speech_tagging_test()
