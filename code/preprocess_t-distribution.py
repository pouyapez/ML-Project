import numpy as np
import random

output = np.load("data/out.npy")

SNR = 0.5011 # -3: 1.413, -2: 1.259, -1: 1.122, 0: 1, 1: 0.8913, 2: 0.7943, 3: 0.708, 4: 0.631, 5: 0.5623, 6: 0.5011
snr = 6
df = 3
input = []
train_out = []
test_out = []

print len(output[0])

for i in range(len(output)): # produce convolutional decoder and add t-distribution noise 
	S = [0, 0]
	c = [[0, 0]]
	inp = []
	tr_out = []
	te_out = []
	for j in range(len(output[0])):
		c[0][0] = 2*output[i][j]-1
		c[0][1] = 2*np.absolute(output[i][j]-S[0])-1
		inp += c 
		S = [np.absolute(np.absolute(output[i][j]-S[0])-S[1]), S[0]]
		tr_out += [[c[0][0]+np.random.standard_t(df, size=1), c[0][1]+np.random.standard_t(df, size=1)]] 
		te_out += [[c[0][0]+np.random.standard_t(df, size=1), c[0][1]+np.random.standard_t(df, size=1)]]
	input += [inp]
	train_out += [tr_out]
	test_out += [te_out]

np.save("data/train_out_G_"+str(snr)+".npy", train_out)
np.save("data/test_out_G_"+str(snr)+".npy", test_out)
np.save("data/input.npy", input)
