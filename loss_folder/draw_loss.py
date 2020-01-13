import numpy as np
import matplotlib.pyplot as plt
def get_data(f1,f2,f3,f4):
	"""
 	four folder loss
	"""
	epoches = []
	train_word = []
	test_word = []
	train_char = []
	test_char = []
	with open(f1,"r") as f:
		for line in f:
			tmp = line.rstrip("\n\r").split("\t")
			train_word.append(float(tmp[1])*100)
			epoches.append(tmp[0])
		
	with open(f2,"r") as f:
		for line in f:
			tmp = line.rstrip("\n\r").split("\t")
			test_word.append(float(tmp[1])*100)
	with open(f3,"r") as f:
		for line in f:
			tmp = line.rstrip("\n\r").split("\t")
			train_char.append(float(tmp[1])*100)

	with open(f4,"r") as f:
		for line in f:
			tmp = line.rstrip("\n\r").split("\t")
			test_char.append(float(tmp[1])*100)

	return epoches, train_word, test_word, train_char, test_char
def reduce_array(array):
	tmp = []
	tmp.append(array[0])
	for i,j in enumerate(array):
		if i%9 == 9:
			tmp.append(j)

	print(tmp)

def draw(epoches, train_word, test_word, train_char, test_char):
	plt.plot(epoches,train_word, label = 'train_word')
	plt.plot(epoches,test_word, label = 'test_word')
	plt.plot(epoches,train_char, label = 'train_char')
	plt.plot(epoches,test_char, label = 'test_char')
	plt.legend(loc='best')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()
	plt.xticks(np.arange(1,len(train_word),step=10))
	plt.savefig('loss_13_1.png')	


if __name__ == "__main__":

	epoches = []
	train_word = []
	test_word = []
	train_char = []
	test_char = []
	epoches, train_word, test_word, train_char, test_char = get_data("./word/train.txt","./word/test.txt","./spelling/train.txt","./spelling/test.txt")	
	draw(epoches, train_word, test_word, train_char, test_char)
	
