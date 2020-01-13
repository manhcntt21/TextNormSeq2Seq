
#!/usr/bin/python

# -*- coding: utf8 -*-

import json
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer, ViPosTagger
import random
import re
import copy 
from sklearn.utils import shuffle

def convert(fn1,fn2,fn3):
	"""
		loc cac item rieng ra tung loai
	"""
	data = []
	with open(fn1, 'r') as json_data:
	    for f in json_data:
	    	data.append(json.loads(f))

	with open(fn2, 'w') as outfile:
		for f in data:
			tmp = f['id'].split("_")
			if(tmp[0] == fn3):
				json.dump(f, outfile, ensure_ascii=False)
				outfile.write('\n')
def split_train_test(fn1,fn2,fn3):
	"""
		tach ra train vs test theo ti le 8 :2
		ghi train vao f2
		test f3
	"""
	data = []
	with open(fn1, 'r') as json_data:
	    for f in json_data:
	    	data.append(json.loads(f))

	train, test = train_test_split(data, test_size = 0.2)
	with open(fn2, 'w') as outfile:
		for f in train:
			json.dump(f, outfile, ensure_ascii=False)
			outfile.write('\n')

	with open(fn3, 'w') as outfile:
		for f in test:
			json.dump(f, outfile, ensure_ascii=False)
			outfile.write('\n')
def split_token_json(f,data1,data2):
	"""
		tach tu 
		data1= la du lieu chuan
		data2 la violation
	"""
	with open(f, 'r') as json_data:
	    for f in json_data:
	    	tmp = json.loads(f)
	    	x = tmp['raw']
	    	y = ViTokenizer.tokenize(x)
	    	y = y.split(" ")
	    	tmp['raw'] = y
	    	x1 = tmp['original']
	    	y1 = ViTokenizer.tokenize(x1)
	    	y1 = y1.split(" ")
	    	tmp['original'] = y1
	    	tmp.update({'tid':0})
	    	if len(y) == len(y1):
	    		data1.append(tmp)
	    	else:
	    		data2.append(tmp)
def merge_data_save(f1,f2,f3,f4,f5,f6,f7,f8):
	"""
		luu vao f7, f8
		data1 - train
		data2- violation
	"""
	data1 = []
	data2 = []
	split_token_json(f1,data1,data2)
	split_token_json(f2,data1,data2)
	split_token_json(f3,data1,data2)
	split_token_json(f4,data1,data2)
	split_token_json(f5,data1,data2)
	split_token_json(f6,data1,data2)

	fillter_number_d_underscore(data1) # chi filter data chuan
	add_noise_sequen(data1) # add_noise
	data1 = shuffle(data1) # shuffle data
	with open(f7, 'w') as outfile:
			json.dump(data1, outfile, ensure_ascii=False)
			# outfile.write(',\n')
	with open(f8, 'w') as outfile:
			json.dump(data2, outfile, ensure_ascii=False)   		    	
def result():
	data = 'data_.jsonl'
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news,json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
	path = './data/'
	for i in range(len(a)):
		convert(data,path+a[i],d[i])
		split_train_test(path+a[i],path+b[i],path+c[i])

	merge_data_save(path+b[0],path+b[1],path+b[2],path+b[3],path+b[4],path+b[5],path+f[0],path+f[2]) # train
	merge_data_save(path+c[0],path+c[1],path+c[2],path+c[3],path+c[4],path+c[5],path+f[1],path+f[3]) # test
def test_length(f1,f2):
	path = './data/'
	# test length cau
	with open(path+f1, 'r') as json_data:
		data = json.load(json_data)
	max_ = 0
	for i in data:
		if max_ < len(i['original']):
			max_ = len(i['original'])
	print(max_) 
	print('len train', len(data))
	# 68 + 2 = 70
	# 65 + 2 = 67 =>>>>>>>69
	# test length tu

	with open(path+f2, 'r') as json_data:
		data1 = json.load(json_data)
	max_ = 0
	for i in data1:
		tmp1 = i['raw']
		tmp2 = i['original']
		for j in tmp1:
			if max_ < len(j):
				max_ = len(j)
				# print(i['id'])
		for k in tmp2:
			if max_ <len(k):
				max_len(k)
				# print(i['id'])
	print(max_)
	print('len test ' , len(data1))
	# 47 + 4 = 
	# 23 + 4 = 25  ========> 
def random_10_sequence(number):
	"""
		test
		comment number
		delete underscore
	"""
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news,json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
	path = './data/'

	regex = re.compile('\S*\d+\S*',re.UNICODE)
	regex1 = re.compile('\S*_\S*',re.UNICODE)
	data1 = [] 

	# split_token_json(path+a[0],data1,data2)
	# split_token_json(path+a[1],data1,data2)
	# # split_token_json(path+a[2],data1,data2)

	with open(path+'train_data.json','r') as outfile:
		data1 = json.load(outfile)
	element_random = random.sample(range(len(data1)), number)

	for i in element_random: 	
		print('original  ')
		print(data1[i]['original'])
		print(data1[i]['id'])
		print('edit raw  ')
		print(data1[i]['raw'])
	#for i in range(len(data1)):
	#	if data1[i]['id'] == 'CALS_00039470':
	#		print(data1[i])

def fillter_number_d_underscore(data1):
	"""
		loc string co chua so:
		vd: so dien thoai
	"""
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news.json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
	path = './data/'
	# data1 = [] ## train
	# data2 = [] ## test 
	regex1 = re.compile("\S*\d+\S*",re.UNICODE)
	regex = re.compile('\S*_\S*',re.UNICODE)
	# split_token_json(path+a[0],data1,data2)
	# split_token_json(path+a[1],data1,data2)

	for i in range(len(data1)):
		undersocre = {}
		delete_undersocre = random.random()
		for j, k in enumerate(data1[i]['original']):
			if re.search(regex1, k):
				data1[i]['raw'][j] = '@' +data1[i]['raw'][j]
				data1[i]['original'][j] = '@' +data1[i]['original'][j]
			elif re.search(regex, k):
				index = []
				for m,n in enumerate(k):   # tach tahnh ky tu
					if n == '_':
						index.append(m)
				if len(index) != 0:
					undersocre[j] = index
		if bool(undersocre):
			# print(undersocre)
			tmp = random.choice(list(undersocre.keys())) # chon ngau nhien mot tu trong cau
			# print('word duoc chon = ',tmp)		
			# print('delete_undersocre = ', delete_undersocre)
			if delete_undersocre > 0.65 :
				if len(undersocre[tmp]) == 1:
					index_underscore = random.choice(undersocre[tmp])  # chon ngau nhien 1 dau cach trong cau
					# data1[i]['original'][tmp] = data1[i]['original'][tmp][:index_underscore] + data1[i]['original'][tmp][index_underscore+1:]
					data1[i]['raw'][tmp] = data1[i]['raw'][tmp][:index_underscore] + data1[i]['raw'][tmp][index_underscore+1:]
				else:	
					n_index = random.randrange(1,len(undersocre[tmp]),1)
					for u in range(n_index):
						index_underscore = random.choice(undersocre[tmp])  # chon ngau nhien 1 dau cach trong cau
						# print(data1[i]['original'][tmp][index_underscore])
						# data1[i]['original'][tmp] = data1[i]['original'][tmp][:index_underscore] + data1[i]['original'][tmp][index_underscore+1:]
						data1[i]['raw'][tmp] = data1[i]['raw'][tmp][:index_underscore] + data1[i]['raw'][tmp][index_underscore+1:]
	# with open(path+'test.json', 'w') as outfile:
	# 		json.dump(data1, outfile, ensure_ascii=False)
def get_change_sign(word):
    """
        # sign['a'] =  # a à á â ã ạ ả ấ  ầ  ậ ắ  ặ 
        # sign['e'] =  # è é ê  ẹ ẻ ẽ ế  ề  ể  ễ  ệ 
        # sign['i'] =  # ì í ỉ ị 
        # sign['o'] =  # ò ó ô  õ ọ ỏ  ố ồ  ổ ộ ớ ờ ỡ ợ 
        # sign['u'] =  # ù ú ụ ủ ứ ừ ữ ự 
        # sign['y'] =  # ý ỳ ỵ ỷ 
    """
    sign_ = {}
    sign_['a'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['à'] = ['a', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['á'] = ['à', 'a', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['â'] = ['à', 'á', 'a', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['ã'] = ['à', 'á', 'â', 'a', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['ạ'] = ['à', 'á', 'â', 'ã', 'a', 'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['ả'] = ['à', 'á', 'â', 'ã', 'ạ', 'a', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['ấ'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'a', 'ầ', 'ậ', 'ắ', 'ặ']
    sign_['ầ'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'a', 'ậ', 'ắ', 'ặ']
    sign_['ậ'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'a', 'ắ', 'ặ']
    sign_['ắ'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'a', 'ặ']
    sign_['ặ'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'a', 'a']
    sign_['e'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['è'] = ['e', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['é'] = ['è', 'e', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ê'] = ['è', 'é', 'e', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ẹ'] = ['è', 'é', 'ê', 'e', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ẻ'] = ['è', 'é', 'ê', 'ẹ', 'e', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ẽ'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'e', 'ế', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ế'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'e', 'ề', 'ể', 'ễ', 'ệ']
    sign_['ề'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'e', 'ể', 'ễ', 'ệ']
    sign_['ể'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'e', 'ễ', 'ệ']
    sign_['ễ'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'e', 'ệ']
    sign_['ệ'] = ['è', 'é', 'ê', 'ẹ', 'ẻ', 'ẽ', 'ế', 'ề', 'ể', 'ễ', 'e']
    sign_['i'] = ['ì', 'í', 'ỉ', 'ị']
    sign_['ì'] = ['i', 'í', 'ỉ', 'ị']
    sign_['í'] = ['ì', 'i', 'ỉ', 'ị']
    sign_['ỉ'] = ['ì', 'í', 'i', 'ị']
    sign_['ị'] = ['ì', 'í', 'ỉ', 'i']
    sign_['o'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ò'] = ['o', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ó'] = ['ò', 'o', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ô'] = ['ò', 'ó', 'o', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['õ'] = ['ò', 'ó', 'ô', 'o', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ọ'] = ['ò', 'ó', 'ô', 'õ', 'o', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ỏ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'o', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ố'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'o', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ồ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'o', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ổ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'o', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ộ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'o', 'ớ', 'ờ', 'ỡ', 'ợ']
    sign_['ớ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'o', 'ờ', 'ỡ', 'ợ']
    sign_['ờ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'o', 'ỡ', 'ợ']
    sign_['ỡ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'o', 'ợ']
    sign_['ợ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'o']
    sign_['u'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự']
    sign_['ù'] = ['u', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự']
    sign_['ú'] = ['ù', 'u', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự']
    sign_['ụ'] = ['ù', 'ú', 'u', 'ủ', 'ứ', 'ừ', 'ữ', 'ự']
    sign_['ủ'] = ['ù', 'ú', 'ụ', 'u', 'ứ', 'ừ', 'ữ', 'ự']
    sign_['ứ'] = ['ù', 'ú', 'ụ', 'ủ', 'u', 'ừ', 'ữ', 'ự']
    sign_['ừ'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'u', 'ữ', 'ự']
    sign_['ữ'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'u', 'ự']
    sign_['ự'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'u']
    sign_['y'] = ['ý', 'ỳ', 'ỵ', 'ỷ']
    sign_['ý'] = ['y', 'ỳ', 'ỵ', 'ỷ']
    sign_['ỳ'] = ['ý', 'y', 'ỵ', 'ỷ']
    sign_['ỵ'] = ['ý', 'ỳ', 'y', 'ỷ']
    sign_['ỷ'] = ['ý', 'ỳ', 'ỵ', 'y']
    sign_['_'] = ['_']
    try:
        return sign_[word]
    except:
        return word
def get_repleace_character(word):
    repleace_character = {}


    repleace_character['l'] = ['n']
    repleace_character['n'] = ['l']
    repleace_character['x'] = ['s']
    repleace_character['s'] = ['x']
    repleace_character['r'] = ['d', "gi"]
    repleace_character['d'] = ['r', "gi"]
    
    repleace_character['c'] = ['q', 'k']
    repleace_character['k'] = ['q', 'c']
    repleace_character['q'] = ['c', 'k']
    repleace_character['i'] = ['y']
    repleace_character['y'] = ['i']
    repleace_character['_'] = ['_']
    try:
        return repleace_character[word]
    except:
        return word
def exchange_2_character(word):

	"""
		truong hop dau vao 2 ki tu
		photology + replace_character
	"""
	exchange_2_character = {}
	exchange_2_character['ch'] = ['tr']
	exchange_2_character['tr'] = ['ch']
	exchange_2_character['gi'] = ['d', 'r']
	exchange_2_character['ăc'] = ['ắt']
	exchange_2_character['ac'] = ['at']
	exchange_2_character['âc'] = ['ât']
	exchange_2_character['ưc'] = ['ưt']
	exchange_2_character['ôc'] = ['ôt']
	exchange_2_character['uc'] = ['ut']
	try:
		return exchange_2_character[word]
	except:
		return word
def exchange_3_character(word):
	"""
		truong hop dau vao 3 ki tu
		photology
	"""
	exchange_3_character = {}
	exchange_3_character['inh'] = 'in'
	exchange_3_character['ênh'] = 'ên'
	exchange_3_character['êch'] = 'ết'
	exchange_3_character['ich'] = 'ít'
	exchange_3_character['ăng'] = 'ăn'
	exchange_3_character['ang'] = 'an'
	exchange_3_character['âng'] = 'ân'
	exchange_3_character['ưng'] = 'ưn'
	exchange_3_character['ông'] = 'ôn'
	exchange_3_character['ung'] = 'un'
	exchange_3_character['iêc'] = 'iêt'
	exchange_3_character['ước'] = 'ươt'
	exchange_3_character['uôc'] = 'uôt'
	try:
		return exchange_3_character[word]
	except:
		return word
def exchange_4_character(word):

	"""
		truong hop dau vao 3 ki tu
		photology
	"""
	exchange_4_character = {}
	exchange_4_character['iêng'] = 'iên'
	exchange_4_character['ương'] = 'ươn'
	exchange_4_character['uông'] = 'uôn'
	try:
		return exchange_4_character[word]
	except:
		return word     
def get_prox_keys(word):
    array_prox = {}
    array_prox['a'] = ['q', 'w', 'z', 'x', 's']
    array_prox['b'] = ['v', 'f', 'g', 'h', 'n', ' ']
    array_prox['c'] = ['x', 's', 'd', 'f', 'v']
    array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c']
    array_prox['e'] = ['w', 's', 'd', 'f', 'r']
    array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v']
    array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'y', 'h', 'n']
    array_prox['h'] = ['b', 'g', 't', 'y', 'u', 'j', 'm', 'n']
    array_prox['i'] = ['u', 'j', 'k', 'l', 'o']
    array_prox['j'] = ['n', 'h', 'y', 'u', 'i', 'k', 'm']
    array_prox['k'] = ['u', 'j', 'm', 'l', 'o']
    array_prox['l'] = ['p', 'o', 'i', 'k', 'm']
    array_prox['m'] = ['n', 'h', 'j', 'k', 'l']
    array_prox['n'] = ['b', 'g', 'h', 'j', 'm']
    array_prox['o'] = ['i', 'k', 'l', 'p']
    array_prox['p'] = ['o', 'l']
    array_prox['q'] = ['w', 'a']
    array_prox['r'] = ['e', 'd', 'f', 'g', 't']
    array_prox['s'] = ['q', 'w', 'e', 'z', 'x', 'c']
    array_prox['t'] = ['r', 'f', 'g', 'h', 'y']
    array_prox['u'] = ['y', 'h', 'j', 'k', 'i']
    array_prox['v'] = ['', 'c', 'd', 'f', 'g', 'b']
    array_prox['w'] = ['q', 'a', 's', 'd', 'e']
    array_prox['x'] = ['z', 'a', 's', 'd', 'c']
    array_prox['y'] = ['t', 'g', 'h', 'j', 'u']
    array_prox['z'] = ['x', 's', 'a']
    array_prox['1'] = ['q', 'w']
    array_prox['2'] = ['q', 'w', 'e']
    array_prox['3'] = ['w', 'e', 'r']
    array_prox['4'] = ['e', 'r', 't']
    array_prox['5'] = ['r', 't', 'y']
    array_prox['6'] = ['t', 'y', 'u']
    array_prox['7'] = ['y', 'u', 'i']
    array_prox['8'] = ['u', 'i', 'o']
    array_prox['9'] = ['i', 'o', 'p']
    array_prox['0'] = ['o', 'p']
    array_prox['_'] = ['_']
    try:
        return array_prox[word]
    except:
        return word
def noise_telex(word):
	#2
	noise_tele['à'] = 'af'
	noise_tele['á'] = 'as'
	noise_tele['â'] = 'aa'
	noise_tele['ã'] = 'ax'
	noise_tele['ạ'] = 'aj'
	noise_tele['ả'] = 'ar'
	#3
	noise_tele['ấ'] = 'aas'
	noise_tele['ầ'] = 'aaf'
	noise_tele['ẩ'] = 'aar'
	noise_tele['ậ'] = 'aaj'
	noise_tele['ắ'] = 'aws'
	noise_tele['ặ'] = 'awj'
	#2
	noise_tele['è'] = 'ef'
	noise_tele['é'] = 'es'
	noise_tele['ê'] = 'ee'
	noise_tele['ẽ'] = 'ex'
	noise_tele['ẹ'] = 'ej'
	noise_tele['ẻ'] = 'ar'
	#3
	noise_tele['ế'] = 'ees'
	noise_tele['ề'] = 'eef'
	noise_tele['ể'] = 'eer'
	noise_tele['ệ'] = 'eej'
	noise_tele['ễ'] = 'eex'
	#2
	noise_tele['ì'] = 'if'
	noise_tele['í'] = 'is'
	noise_tele['ỉ'] = 'ir'
	noise_tele['ị'] = 'ij'
	#2
	noise_tele['ò'] = 'of'
	noise_tele['ó'] = 'os'
	noise_tele['ô'] = 'ô'
	noise_tele['õ'] = 'ox'
	noise_tele['ọ'] = 'oj'
	noise_tele['ỏ'] = 'or'
	noise_tele['ơ'] = 'ow'
	#3
	noise_tele['ố'] = 'oos'
	noise_tele['ồ'] = 'oof'
	noise_tele['ổ'] = 'oor'
	noise_tele['ộ'] = 'ooj'
	noise_tele['ớ'] = 'ows'
	noise_tele['ờ'] = 'owf'
	noise_tele['ỡ'] = 'owx'
	noise_tele['ợ'] = 'owj'
	#2
	noise_tele['ù'] = 'uf'
	noise_tele['ú'] = 'us'
	noise_tele['ư'] = 'uw'
	noise_tele['ũ'] = 'ux'
	noise_tele['ụ'] = 'uj'
	noise_tele['ủ'] = 'ur'
	#3
	noise_tele['ứ'] = 'uws'
	noise_tele['ừ'] = 'uwf'
	noise_tele['ữ'] = 'uwx'
	noise_tele['ự'] = 'uwj'
	#2
	noise_tele['ý'] = 'ys'
	noise_tele['ỳ'] = 'yf'
	noise_tele['ỵ'] = 'yj'
	noise_tele['ỷ'] = 'yr'

	noise_tele['_'] = '_'
	try:
		return array_prox[word]
	except:
		return word
def add_noise(word,op):
	i = random.randint(0, len(word) -1 )
	#op = random.randint(0,30)
	# i  = 1
	if op == 0:
		return word[:i] + word[i+1:]

	if op == 1:
		i += 1
		return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]

	if op == 2:
		i+=1
		if i <= len(word) - 1:
			return word[:i] + '_'+ word[i:] 

	if op == 3:
		idx = word.find("_")
		if idx != -1:
			return word[:idx] + word[idx+1:];
	if op == 4:
		return word[:i] + random.choice(get_repleace_character(word[i])) + word[i+1:]
		# thieu truong hop, dau vao 2 ki tu
	if op == 5:
		return word[:i] + random.choice(get_change_sign(word[i])) + word[i+1:]
		# da du
	if op == 6:
		#  truong hop, dau vao 2 
		if i <= len(word) - 2:
			return word[:i] + random.choice(exchange_2_character(word[i]+word[i+1])) + word[i+2:]
	if op == 7:
		# truong hop dau vao 3 
		if i <= len(word) - 3:
			return word[:i] + random.choice(exchange_3_character(word[i]+word[i+1]+word[i+2])) + word[i+3:]
	if op == 8:
		# truong hop 4 ki tu, dau vao
		if i <= len(word) - 4:
			return word[:i] + exchange_4_character(word[i]+word[i+1]+word[i+2] + word[i+3]) + word[i+4:]
	# ban phim
	try:
		return word[:i] + random.choice(get_prox_keys(word[i])) + word[i+1:]
	except	:
		return word
def add_noise_sequen(data1):
	"""
		chay ham nay khi da chay filter_number_d_underscore
	"""
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news,json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
	path = './data/'
	# data1 = []
	data2 = [] # noise tao ra
	regex1 = re.compile("\S*\d+\S*",re.UNICODE)
	# with open(path+'train_tiny.json','r') as outfile:
	# 	data1 = json.load(outfile)
	# co 9 loai loi
	error = [0,1,2,3,4,5,6,7,8,9]
	sequence = [1,2,3]
	index = 0
	for i in range(len(data1)):
		m = random.random()
		# print('-------------------------------1-----------------------------------')
		# print(data1[i]['raw'])
		# print(data1[i]['original'])
		if m > 0.3: # xac suat chon cau de them nhieu
			# print('before = ',data1[i]['id'])
			n_quence = random.choice(sequence)
			for j in range(n_quence):
				n_error = random.randint(0,30)*len(data1[i]['original'])/100
				tmp = copy.deepcopy(data1[i])
				data2.append(tmp)
				# print(j)
				data2[index]['id'] = data2[index]['id'] + str(j) 
				# print('after = ',data2[index]['id'])
				for j1 in range(int(n_error)):
					# print(len(data1[i]['original']))
					n = random.randint(0, len(data2[index]['original']) - 1)
					word = data2[index]['original'][n]
					if not re.search(regex1,word): # so thi loai
						op = random.choice(error)
						word = add_noise(word,op)
						data2[index]['raw'][n] = word
				# print('<<<<noise>>>>>')
				# print(data2[index]['raw'])
				# print(data2[index]['original'])
				index+=1
		# print('-------------------------------2-----------------------------------')

	for i in range(len(data2)):
		data1.append(data2[i])
	# with open(path + fileout, 'w') as outfile:
	# 	json.dump(data1, outfile, ensure_ascii=False)

def create_tiny():
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news.json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json','train_tiny.json', 'test_tiny.json']
	path = './data/'
	data1 = []
	data2 = []
	data3 = []
	data4 = []
	split_token_json(path+b[0],data1,data2)
	fillter_number_d_underscore(data1) # chi filter data chuan 
	add_noise_sequen(data1)
	with open(path+f[4], 'w') as outfile:
			json.dump(data1, outfile, ensure_ascii=False)

	split_token_json(path+c[0],data3,data4)
	fillter_number_d_underscore(data3) # chi filter data chuan 
	add_noise_sequen(data3)
	with open(path+f[5], 'w') as outfile:
			json.dump(data3, outfile, ensure_ascii=False)

def isUrl(word):
	"""
		loc url
	"""
	regex = re.compile(
	r'^(?:http|ftp)s?://' # http:// or https://
	r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
	r'localhost|' #localhost...
	r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
	r'(?::\d+)?' # optional port
	r'(?:/?|[/?]\S+)$', re.IGNORECASE)
	match = re.match(regex, token)
	if match is None:
		return False
	else:
		return True


if __name__ == '__main__':

	# create_tiny()
	#result()
	# random_10_sequence(10)
	# test_length('train_data.json','test_data.json')
	
	# print(len('ấ')) ## = 1