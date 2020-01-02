
#!/usr/bin/python

# -*- coding: utf8 -*-

# # a = "Trường đại học bách khoa hà nội"
# # b = ViTokenizer.tokenize(a)
# # print(b.split(" "))
# # print(a)

# sign = 'tr\u01b0\u1eddng'

# # # sign = [u'tr\u01b0\u1eddng']
# # for i in sign:
# # 	print(i)
# # 	# if i == 'à':
# # 	# print(i)
# print(sign)





# data = []
# with open('data_.jsonl', 'r') as json_data:
#     for f in json_data:
#     	data.append(json.loads(f)) 

# with open('data_book.json', 'w', encoding='utf-8') as outfile:
# 	for f in data:
# 		tmp = f['id'].split("_")
# 		if(tmp[0] == "BOOK"):
# 			json.dump(f, outfile, ensure_ascii=False)
# 			outfile.write('\n')

# with open('data_book.json', 'w') as outfile:
# 	for f in data:
# 		tmp = f['id'].split("_")
# 		if(tmp[0] == "BOOK"):
# 			json.dump(f, outfile)
# 			outfile.write('\n')
# 			print(f)


# data_book = []

# with open('data_book.json', 'r') as json_data:
#     for f in json_data:
#     	data_book.append(json.loads(f)) 

# train, test = train_test_split(data_book, test_size = 0.2)

# with open('train_book.json', 'w') as outfile:
# 	for f in train:
# 		json.dump(f, outfile, ensure_ascii=False)
# 		outfile.write('\n')

import json
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer, ViPosTagger
import random
import re
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
def test_length():
	path = './data/'
	# test length cau
	with open(path+"train_data.json", 'r') as json_data:
		data = json.load(json_data)
	max_ = 0
	for i in data:
		if max_ < len(i['original']):
			max_ = len(i['original'])
	print(max_) 
	# 68 + 2 = 70
	# 65 + 2 = 67 =>>>>>>>69
	# test length tu
	with open(path+"test_data.json", 'r') as json_data:
		data = json.load(json_data)
	max_ = 0
	for i in data:
		tmp1 = i['raw']
		tmp2 = i['original']
		for j in tmp1:
			if max_ < len(j):
				max_ = len(j)
				print(i['id'])
		for k in tmp2:
			if max_ <len(k):
				max_len(k)
				print(i['id'])
	print(max_)
	# 47 + 2 = 49
	# 23 + 2 = 25  ========> 49
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

	with open(path+'train_tiny.json','r') as outfile:
		data1 = json.load(outfile)
	element_random = random.sample(range(len(data1)), number)

	for i in element_random:

		print('original  ')
		print(data1[i]['original'])
		print(data1[i]['id'])
		print('edit raw  ')
		print(data1[i]['raw'])
    
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
    sign_['ă'] = ['à', 'á', 'â', 'ã', 'ạ', 'ả', 'ấ', 'ầ', 'ậ', 'a', 'ặ']
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
def add_noise(word):
	#i = random.randint(0, len(word) -1 )
	#op = random.randint(0,30)
	op = 8
	i  = 1
	if op == 0:
		return word[:i] + word[i+1:]

	if op == 1:
		i += 1
		return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]

	if op == 2:
		i+=1
		return word[:i] + '_'+ word[i:] 

	if op == 3:
		idx = word.find("_")
		if idx != -1:
			return word[:i] + word[i+1:];
	if op == 4:
		return word[:i] + random.choice(get_repleace_character(word[i])) + word[i+1:]
		# thieu truong hop, dau vao 2 ki tu
	if op == 5:
		return word[:i] + random.choise(get_change_sign(word[i])) + word[i+1:]
		# da du
	if op == 6:
		#  truong hop, dau vao 2 
		if i <= len(word) - 2:
			return word[:i] + random.choise(exchange_2_character(word[i]+word[i+1])) + word[i+2:]
	if op == 7:
		# truong hop dau vao 3 
		if i <= len(word) - 3:
			return word[:i] + random.choise(exchange_3_character(word[i]+word[i+1]+word[i+2])) + word[i+3:]
	if op == 8:
		# truong hop 4 ki tu, dau vao
		if i <= len(word) - 4:
			return word[:i] + exchange_4_character(word[i]+word[i+1]+word[i+2] + word[i+3]) + word[i+4:]
	# ban phim
	return word[:i] + get_prox_keys(word[i]) + word[i+1:]


def add_noise_sequen(data):
	"""
		chay ham nay khi da chay filter_number_d_underscore
	"""
	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news,json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
	path = './data/'
	data1 = []
	with open(path+'train_tiny.json','r') as outfile:
		data1 = json.load(outfile)

	for i in len(data1):
		





if __name__ == '__main__':


	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news.json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json','train_tiny.json', 'test_tiny.json']
	path = './data/'
	data1 = []
	data2 = []
	# split_token_json(path+c[0],data1,data2)
	# split_token_json(f2,data1,data2)
	# split_token_json(f3,data1,data2)
	# split_token_json(f4,data1,data2)
	# split_token_json(f5,data1,data2)
	# split_token_json(f6,data1,data2)



	# fillter_number_d_underscore(data1) # chi filter data chuan 
	# with open(path+f[5], 'w') as outfile:
	# 		json.dump(data1, outfile, ensure_ascii=False)


			# outfile.write(',\n')
	# with open(path+f[5], 'w') as outfile:
	# 		json.dump(data2, outfile, ensure_ascii=False)   

	# result()
	random_10_sequence(10)
	#ghv = get_phonology_vietnamese('')
	#gs = random.choice(get_change_sign('u'))
	#gc = random.choice(get_repleace_character('ch'))
	#print(ghv)
	#print(gs)
	#print(gc)

	#aaa = add_noise('liêng')
	#print(aaa)

	#aaa = 'aaaa'

	#print(aaaa[4])
	# regex = re.compile('^([0-9]+|[])$')
	# matches = re.search(regex,'g80')
	# print(matches)
	# data = []
	# with open('./word_model/'+'valid.pred', 'r') as outfile:
	# 	data.append(json.loads(outfile))
	# print(data)

	# data = []
	# data2 = []
	# save_file_json(path+"test_book.json",data,data2)
	# with open(path+"test_book_standart.json", 'w') as outfile:
	# 	json.dump(data, outfile, ensure_ascii=False)


	# import codecs
	# import re
	# a = '1_'
	# b = "b\u1edfi_v\u1eady"

	# regex = re.compile(r'\u00E0\u00E1\u00E2\u00E3\u00E8\u00E9\u00EA\u00EC\u00ED\u00F2\u00F3\u00F4\u00F5\u00F9\u00FA\u00FD\u00E5\u0111\u0123\u0169\u01A1\u01B0\u1EA1\u1EA3\u1EA5\u1EA7\u1EA9\u1EAB\u1EAD\u1EAF\u1EB1\u1EB3\u1EB5\u1EB7\u1EB9\u1EBB\u1EBD\u1EBF\u1EC1\u1EC3\u1EC5\u1EC7\u1EC9\u1ECB\u1ECD\u1ECF\u1ED1\u1ED3\u1ED5\u1ED7\u1ED9\u1EDB\u1EDD\u1EDF\u1EE1\u1EE3\u1EE5\u1EE7\u1EE9\u1EEB\u1EED\u1EEF\u1EF1\u1EF3\u1EF5\u1EF7\u1EF9abcdefghiklmnopqrstuvxy0123456789_')
	
	# b = u'b\u1edfi_v\u1eady'
	# regex  = re.compile('^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$')
	# matches = re.search(regex,b)
	# c  = 'b\xe1\xbb\x9fi_v\xe1\xba\xady'
	# # c.encode("windows-1252").decode("utf-8")
	# c1 =  unicode(c, "utf-8")
	# print(c1)
	# print(matches.group(0))
	# print(len(matches.group(0)))
	# print(len(b))
	# print(u'ph\u1ea3i')
	# print(len(list(matches)))
	# if match is None:
	# 	print("FALSE")
	# else :
	# 	print("TRUE")
	# print(b)
	# c = u'b\u1edfi_v\u1eady'
	# for i in c:
	# 	print(i)
	# b = codecs.decode(a,'utf-8')
	# print(type(b))
	# print(a.isalnum())
	# print(len(a))
	# for i in a:
	# 	print(i)
	# # print(a[0])
				

	# b = '\u00E0\u00E1\u00E2\u00E3\u00E8\u00E9\u00EA\u00EC\u00ED\u00F2\u00F3\u00F4\u00F5\u00F9\u00FA\u00FD\u00E5\u0111\u0123\u0169\u01A1\u01B0\u1EA1\u1EA3\u1EA5\u1EA7\u1EA9\u1EAB\u1EAD\u1EAF\u1EB1\u1EB3\u1EB5\u1EB7\u1EB9\u1EBB\u1EBD\u1EBF\u1EC1\u1EC3\u1EC5\u1EC7\u1EC9\u1ECB\u1ECD\u1ECF\u1ED1\u1ED3\u1ED5\u1ED7\u1ED9\u1EDB\u1EDD\u1EDF\u1EE1\u1EE3\u1EE5\u1EE7\u1EE9\u1EEB\u1EED\u1EEF\u1EF1\u1EF3\u1EF5\u1EF7\u1EF9abcdefghiklmnopqrstuvxy0123456789_'
	# c = '[\u00E0-\u00ED]'
	# print('U+0041-U+005A'.encode('utf-8').encode('hex'))
	# result = b.decode('utf8')


	# a = "aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789"
	# print(len(a))

	# test_dict = {}
	# print(test_dict)
	# if not test_dict:
	#     print ("Dict is Empty")


	# if not bool(test_dict):
	#     print ("Dict is Empty")


	# if len(test_dict) == 0:
	#     print ("Dict is Empty")


	# import random 
	  
	# # Generates a random number between 
	# # a given positive range 
	# r1 = random.randint(0, 10) 
	# print("Random number between 0 and 10 is % s" % (r1)) 
	# print(u'm\u1edbn')


	# a = 2
	# if a == 2 or a == 3:
	# 	print('manh')