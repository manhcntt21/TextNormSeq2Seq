
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

	with open(path+'train_data.json','r') as outfile:
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

if __name__ == '__main__':


	a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news.json', 'data_stors.json','data_dial2.json']
	b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
	c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
	d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
	f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json','train_tiny.json', 'test_tiny.json']
	path = './data/'
	data1 = []
	data2 = []
	split_token_json(path+c[0],data1,data2)
	# split_token_json(f2,data1,data2)
	# split_token_json(f3,data1,data2)
	# split_token_json(f4,data1,data2)
	# split_token_json(f5,data1,data2)
	# split_token_json(f6,data1,data2)

	fillter_number_d_underscore(data1) # chi filter data chuan 
	with open(path+f[5], 'w') as outfile:
			json.dump(data1, outfile, ensure_ascii=False)
			# outfile.write(',\n')
	# with open(path+f[5], 'w') as outfile:
	# 		json.dump(data2, outfile, ensure_ascii=False)   

	# result()
	# random_10_sequence(10)

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