
# coding=utf-8

import json
# from  core_nlp.tokenization.crf_tokenizer import *

data = []
with open('data_.jsonl', 'r') as json_data:
    for f in json_data:
    	# tmp = f['raw'].split()
    	data.append(json.loads(f))    
# i = 0

for d in data:
	d['raw'] = d['raw'].split()
	d['original'] = d['original'].split()
	# print(d)
# # print(data[1])
# # from pyvi import ViTokenizer, ViPosTagger
# # a = "Trường đại học bách khoa hà nội"
# # b = ViTokenizer.tokenize(a)
# # print(b.split(" "))
# # print(a)

# sign = 'tr\u01b0\u1eddng'

# # sign = [u'tr\u01b0\u1eddng']
# for i in sign:
# 	print(i)
# 	# if i == 'à':
# 	# print(i)
# print(sign[1])