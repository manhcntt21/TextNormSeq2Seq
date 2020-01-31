
#!/usr/bin/python

# -*- coding: utf8 -*-

import json
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer, ViPosTagger
import random
import re
import copy
from sklearn.utils import shuffle
import string
import collections
import csv

import numpy as np
random.seed(3255)

file = 'data_.jsonl'
data_by_label_json = ['data_book.json', 'data_cals.json',
                      'data_dial.json', 'data_news.json', 'data_stors.json', 'data_dial2.json']
data_by_label_json_noise = ['data_book_noise.json', 'data_cals_noise.json',
                            'data_dial_noise.json', 'data_news_noise.json', 'data_stors_noise.json', 'data_dial2_noise.json']
data_by_label_json_violation = ['data_book_violation.json', 'data_cals_violation.json',
                                'data_dial_violation.json', 'data_news_violation.json', 'data_stors_violation.json', 'data_dial2_violation.json']


data_by_label_train_json = ['train_book.json', 'train_cals.json',
                            'train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
data_by_label_test_json = ['test_book.json', 'test_cals.json',
                           'test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
data_label = ["BOOK", "CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
final_json = ['train_data.json', 'test_data.json', 'train_data_view.json', 'test_data_view.json', 'tran_violation.json', 'test_violation.json',
              'train_data_tiny.json', 'test_data_tiny.json', 'train_data_view_tiny.json', 'test_data_view_tiny.json']


def convert(fn1, fn2, fn3):
    """
            loc cac item rieng ra tung loai
            va tinh tong luon
    """
    data = []
    with open(fn1, 'r') as json_data:
        for f in json_data:
            data.append(json.loads(f))
    count_data = 0
    with open(fn2, 'w') as outfile:
        for f in data:
            tmp = f['id'].split("_")
            if(tmp[0] == fn3):
                count_data = count_data + 1
                json.dump(f, outfile, ensure_ascii=False)
                outfile.write('\n')
    # print("length data = {}".format(count_data))


def split_train_test(data, fn2, fn3):
    """
            tach ra train vs test theo ti le 8 :2
            ghi train vao f2
            test f3
    """
    train, test = train_test_split(data, test_size=0.2, random_state=3255)
    with open(fn2, 'w') as outfile:
        for f in train:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')

    with open(fn3, 'w') as outfile:
        for f in test:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')
    return train, test


def split_token_json(f, data1, data2):
    """
            tach tu
            data1= la du lieu chuan
            data2 la violation
    """
    with open(f, 'r') as json_data:
        for f in json_data:
            tmp = json.loads(f)
            if flag == 1:
                x1 = copy.copy(tmp['original'])
                y1 = ViTokenizer.tokenize(x1)
                y1 = filter_punctuation(y1)  # loc dau cau
                y1 = y1.split(" ")
                tmp['original'] = copy.copy(y1)
                tmp['raw'] = copy.copy(y1)
                tmp.update({'tid': 0})
                data1.append(tmp)
            elif flag == 2:
                x1 = copy.copy(tmp['original'])
                y1 = ViTokenizer.tokenize(x1)
                y1 = filter_punctuation(y1)  # loc dau cau
                y1 = y1.split(" ")
                tmp['original'] = copy.copy(y1)
                tmp['raw'] = copy.copy(y1)
                tmp.update({'tid': 0})
                data1.append(tmp)
            else:
                x = copy.copy(tmp['raw'])
                y = ViTokenizer.tokenize(x)
                y = filter_punctuation(y)  # loc dau cau
                y = y.split(" ")
                tmp['raw'] = copy.copy(y)
                x1 = copy.copy(tmp['original'])
                y1 = ViTokenizer.tokenize(x1)
                y1 = filter_punctuation(y1)  # loc dau cau
                y1 = y1.split(" ")
                tmp['original'] = copy.copy(y1)
                tmp.update({'tid': 0})
                if len(y) == len(y1):
                    data1.append(tmp)
                else:
                    data2.append(tmp)


def data_add_noise(f1, f7, f8):
    """
            data1 - standart
            data2- violation
    """
    data1 = []
    data2 = []
    split_token_json(f1, data1, data2)
    # print('khong bi vioaltion {}'.format(len(data1)))
    fillter_number_d_underscore(data1)  # chi filter data chuan
    data1 = add_noise_sequen(data1)  # add_noise
    data1 = shuffle(data1)  # shuffle data
    with open(f7, 'w') as outfile:
        json.dump(data1, outfile, ensure_ascii=False)
        # outfile.write(',\n')
    if len(data2) != 0:
        with open(f8, 'w') as outfile:
            json.dump(data2, outfile, ensure_ascii=False)


def result():
    for i in range(len(data_by_label_json)):
        convert(file, path+data_by_label_json[i], data_label[i])
        data_add_noise(
            path + data_by_label_json[i], path + data_by_label_json_noise[i], path + data_by_label_json_violation[i])
    merge_data_noise()


def test_length(f1, f2):
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
            if max_ < len(k):
                max_= len(k)
                # print(i['id'])
    print(max_)
    print('len test ', len(data1))
    # 47 + 4 =
    # 23 + 4 = 25  ========>


def random_10_sequence(number):
    """
            test
            comment number
            delete underscore
    """
    regex = re.compile(r'\S*\d+\S*', re.UNICODE)
    regex1 = re.compile(r'\S*_\S*', re.UNICODE)
    data1 = []

    # split_token_json(path+a[0],data1,data2)
    # split_token_json(path+a[1],data1,data2)
    # # split_token_json(path+a[2],data1,data2)

    with open(path+'train_data.json', 'r') as outfile:
        data1 = json.load(outfile)
    # element_random = random.sample(range(len(data1)), number)

    # for i in element_random:
    for i in range(number):
        print('original  ')
        print(data1[i]['original'])
        # print(data1[i]['id'])
        print('edit raw  ')
        print(data1[i]['raw'])
    # for i in range(len(data1)):
    #	if data1[i]['id'] == 'CALS_00039470':
    #		print(data1[i])


def fillter_number_d_underscore(data1):
    """
            loc string co chua so:
            vd: so dien thoai
    """
    # a = ['data_book.json', 'data_cals.json','data_dial.json', 'data_news.json', 'data_stors.json','data_dial2.json']
    # b = ['train_book.json', 'train_cals.json','train_dial.json', 'train_news.json', 'train_stors.json', 'train_dial2.json']
    # c = ['test_book.json', 'test_cals.json','test_dial.json', 'test_news.json', 'test_stors.json', 'test_dial2.json']
    # d = ["BOOK","CALS", "DIAL", "NEWS", "STORS", "DIAL2"]
    # f = ['train_data.json','test_data.json','tran_violation.json','test_violation.json']
    # path = './data/'

    # data1 = [] ## train
    # data2 = [] ## test
    regex1 = re.compile(r"\S*\d+\S*", re.UNICODE)
    regex = re.compile(r'\S*_\S*', re.UNICODE)

    for i in range(len(data1)):
        undersocre = {}
        delete_undersocre = random.random()
        for j, k in enumerate(data1[i]['original']):
            if re.search(regex1, k):
                data1[i]['raw'][j] = '@' + data1[i]['raw'][j]
                data1[i]['original'][j] = '@' + data1[i]['original'][j]
            elif re.search(regex, k):
                index = []
                for m, n in enumerate(k):   # tach tahnh ky tu
                    if n == '_':
                        index.append(m)
                if len(index) != 0:
                    undersocre[j] = index
        if bool(undersocre):
            # print(undersocre)
            # chon ngau nhien mot tu trong cau
            tmp = random.choice(list(undersocre.keys()))
            # print('word duoc chon = ',tmp)
            # print('delete_undersocre = ', delete_undersocre)
            if delete_undersocre > 0.65:
                if len(undersocre[tmp]) == 1:
                    # chon ngau nhien 1 dau cach trong cau
                    index_underscore = random.choice(undersocre[tmp])
                    # data1[i]['original'][tmp] = data1[i]['original'][tmp][:index_underscore] + data1[i]['original'][tmp][index_underscore+1:]
                    data1[i]['raw'][tmp] = data1[i]['raw'][tmp][:index_underscore] + \
                        data1[i]['raw'][tmp][index_underscore+1:]
                else:
                    n_index = random.randrange(1, len(undersocre[tmp]), 1)
                    for u in range(n_index):
                        # chon ngau nhien 1 dau cach trong cau
                        index_underscore = random.choice(undersocre[tmp])
                        # print(data1[i]['original'][tmp][index_underscore])
                        # data1[i]['original'][tmp] = data1[i]['original'][tmp][:index_underscore] + data1[i]['original'][tmp][index_underscore+1:]
                        data1[i]['raw'][tmp] = data1[i]['raw'][tmp][:index_underscore] + \
                            data1[i]['raw'][tmp][index_underscore+1:]
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
    sign_['a'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ẫ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'a', 'ẵ', 'ẳ', 'ẩ']
    sign_['ă'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'a', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['à'] = ['a', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['á'] = ['à', 'a', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['â'] = ['à', 'á', 'a', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ã'] = ['à', 'á', 'â', 'a', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ạ'] = ['à', 'á', 'â', 'ã', 'a',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ả'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'a', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ấ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'a', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ầ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'a', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ậ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'a', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ắ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'a', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ặ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'a', 'a', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ằ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'a', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ắ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'a', 'ă', 'ẫ', 'ẵ', 'ẳ', 'ẩ']
    sign_['ẵ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'a', 'ẳ', 'ẩ']
    sign_['ẳ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'a', 'ẩ']
    sign_['ẩ'] = ['à', 'á', 'â', 'ã', 'ạ',
                  'ả', 'ấ', 'ầ', 'ậ', 'ắ', 'ặ', 'ằ', 'ắ', 'ă', 'ẫ', 'ẵ', 'ẳ', 'a']

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
    sign_['i'] = ['ì', 'í', 'ỉ', 'ị', 'ĩ']
    sign_['ì'] = ['i', 'í', 'ỉ', 'ị', 'ĩ']
    sign_['í'] = ['ì', 'i', 'ỉ', 'ị', 'ĩ']
    sign_['ỉ'] = ['ì', 'í', 'i', 'ị', 'ĩ']
    sign_['ị'] = ['ì', 'í', 'ỉ', 'i', 'ĩ']
    sign_['ĩ'] = ['ì', 'í', 'ỉ', 'ị', 'i']
    sign_['o'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ỗ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'o']
    sign_['ò'] = ['o', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ó'] = ['ò', 'o', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ô'] = ['ò', 'ó', 'o', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['õ'] = ['ò', 'ó', 'ô', 'o', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ọ'] = ['ò', 'ó', 'ô', 'õ', 'o', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ỏ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'o',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ố'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'o', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ồ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'o', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ổ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'o', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ộ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'o', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ớ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'o', 'ờ', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ờ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'o', 'ỡ', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ỡ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'o', 'ợ', 'ở', 'ơ', 'ỗ']
    sign_['ợ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'o', 'ở', 'ơ', 'ỗ']
    sign_['ở'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ',
                  'ố', 'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'o', 'ơ', 'ỗ']
    sign_['ơ'] = ['ò', 'ó', 'ô', 'õ', 'ọ', 'ỏ', 'ố',
                  'ồ', 'ổ', 'ộ', 'ớ', 'ờ', 'ỡ', 'ợ', 'ở', 'o', 'ỗ']
    sign_['u'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ù'] = ['u', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ú'] = ['ù', 'u', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ụ'] = ['ù', 'ú', 'u', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ủ'] = ['ù', 'ú', 'ụ', 'u', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ứ'] = ['ù', 'ú', 'ụ', 'ủ', 'u', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ừ'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'u', 'ữ', 'ự', 'ư', 'ử', 'ũ']
    sign_['ữ'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'u', 'ự', 'ư', 'ử', 'ũ']
    sign_['ự'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'u', 'ư', 'ử', 'ũ']
    sign_['ư'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'u', 'ử', 'ũ']
    sign_['ử'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'u', 'ũ']
    sign_['ũ'] = ['ù', 'ú', 'ụ', 'ủ', 'ứ', 'ừ', 'ữ', 'ự', 'ư', 'ử', 'u']
    sign_['y'] = ['ý', 'ỳ', 'ỵ', 'ỷ', 'ỹ']
    sign_['ý'] = ['y', 'ỳ', 'ỵ', 'ỷ', 'ỹ']
    sign_['ỳ'] = ['ý', 'y', 'ỵ', 'ỷ', 'ỹ']
    sign_['ỵ'] = ['ý', 'ỳ', 'y', 'ỷ', 'ỹ']
    sign_['ỷ'] = ['ý', 'ỳ', 'ỵ', 'y', 'ỹ']
    sign_['ỹ'] = ['ý', 'ỳ', 'ỵ', 'ỷ', 'y']
    return sign_[word]


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
    try:
        return array_prox[word]
    except:
        return word


def noise_telex(word):
        # cach 1: viet day du dau tai cho
    noise_tele = {'à': 'af', 'á': 'as', 'ã': 'ax', 'ạ': 'aj', 'ả': 'ar',
                  'â': 'aa', 'ấ': ['aas', 'asa'], 'ầ': ['aaf', 'afa'], 'ẫ': ['aax', 'axa'], 'ẩ': ['aar', 'ara'], 'ậ': ['aaj', 'aja'],
                  'ă': 'aw', 'ắ': ['aws', 'asw'], 'ằ': ['awf', 'afw'], 'ẵ': ['awx', 'axw'], 'ẳ': ['awr', 'arw'], 'ặ': ['awj', 'ajw'],
                  'ê': 'ee', 'ề': ['eef', 'efe'], 'ế': ['ees', 'ese'], 'ễ': ['eex', 'exe'], 'ể': ['eer', 'ere'], 'ệ': ['eej', 'eje'],
                  'é': 'es', 'è': 'ef', 'ẽ': 'ex', 'ẻ': 'er', 'ẹ': 'ej',
                  'ó': 'os', 'ò': 'of', 'õ': 'ox', 'ỏ': 'or', 'ọ': 'oj',
                  'ô': 'oo', 'ồ': ['oof', 'ofo'], 'ộ': ['ooj', 'ojo'], 'ổ': ['oor', 'oro'], 'ỗ': ['oox', 'oxo'], 'ố': ['oos', 'oso'],
                  'ơ': 'ow', 'ớ': ['ows', 'osw'], 'ờ': ['owf', 'ofw'], 'ở': ['owr', 'orw'], 'ợ': ['owj', 'ojw'], 'ỡ': ['owx', 'oxw'],
                  'ụ': 'uj', 'ú': 'us', 'ũ': 'ux', 'ủ': 'ur', 'ù': 'uf',
                  'ư': 'uw', 'ứ': ['uws', 'usw'], 'ừ': ['uwf', 'ufw'], 'ữ': ['uwx', 'uxw'], 'ử': ['uwr', 'urw'], 'ự': ['uwj', 'ujw'],
                  'ý': 'ys', 'ỳ': 'yf', 'ỷ': 'yr', 'ỹ': 'yx', 'ỵ': 'yj',
                  'í': 'is', 'ì': 'if', 'ĩ': 'ix', 'ỉ': 'ir', 'ị': 'ij',
                  'đ': 'dd'}
    # 67
    # viet moi dau o cuoi
    noise_tele2 = {'à': ['a', 'f'], 'á': ['a', 's'], 'ã': ['a', 'x'], 'ạ': ['a', 'j'], 'ả': ['a', 'r'],
                   'â': 'aa', 'ấ': ['aa', 's'], 'ầ': ['aa', 'f'], 'ẫ': ['aa', 'x'], 'ẩ': ['aa', 'r'], 'ậ': ['aa', 'j'],
                   'ă': 'aw', 'ắ': ['aw', 's'], 'ằ': ['aw', 'f'], 'ẵ': ['aw', 'x'], 'ẳ': ['aw', 'r'], 'ặ': ['aw', 'j'],
                   'ê': 'ee', 'ề': ['ee', 'f'], 'ế': ['ee', 's'], 'ễ': ['ee', 'x'], 'ể': ['ee', 'r'], 'ệ': ['ee', 'j'],
                   'é': ['e', 's'], 'è': ['e', 'f'], 'ẽ': ['e', 'x'], 'ẻ': ['e', 'r'], 'ẹ': ['e', 'j'],
                   'ó': ['o', 's'], 'ò': ['o', 'f'], 'õ': ['o', 'x'], 'ỏ': ['o', 'r'], 'ọ': ['o', 'j'],
                   'ô': 'oo', 'ồ': ['oo', 'f'], 'ộ': ['oo', 'j'], 'ổ': ['oo', 'r'], 'ỗ': ['oo', 'x'], 'ố': ['oo', 's'],
                   'ơ': 'ow', 'ớ': ['ow', 's'], 'ờ': ['ow', 'f'], 'ở': ['ow', 'r'], 'ợ': ['ow', 'j'], 'ỡ': ['ow', 'x'],
                   'ụ': ['u', 'j'], 'ú': ['u', 's'], 'ũ': ['u', 'x'], 'ủ': ['u', 'r'], 'ù': ['u', 'f'],
                   'ư': 'uw', 'ứ': ['uw', 's'], 'ừ': ['uw', 'f'], 'ữ': ['uw', 'x'], 'ử': ['uw', 'r'], 'ự': ['uw', 'j'],
                   'ý': ['y', 's'], 'ỳ': ['y', 'f'], 'ỷ': ['y', 'r'], 'ỹ': ['y', 'x'], 'ỵ': ['y', 'j'],
                   'í': ['i', 's'], 'ì': ['i', 'f'], 'ĩ': ['i', 'x'], 'ỉ': ['i', 'r'], 'ị': ['i', 'j'],
                   'đ': 'dd'}

    # viet day du tai cuoi

    noise_tele3 = {'à': ['a', 'f'], 'á': ['a', 's'], 'ã': ['a', 'x'], 'ạ': ['a', 'j'], 'ả': ['a', 'r'],
                   'â': ['a', 'a'], 'ấ': ['a', 'as', 'sa'], 'ầ': ['a', 'af', 'fa'], 'ẫ': ['a', 'ax', 'xa'], 'ẩ': ['a', 'ar', 'ra'], 'ậ': ['a', 'aj', 'ja'],
                   'ă': ['a', 'w'], 'ắ': ['a', 'ws', 'sw'], 'ằ': ['a', 'wf', 'fw'], 'ẵ': ['a', 'wx', 'xw'], 'ẳ': ['a', 'wr', 'rw'], 'ặ': ['a', 'wj', 'jw'],
                   'ê': ['e', 'e'], 'ề': ['e', 'ef', 'fe'], 'ế': ['e', 'es', 'se'], 'ễ': ['e', 'ex', 'xe'], 'ể': ['e', 'er', 're'], 'ệ': ['e', 'ej', 'je'],
                   'é': ['e', 's'], 'è': ['e', 'f'], 'ẽ': ['e', 'x'], 'ẻ': ['e', 'r'], 'ẹ': ['e', 'j'],
                   'ó': ['o', 's'], 'ò': ['o', 'f'], 'õ': ['o', 'x'], 'ỏ': ['o', 'r'], 'ọ': ['o', 'j'],
                   'ô': ['o', 'o'], 'ồ': ['o', 'of', 'fo'], 'ộ': ['o', 'oj', 'jo'], 'ổ': ['o', 'or', 'ro'], 'ỗ': ['o', 'ox', 'xo'], 'ố': ['o', 'os', 'so'],
                   'ơ': ['o', 'w'], 'ớ': ['o', 'ws', 'sw'], 'ờ': ['o', 'wf', 'fw'], 'ở': ['o', 'wr', 'rw'], 'ợ': ['o', 'wj', 'jw'], 'ỡ': ['o', 'wx', 'xw'],
                   'ụ': ['u', 'j'], 'ú': ['u', 's'], 'ũ': ['u', 'x'], 'ủ': ['u', 'r'], 'ù': ['u', 'f'],
                   'ư': ['u', 'w'], 'ứ': ['u', 'ws', 'sw'], 'ừ': ['u', 'wf', 'fw'], 'ữ': ['u', 'wx', 'xw'], 'ử': ['u', 'wr', 'rw'], 'ự': ['u', 'wj', 'jw'],
                   'ý': ['y', 's'], 'ỳ': ['y', 'f'], 'ỷ': ['y', 'r'], 'ỹ': ['y', 'x'], 'ỵ': ['y', 'j'],
                   'í': ['i', 's'], 'ì': ['i', 'f'], 'ĩ': ['i', 'x'], 'ỉ': ['i', 'r'], 'ị': ['i', 'j'],
                   'đ': ['d', 'd']}
    noise_list = []
    tmp1 = copy.copy(noise_tele[word])
    tmp2 = copy.copy(noise_tele2[word])
    tmp3 = copy.copy(noise_tele3[word])
    noise_list.append(tmp1)
    noise_list.append(tmp2)
    noise_list.append(tmp3)
    # print(noise_list)
    # return random.choice(noise_list)
    return noise_list


def noise_vni(word):
    # cach 1: viet day du dau tai cho
    noise_vni1 = {'à': 'a2', 'á': 'a1', 'ã': 'a4', 'ạ': 'a5', 'ả': 'a3',
                  'â': 'a6', 'ấ': ['a61', 'a16'], 'ầ': ['a62', 'a26'], 'ẫ': ['a64', 'a46'], 'ẩ': ['a63', 'a36'], 'ậ': ['a65', 'a56'],
                  'ă': 'a8', 'ắ': ['a81', 'a18'], 'ằ': ['a82', 'a28'], 'ẵ': ['a84', 'a48'], 'ẳ': ['a83', 'a38'], 'ặ': ['a85', 'a58'],
                  'ê': 'e6', 'ề': ['e62', 'e62'], 'ế': ['e61', 'e16'], 'ễ': ['e64', 'e46'], 'ể': ['e63', 'e36'], 'ệ': ['e65', 'e56'],
                  'é': 'e1', 'è': 'e2', 'ẽ': 'e4', 'ẻ': 'e3', 'ẹ': 'e5',
                  'ó': 'o1', 'ò': 'o2', 'õ': 'o4', 'ỏ': 'o3', 'ọ': 'o5',
                  'ô': 'o6', 'ồ': ['o62', 'o26'], 'ộ': ['o65', 'o56'], 'ổ': ['o63', 'o36'], 'ỗ': ['o64', 'o46'], 'ố': ['o61', 'o16'],
                  'ơ': 'o7', 'ớ': ['o71', 'o17'], 'ờ': ['o72', 'o27'], 'ở': ['o73', 'o37'], 'ợ': ['o75', 'o57'], 'ỡ': ['o74', 'o47'],
                  'ụ': 'u5', 'ú': 'u1', 'ũ': 'u4', 'ủ': 'u3', 'ù': 'u2',
                  'ư': 'u7', 'ứ': ['u71', 'u17'], 'ừ': ['u72', 'u27'], 'ữ': ['u74', 'u47'], 'ử': ['u73', 'u37'], 'ự': ['u75', 'u57'],
                  'ý': 'y1', 'ỳ': 'y2', 'ỷ': 'y3', 'ỹ': 'y4', 'ỵ': 'y5',
                  'í': 'i1', 'ì': 'i2', 'ĩ': 'i4', 'ỉ': 'i3', 'ị': 'i5',
                  'đ': 'd9'}
    # 67
    # viet moi dau o cuoi
    noise_vni2 = {'à': ['a', '2'], 'á': ['a', '1'], 'ã': ['a', '4'], 'ạ': ['a', '5'], 'ả': ['a', '3'],
                  'â': 'a6', 'ấ': ['a6', '1'], 'ầ': ['a6', '2'], 'ẫ': ['a6', '4'], 'ẩ': ['a6', '3'], 'ậ': ['a6', '5'],
                  'ă': 'a8', 'ắ': ['a8', '1'], 'ằ': ['a8', '2'], 'ẵ': ['a8', '4'], 'ẳ': ['a8', '3'], 'ặ': ['a8', '5'],
                  'ê': 'e6', 'ề': ['e6', '2'], 'ế': ['e6', '1'], 'ễ': ['e6', '4'], 'ể': ['e6', '3'], 'ệ': ['e6', '5'],
                  'é': ['e', '1'], 'è': ['e', '2'], 'ẽ': ['e', '4'], 'ẻ': ['e', '3'], 'ẹ': ['e', '5'],
                  'ó': ['o', '1'], 'ò': ['o', '2'], 'õ': ['o', '4'], 'ỏ': ['o', '3'], 'ọ': ['o', '5'],
                  'ô': 'o6', 'ồ': ['o6', '2'], 'ộ': ['o6', '5'], 'ổ': ['o6', '3'], 'ỗ': ['o6', '4'], 'ố': ['o6', '1'],
                  'ơ': 'o7', 'ớ': ['o7', '1'], 'ờ': ['o7', '2'], 'ở': ['o7', '3'], 'ợ': ['o7', '5'], 'ỡ': ['o7', '4'],
                  'ụ': ['u', '5'], 'ú': ['u', '1'], 'ũ': ['u', '4'], 'ủ': ['u', '3'], 'ù': ['u', '2'],
                  'ư': 'u7', 'ứ': ['u7', '1'], 'ừ': ['u7', '2'], 'ữ': ['u7', '4'], 'ử': ['u7', '3'], 'ự': ['u7', '5'],
                  'ý': ['y', '1'], 'ỳ': ['y', '2'], 'ỷ': ['y', '3'], 'ỹ': ['y', '4'], 'ỵ': ['y', '5'],
                  'í': ['i', '1'], 'ì': ['i', '2'], 'ĩ': ['i', '4'], 'ỉ': ['i', '3'], 'ị': ['i', '5'],
                  'đ': 'd9'}
    # viet day du tai cuoi

    noise_vni3 = {'à': ['a', '2'], 'á': ['a', '1'], 'ã': ['a', '4'], 'ạ': ['a', '5'], 'ả': ['a', '3'],
                  'â': ['a', '6'], 'ấ': ['a', '61', '16'], 'ầ': ['a', '62', '26'], 'ẫ': ['a', '64', '46'], 'ẩ': ['a', '63', '36'], 'ậ': ['a', '65', '56'],
                  'ă': ['a', '8'], 'ắ': ['a', '81', '18'], 'ằ': ['a', '82', '28'], 'ẵ': ['a', '84', '48'], 'ẳ': ['a', '83', '38'], 'ặ': ['a', '85', '58'],
                  'ê': ['e', '6'], 'ề': ['e', '62', '26'], 'ế': ['e', '61', '16'], 'ễ': ['e', '64', '46'], 'ể': ['e', '63', '36'], 'ệ': ['e', '65', '56'],
                  'é': ['e', '1'], 'è': ['e', '2'], 'ẽ': ['e', '4'], 'ẻ': ['e', '3'], 'ẹ': ['e', '5'],
                  'ó': ['o', '1'], 'ò': ['o', '2'], 'õ': ['o', '4'], 'ỏ': ['o', '3'], 'ọ': ['o', '5'],
                  'ô': ['o', '6'], 'ồ': ['o', '62', '26'], 'ộ': ['o', '65', '56'], 'ổ': ['o', '63', '36'], 'ỗ': ['o', '64', '46'], 'ố': ['o', '61', '16'],
                  'ơ': ['o', '7'], 'ớ': ['o', '71', '17'], 'ờ': ['o', '72', '27'], 'ở': ['o', '73', '37'], 'ợ': ['o', '75', '57'], 'ỡ': ['o', '74', '47'],
                  'ụ': ['u', '5'], 'ú': ['u', '1'], 'ũ': ['u', '4'], 'ủ': ['u', '3'], 'ù': ['u', '2'],
                  'ư': ['u', '7'], 'ứ': ['u', '71', '17'], 'ừ': ['u', '72', '27'], 'ữ': ['u', '74', '47'], 'ử': ['u', '73', '37'], 'ự': ['u', '75', '57'],
                  'ý': ['y', '1'], 'ỳ': ['y', '2'], 'ỷ': ['y', '3'], 'ỹ': ['y', '4'], 'ỵ': ['y', '5'],
                  'í': ['i', '1'], 'ì': ['i', '2'], 'ĩ': ['i', '4'], 'ỉ': ['i', '3'], 'ị': ['i', '5'],
                  'đ': ['d', '9']}
    noise_list = []
    tmp1 = copy.copy(noise_vni1[word])
    tmp2 = copy.copy(noise_vni2[word])
    tmp3 = copy.copy(noise_vni3[word])
    noise_list.append(tmp1)
    noise_list.append(tmp2)
    noise_list.append(tmp3)
    # return random.choice(noise_list)
    return noise_list


def closely_pronunciation1(subword):
    init_consonants = {'l': ['n'], 'n': ['l'], 'ch': ['tr'], 'tr': ['ch'], 'x': ['s'], 's': ['x'], 'r': ['d', 'gi', 'v'],
                       'd': ['r', 'gi', 'v'], 'gi': ['r', 'd', 'v'], 'v': ['r', 'gi', 'd']}
    return init_consonants[subword]


# def closely_pronunciation2(subword):
#     finish_consonants = {'n': ['ng', 'nh'], 'ng': ['n', 'nh'], 'nh': ['n', 'ng'],
#                          'c': ['t', 'ch'], 't': ['c', 'ch'], 'ch': ['c', 't']}
#     return finish_consonants[subword]


# def like_pronunciation1(subword):
#     like_pronunciation_fini = {'y': 'i', 'i': 'y'}
#     # {
#     #     'ênh':'ên', 'inh':' in', 'ăng':'ăn'. 'êch':'ết', 'ich':'it',\

#     # }
#     return like_pronunciation_fini[subword]

def saigon_final3(word):
    exchange_3_character = {}
    exchange_3_character['inh'] = 'in'
    exchange_3_character['ênh'] = 'ên'
    exchange_3_character['iên'] = 'iêng'
    exchange_3_character['ươn'] = 'ương'
    exchange_3_character['uôn'] = 'uông'
    exchange_3_character['iêt'] = 'iêc'
    exchange_3_character['ươt'] = 'ươc'
    exchange_3_character['uôt'] = 'uôc'
    return exchange_3_character[word]


def saigon_final2(word):
    exchange_2_character = {}
    exchange_2_character['ăn'] = 'ăng'
    exchange_2_character['an'] = 'ang'
    exchange_2_character['ân'] = 'âng'
    exchange_2_character['ưn'] = 'ưng'
    exchange_2_character['ắt'] = 'ăc'
    exchange_2_character['ât'] = 'âc'
    exchange_2_character['ưt'] = 'ưc'
    exchange_2_character['ôn'] = 'ông'
    exchange_2_character['un'] = 'ung'
    exchange_2_character['ôt'] = 'ôc'
    exchange_2_character['ut'] = 'uc'
    return exchange_2_character[word]


def like_pronunciation2(subword):
    like_pronunciation_init = {'g': ['gh'], 'gh': ['g'],
                               'c': ['q', 'k'], 'q': ['c', 'k'], 'k': ['c', 'q'],
                               'ng': ['ngh'], 'ngh': ['ng']}
    return like_pronunciation_init[subword]


def consonant_digraphs(word):
    consonant_digraph = {'ch': 'hc', 'gh': 'hg', 'gi': 'ig', 'kh': 'hk',
                         'nh': 'hn', 'ng': 'gn', 'ph': 'hp', 'th': 'ht', 'tr': 'rt', 'qu': 'uq'}
    return consonant_digraph[word]


def consonant_trigraphs(word):
    consonant_trigraph = {'ngh': ['gnh', 'nhg']}
    return consonant_trigraph[word]


def add_noise(word, op):
    """
    0. xoa 1 ki tu trong tu, khong tinh dau _
    1. doi cho 2 ki tu lien tiep trong 1 tu
    2. them _ trong 1 tu
    3. bot _ trong 1 tu , neu co nhieu thi chon ngau nhien
    4. gan nhau tren ban phim
    5. noise_telex
    6. noise_vni
    7. lap lai mot so nguyen am khong dung o dau am tiet

    8. cac am vi o dau cua am tiet gan giong nhau
    9. saigon phonology  am cuoi
    10 nga va hoi

    11 cac am vi o dau co cach phat am giong nhau trong mot so truong hop, vi du c q k
    12 sai vi tri dau va thay doi chu cai voi to hop cac dau neu am tiet co 1 nguyen am
    """

    i = random.randint(0, len(word) - 1)
    # op = random.randint(0,12)
    # print(op)
    # i  = 1
    if op == 0 or op == 13:
        if word[i] != '_' and len(word) >= 2:
            return word[:i] + word[i+1:]
    if op == 1 or op == 14:
        i += 1
        consonants = ['ch', 'tr', 'kh', 'nh', 'ng',
                      'qu', 'th', 'ph', 'gh', 'gi', 'ngh']  # 11
        tmp = [i for i in consonants if i in word]
        r = random.random()
        if len(tmp) > 0 and r > 0.1:
            tmp1 = random.choice(tmp)
            if tmp1 == 'ngh':
                return word.replace(tmp1, random.choice(consonant_trigraphs(tmp1)))
            else:
                return word.replace(tmp1, consonant_digraphs(tmp1))
        elif i <= len(word) - 1:
            return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]
        else:
            return word
    if op == 2 or op == 15:
        i += 1
        if i <= len(word) - 1:
            return word[:i] + '_' + word[i:]
        else:
            return word
    if op == 3 or op == 16:
        idx = word.find("_")  # check xem co underscore khong
        if idx != -1:
            list_underscore = [m.start() for m in re.finditer('_', word)]
            idx1 = random.choice(list_underscore)
            return word[:idx1] + word[idx1 + 1:]
        else:
            return word
    if op == 4:
        string_list = string.ascii_lowercase
        if word[i] in string_list:
            return word[:i] + random.choice(get_prox_keys(word[i])) + word[i+1:]
        else:
            return word
    # tam thoi coi 2 cai duoi la 1
    if op == 5:
        string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
        syllable = word.split('_')
        tmp = []
        for syllable_i in syllable:
            letter = [str(i) for i in string_list if i in syllable_i]
            for letter_i in letter:
                tmp1 = noise_telex(letter_i)
                # print(tmp1)
                r = random.random()
                if r < 0.9:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'str':
                            # print('1')
                            syllable_i = syllable_i.replace(letter_i, tmp1[0])

                        else:
                            # print('2')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][0])
                            # return syllable_i

                    else:
                        if type(tmp1[1]).__name__ == 'str':
                            # print('3')
                            syllable_i = syllable_i.replace(letter_i, tmp1[1])

                        else:
                            # print('4')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[1][0]) + tmp1[1][1]

                            # return syllable_i
                else:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'list':
                            # print('5')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][1])

                    else:
                        # print('6')
                        syllable_i = syllable_i.replace(
                            letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
            tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word

    if op == 6:
        string_list = 'àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵ'
        syllable = word.split('_')
        tmp = []
        for syllable_i in syllable:
            letter = [str(i) for i in string_list if i in syllable_i]
            for letter_i in letter:
                tmp1 = noise_vni(letter_i)
                # print(tmp1)
                r = random.random()
                if r < 0.9:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'str':
                            # print('1')
                            syllable_i = syllable_i.replace(letter_i, tmp1[0])

                        else:
                            # print('2')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][0])
                            # return syllable_i

                    else:
                        if type(tmp1[1]).__name__ == 'str':
                            # print('3')
                            syllable_i = syllable_i.replace(letter_i, tmp1[1])

                        else:
                            # print('4')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[1][0]) + tmp1[1][1]

                            # return syllable_i
                else:
                    rr = random.random()
                    if rr > 0.5:
                        if type(tmp1[0]).__name__ == 'list':
                            # print('5')
                            syllable_i = syllable_i.replace(
                                letter_i, tmp1[0][1])

                    else:
                        # print('6')
                        syllable_i = syllable_i.replace(
                            letter_i, tmp1[2][0]) + random.choice(tmp1[2][1:])
            tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word
    if op == 7:
        l = word[i]
        vowel = 'ouieay'
        if i >= 1 and l in vowel:
            return word[:i] + random.randint(1, 5) * l + word[i+1:]
        else:
            return word

    # thay doi o dau
    if op == 8 or op == 17 or op == 18:
        string_list1 = ['l', 'n', 'x', 's', 'r', 'd', 'v']
        string_list2 = ["ch", 'tr', 'gi']
        syllable = word.split("_")
        for i, syllable_i in enumerate(syllable):
            r = random.random()
            if r > 0.5:
                if len(syllable_i) >= 1 and syllable_i[0] in string_list1:
                    wo = random.choice(closely_pronunciation1(syllable_i[0]))
                    syllable[i] = wo + syllable_i[1:]
                elif len(syllable_i) >= 2 and syllable_i[0]+syllable_i[1] in string_list2:
                    wo = random.choice(closely_pronunciation1(
                        syllable_i[0] + syllable_i[1]))
                    syllable[i] = wo + syllable_i[2:]
        return "_".join(syllable)
    # thay doi o cuoi
    # co mot vai truong hop rieng thoi
    # saigon phonology
    if op == 9 or op == 19 or op == 20:
        string_list1 = ['inh', 'ênh', 'iên', 'ươn', 'uôn', 'iêt', 'ươt', 'uôt']
        string_list2 = ['ăn', 'an', 'ân', 'ưn', 'ắt', 'ât', 'ưt', 'ôn', 'un',
                        'ât', 'ưt', 'ôn', 'un', 'ôt', 'ut']
        syllable = word.split("_")
        tmp = []
        for syllable_i in syllable:
            if len(syllable_i) >= 3 and syllable_i[len(syllable_i) - 3:] in string_list1:
                syllable_i = syllable_i[:len(
                    syllable_i) - 3] + saigon_final3(str(syllable_i[len(syllable_i) - 3:]))
                tmp.append(syllable_i)
            elif len(syllable_i) >= 2 and syllable_i[len(syllable_i) - 2:] in string_list2:
                syllable_i = syllable_i[:len(
                    syllable_i) - 2] + saigon_final2(str(syllable_i[len(syllable_i) - 2:]))
                tmp.append(syllable_i)
        if len(tmp) > 0:
            return "_".join(tmp)
        else:
            return word

    if op == 10 or op == 21:
        string_list = ['ã', 'ả',
                       'ẫ', 'ẩ',
                       'ẵ', 'ẳ',
                       'ẻ', 'ẽ',
                       'ể', 'ễ',
                       'ĩ', 'ỉ',
                       'ũ', 'ủ',
                       'ữ', 'ử',
                       'õ', 'ỏ',
                       'ỗ', 'ổ', 'ỡ', 'ở']
        swap = {'ã': 'ả', 'ả': 'ã', 'ẫ': 'ẩ', 'ẩ': 'ẫ',
                'ẵ': 'ẳ', 'ẳ': 'ẵ', 'ẻ': 'ẽ', 'ẽ': 'ẻ', 'ễ': 'ể', 'ể': 'ễ',
                'ĩ': 'ỉ', 'ỉ': 'ĩ', 'ũ': 'ủ', 'ủ': 'ũ', 'ữ': 'ử', 'ử': 'ữ',
                'õ': 'ỏ', 'ỏ': 'õ', 'ỗ': 'ổ', 'ổ': 'ỗ', 'ỡ': 'ở', 'ở': 'ỡ'}
        tmp = [i for i in string_list if i in word]
        for letters in tmp:
            word = word.replace(letters, swap[letters])
        return word

    if op == 11 or op == 22 or op == 23:
        string_list0 = ['ngh']
        string_list1 = ['gh', 'ng']
        string_list2 = ['g', 'c', 'q', 'k']
        syllable = word.split("_")
        for i, syllable_i in enumerate(syllable):
            r = random.random()
            if r > 0.5:
                if len(syllable_i) >= 3 and syllable_i[0] + syllable_i[1] + syllable_i[2] in string_list0:
                    wo = random.choice(like_pronunciation2(
                        syllable_i[0] + syllable_i[1] + syllable_i[2]))
                    syllable[i] = wo + syllable_i[3:]
                elif len(syllable_i) >= 2 and syllable_i[0] + syllable_i[1] in string_list1:
                    wo = random.choice(like_pronunciation2(
                        syllable_i[0] + syllable_i[1]))
                    syllable[i] = wo + syllable_i[2:]
                elif len(syllable_i) >= 1 in string_list2:
                    wo = random.choice(like_pronunciation2(syllable_i[0]))
                    syllable[i] = wo + syllable_i[1:]
        return "_".join(syllable)

    """
        thay doi vi tri dau
    """

    string_list1 = 'àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵaeiouy'
    string_list2 = ['óa', 'oá', 'òa','oà', 'ỏa', 'oả', 'õa', 'oã', 'ọa', 'oạ',\
            'áo', 'aó', 'ào','aò', 'ảo', 'aỏ', 'ão', 'aõ', 'ạo', 'aọ',\
            'éo', 'eó', 'èo','eò', 'ẻo', 'eỏ', 'ẽo', 'eõ', 'ẹo', 'eọ',\
            'óe', 'oé', 'òe','oè', 'ỏe', 'oẻ', 'õe', 'oẽ', 'ọe', 'oẹ',\
            'ái', 'aí', 'ài','aì', 'ải', 'aỉ', 'ãi', 'aĩ', 'ại', 'aị',\
            'ói', 'oí', 'òi','oì', 'ỏi', 'oỉ', 'õi', 'oĩ', 'ọi', 'oị'] # convert ve khong dau 

    dict_change = {'óa': 'oá', 'òa':'oà', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',\
                    'oá': 'óa', 'oà':'òa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',\
                    'áo': 'aó', 'ào':'aò', 'ảo': 'aỏ', 'ão': 'aõ', 'ạo': 'aọ',\
                    'aó': 'áo', 'aò':'ào', 'aỏ': 'ảo', 'aõ': 'ão', 'aọ': 'ạo',\
                    'éo': 'eó', 'èo':'eò', 'ẻo': 'eỏ', 'ẽo': 'eõ', 'ẹo': 'eọ',\
                    'eó': 'éo', 'eò':'èo', 'eỏ': 'ẻo', 'eõ': 'ẽo', 'eọ': 'ẹo',\
                    'óe': 'oé', 'òe':'oè', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',\
                    'oé': 'óe', 'oè':'òe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe', 'ái': 'aí', 'ài':'aì', 'ải': 'aỉ', 'ãi': 'aĩ', 'ại': 'aị', 'aí': 'ái', 'aì':'ài', 'aỉ': 'ải', 'aĩ': 'ãi', 'aị': 'ại', 'ói': 'oí', 'òi':'oì', 'ỏi': 'oỉ', 'õi': 'oĩ', 'ọi': 'oị',\
                    'oí': 'ói', 'oì':'òi', 'oỉ': 'ỏi', 'oĩ': 'õi', 'oị': 'ọi'}
    '''
        gom 3 truong hop
        truong hop 1: neu chi co 1 nguyen am, thi doi dauis unsubscriptable
        neu co am dem va am chinh (2 nguyen am), thi chuyen dau sang am dem
        (hoac co ca am dem, am chinh va am cuoi)
        chua xac dinh duoc y,i la am chinh hay am cuoi

    '''

    syllable = word.split("_")
    word_add = []
    for i, syllable_i in enumerate(syllable):
        tmp = [i for i in string_list1 if i in syllable_i]
        if len(tmp) == 1:
            syllable_i = syllable_i.replace(tmp[0], random.choice(get_change_sign(tmp[0])))
            word_add.append(syllable_i)
        else :
            tmp1 = [i for i in string_list2 if i in syllable_i]
            for letters in tmp1:
                syllable_i = syllable_i.replace(letters, dict_change[letters])
                word_add.append(syllable_i)
    if len(word_add) != 0:
        return "_".join(word_add)
    else:
        return word


def filter_punctuation(my_str):
    """
    chi nhung tu nay moi can loc
    regex2  = re.compile(
        '^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$',re.UNICODE)
j = 0
for i in b:
            if re.search(regex2,i):
            # print(add_noise(i,0))
            j+=1
            sau khi tokenzier thi loc

    """
    punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
        else:
            no_punct = no_punct + ' @' + char
    return no_punct

fff_file = open("text.txt", "w+")
def add_noise_sequen(data1):
    """
            chay ham nay khi da chay filter_number_d_underscore
    """
    
    print('len data ban dau {}'.format(len(data1)))

    data2 = []  # noise tao ra
    data3 = []
    regex1 = re.compile(r"\S*\d+\S*", re.UNICODE)
    regex2 = re.compile(
        r'^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$', re.UNICODE)
    error = np.arange(26)
    sequence = [1, 2, 3]
    index = 0
    # dem so loi
    count = 0  # dem so noise tao ra
    file_errorr = []
    for i in range(26):
        file_errorr.append(0)
    # print("length data befor noise = {} ".format(len(data1)))
    element_random = random.sample(range(len(data1)), int(0.7*len(data1)))
    for i in range(len(data1)):
        m = random.random()
        if i in element_random:  # xac suat chon cau de them nhieu
            n_quence = random.choice(sequence)
            for j in range(n_quence):
                n_error = random.randint(0, 30)*len(data1[i]['original'])/100
                tmp = copy.deepcopy(data1[i])
                data2.append(tmp)
                data2[index]['id'] = data2[index]['id'] + str(j)
                for j1 in range(int(n_error)):
                    n = random.randint(0, len(data2[index]['original']) - 1)
                    word = data2[index]['original'][n]
                    # so thi loai, chi cac tu trong trong bang chu cai thoi, cac dau cau khong tinh...
                    if not re.search(regex1, word) and re.search(regex2, word):
                        op = random.choice(error)

                        file_word1 = copy.copy(word)
                        word = add_noise(word, op)
                        file_word2 = copy.copy(word)
                        while file_word1 == file_word2:
                            op = random.choice(error)
                            word = add_noise(word, op)
                            file_word2 = copy.copy(word)
                        # while file_word2 == file_word1:
                        file_errorr[op] = file_errorr[op] + 1
                        #     print('1')
                        fff_file.write('%-15s  <%-2d>  %-15s\n' %
                                       (file_word1, op, file_word2))
                        data2[index]['raw'][n] = word
                count += 1
                data3.append(tmp)
                index += 1
        else:
            tmp1 = copy.deepcopy(data1[i])
            data3.append(tmp1)
    # print(len(data3))
    print(file_errorr)  # thong ke cac loi khi them
    print('length data noise tao ra {}'.format(count))
    # print("length data  noise = {} ".format(len(data2)))
    if flag == 1:
        for i in range(len(data2)):
            data1.append(data2[i])
    elif flag == 2:
        data1 = []
        for i in range(len(data3)):
            data1.append(data3[i])
    else:
        for i in range(len(data2)):
            data1.append(data2[i])
    print('sum data {}'.format(len(data1)))
    return data1


def create_tiny():
    convert(file, path_tiny + data_by_label_json[0], data_label[0])
    data_add_noise(path_tiny + data_by_label_json[0], path_tiny +
                   data_by_label_json_noise[0], path_tiny + data_by_label_json_violation[0])
    with open(path_tiny + data_by_label_json_noise[0], "r") as f:
        data = json.load(f)
    train, test = split_train_test(
        data, path_tiny + data_by_label_train_json[0], path_tiny + data_by_label_test_json[0])
    # cai nay tao view  roi nen khong can tao nua

    with open(path_tiny+final_json[6], 'w') as outfile:
        json.dump(train, outfile, ensure_ascii=False)

    with open(path_tiny+final_json[7], 'w') as outfile:
        json.dump(train, outfile, ensure_ascii=False)

    # ghi vao file train va test o dang json va bo tach tu di
    for i in range(len(train)):
        train[i]['raw'] = ' '.join(train[i]['raw'])
        train[i]['original'] = ' '.join(train[i]['original'])

    for i in range(len(test)):
        train[i]['raw'] = ' '.join(test[i]['raw'])
        train[i]['original'] = ' '.join(test[i]['original'])

    with open(path_tiny+final_json[8], 'w') as outfile:
        for f in train:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')

    with open(path_tiny+final_json[9], 'w') as outfile:
        for f in test:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')


def isUrl(word):
    """
            loc url
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        # domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    match = re.match(regex, token)
    if match is None:
        return False
    else:
        return True


def merge_data_noise():
    """
            tong tat ca data
    """
    total_train = []
    total_test = []
    for i, j in enumerate(data_by_label_json_noise):
        with open(path + j, "r") as f:
            data = json.load(f)
        train, test = split_train_test(
            data, path + data_by_label_train_json[i], path + data_by_label_test_json[i])
        total_train = total_train + train
        total_test = total_test + test
    print('tong tat ca du lieu {}'.format(len(total_train)+len(total_test)))
    # print(len(total_train))
    # print(len(total_test))

    # ghi vao file train va test o dang list de dua vao model
    with open(path+final_json[0], 'w') as outfile:
        json.dump(total_train, outfile, ensure_ascii=False)

    with open(path+final_json[1], 'w') as outfile:
        json.dump(total_train, outfile, ensure_ascii=False)

    for i in range(len(total_train)):
        total_train[i]['raw'] = ' '.join(total_train[i]['raw'])
        total_train[i]['original'] = ' '.join(total_train[i]['original'])

    for i in range(len(total_test)):
        total_test[i]['raw'] = ' '.join(total_test[i]['raw'])
        total_test[i]['original'] = ' '.join(total_test[i]['original'])
    # ghi vao file train va test o dang json va bo tach tu
    with open(path+final_json[2], 'w') as outfile:
        for f in total_train:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')

    with open(path+final_json[3], 'w') as outfile:
        for f in total_test:
            json.dump(f, outfile, ensure_ascii=False)
            outfile.write('\n')


path = './data1/'
path_tiny = './data_tiny/1/'
flag = 1  # sinh ra data loai 1

# path = './data2/'
# path_tiny = './data_tiny/2/'
# flag = 2 # sinh ra data loai 2

# path = './data3/'
# path_tiny = './data_tiny/3/'
# flag = 3


def test(word, i):
    idx = word.find("_")  # check xem co underscore khong
    if idx != -1:
        list_underscore = [m.start() for m in re.finditer('_', word)]
        idx1 = random.choice(list_underscore)
    c = 'ấ'
    a1 = noise_telex(c)
    a2 = noise_telex(c)
    a3 = noise_telex(c)
    a4 = noise_telex(c)


def convert_file_dic(fn1, fn2):
    """
        voca
    """
    data = []
    regex2 = re.compile(
        r'^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệghiìỉĩíịklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvxyỳỷỹýỵ_]+$', re.UNICODE)
    with open(fn1, 'r') as json_data:
        for f in json_data:
            data.append(json.loads(f))
    count_data = 0

    data_word = []
    for f in data:
        f1 = f['original']
        # count_data = count_data + 1
        ff = ViTokenizer.tokenize(f1)
        ff = filter_punctuation(ff)
        ff = ff.split()

        for _, element in enumerate(ff):
            if re.search(regex2, element):
                tmp = copy.copy(element)
                data_word.append(tmp)

    data_word = collections.Counter(data_word)
    with open(fn2, 'w') as outfile:
        for f0 in data_word:
            f0 = f0.split("_")
            if f0[0] != '':
                outfile.write(str(f0))
                outfile.write('\t')
                outfile.write(" ".join(f0))
                outfile.write('\n')


if __name__ == '__main__':
    # result()
	print(test_length('train_data.json', 'test_data.json'))
    # create_tiny()
    # convert_file_dic(file, 'word.txt')
    # word='à'
    # consonants=['ch', 'tr', 'kh', 'nh', 'ng',
    #     'qu', 'th', 'ph', 'gh', 'gi', 'ngh']  # 11
    # a=[i for i in consonants if i in word]
    # b=random.choice((a))
    # print(b)
    # print(word.find(b))
    # print(a)

    # print(add_noise('được', 5))
    # print(add_noise('linh_tinh', 9))
    # print(add_noise('linh_tinh', 9))
    # print(add_noise('linh_tinh', 9))
    # print(add_noise('linh_tinh', 9))
    # print(add_noise('linh_tinh', 9))
    # print(add_noise('mẩy', 5))
    # print(add_noise('mẩy', 5))
    # print(add_noise('mẩy', 5))
    # print(add_noise('mẩy', 5))
    # print(add_noise('mẩy', 5))
    # print(add_noise('mẩy', 5))

    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))
    # print(add_noise('lải_nhải', 12))

    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # add_noise('mẩy', 5)
    # a = 'mẩy'
    # for i in a:
    #     print(i)
