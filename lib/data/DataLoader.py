# -*- coding: utf-8 -*-
from .Tweet import Tweet, Preprocessor
from .Dict import Dict
import random
import lib
import json
import copy
from pyvi import ViTokenizer
import io
import re
import os

import string
import collections
import csv
import numpy as np
# random.seed(3255)

class DataLoader(object):
    def __init__(self, tweets, vocab, mappings, opt, file):
        self.opt = opt
        self.file = file
        self.get_change_sign = self.get_change_sign()
        # self.get_prox_keys = self.get_prox_keys()
        # self.noise_telex = self.noise_telex()
        # self.noise_vni = self.noise_vni()
        # self.closely_pronunciation1 = self.closely_pronunciation1()
        # self.saigon_final3 = self.saigon_final3()
        # self.saigon_final2 = self.saigon_final2()
        # self.saigon_final1 = self.saigon_final1()
        # self.like_pronunciation2 = self.like_pronunciation2()
        # self.consonant_digraphs = self.consonant_digraphs()
        # self.consonant_trigraphs = self.consonant_trigraphs()


        self.mappings = mappings if mappings else {}
        self.tweets, self.source_vocab, self.target_vocab = self.load_data(
            tweets)
        if(vocab):
            self.source_vocab = vocab["src"]
            self.target_vocab = vocab["tgt"]
        self.ret = self.encode_tweets()

    def tweets_toIdx(self):
        for tweet in self.tweets:
            input = copy.deepcopy(tweet.input)
            if(self.opt.correct_unique_mappings):
                for index, wi in enumerate(input):
                    mapping = self.mappings.get(wi)
                    if mapping and len(mapping) == 1 and list(mapping)[0] != wi:
                        input[index] = list(mapping)[0]
            tweet.set_inputidx(self.source_vocab.to_indices(
                input, bosWord=self.opt.bos, eosWord=self.opt.eos))
            tweet.set_outputidx(self.target_vocab.to_indices(
                tweet.output, bosWord=self.opt.bos, eosWord=self.opt.eos))

    def encode_tweets(self):
        self.tweets_toIdx()
        src_sents, tgt_sents, tgt_sents_words, src_sents_words, indices, tids = [], [], [], [], [], []
        for tweet in self.tweets:
            src_sents.append(tweet.inputidx)
            tgt_sents.append(tweet.outputidx)
            src_sents_words.append(tweet.input)
            tgt_sents_words.append(tweet.output)
            indices.append(tweet.ind)
            tids.append(tweet.tid)

        ret = {'src': src_sents,
               'src_sent_words': src_sents_words,
               'tgt': tgt_sents,
               'tgt_sent_words': tgt_sents_words,
               'pos': range(len(src_sents)),
               'index': indices,
               'tid': tids}

        return ret

    def vector_repr(self, inp_i, inp_o, update_mappings):
        if update_mappings:
            for k in range(len(inp_i)):
                try:
                    self.mappings[inp_i[k].lower()].add(inp_o[k].lower())
                except KeyError:
                    self.mappings[inp_i[k].lower()] = set()
                    self.mappings[inp_i[k].lower()].add(inp_o[k].lower())

        if(self.opt.self_tok == lib.constants.SELF):
            for i in range(len(inp_i)):
                if(inp_i[i].lower() == inp_o[i].lower()):
                    inp_o[i] = self.opt.self_tok

        if(self.opt.input == 'char'):
            inp_i = list('#'.join(inp_i))
            inp_o = list('#'.join(inp_o))

        return inp_i, inp_o

    def load_data(self, tweets):
        source_vocab = Dict(vocab_size=self.opt.vocab_size,
                            bosWord=self.opt.bos, eosWord=self.opt.eos)
        target_vocab = Dict(vocab_size=self.opt.vocab_size,
                            bosWord=self.opt.bos, eosWord=self.opt.eos)
        if(self.opt.share_vocab):
            target_vocab = source_vocab
        processor = Preprocessor()
        # for test the mappings are predefined and for all other inputs except word level we dont need them, so no updates
        update_mappings = not self.mappings and self.opt.input == 'word'
        word_tweets = []
        regex = re.compile(
            r"^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$", re.UNICODE)
        for tweet in tweets:
            inp_i, pos_i = processor.run(tweet.input, self.opt.lowercase)
            inp_o, pos_o = processor.run(tweet.output, self.opt.lowercase)
            if(self.opt.input == 'spelling'):  # character model word2word corrections
                for iword, oword in zip(inp_i, inp_o):
                    # iword = iword.encode('ascii','ignore')
                    # oword = oword.encode('ascii','ignore')
                    # iword = iword.encode('utf-8')
                    # oword = oword.encode('utf-8')
                                    # print('len cua word',len(iword))
                                    # print('word ',iword)
                    match1 = re.search(regex, iword)
                    match2 = re.search(regex, oword)
                    if iword and oword and match1 and match2:
                        if iword == oword and len(iword) > 1 and len(oword) > 1 and not any(c.isdigit() for c in iword) and not any(c.isdigit() for c in oword):
                            if random.random() > 0.9 and not self.opt.data_augm:
                                continue
                        iwordv, owordv = self.vector_repr(
                            iword, oword, update_mappings)
                        source_vocab.add_words(iwordv)
                        target_vocab.add_words(owordv)
                        tweet.set_input(iwordv)
                        tweet.set_output(owordv)
                        word_tweets.append(copy.deepcopy(tweet))
                        if(self.opt.data_augm):
                            if random.random() > (1 - self.opt.noise_ratio):
                                if iword == oword and len(iword) > 1 and len(oword) > 1 and not any(c.isdigit() for c in iword) and not any(c.isdigit() for c in oword):
                                    file_word1 = copy.copy(iword)
                                    iword = self.add_noise(iword)
                                    while file_word1 == iword:
                                        file_word1 = copy.copy(iword)
                                        iword = self.add_noise(iword)
                                        
                                    # aa_aa = iword
                                    # print('word sau khi add noise', unichr(int(aa_aa,16)))
                                    if(iword == '' or iword == ' '):
                                        continue
                                    iwordv, owordv = self.vector_repr(
                                        iword, oword, update_mappings)
                                    source_vocab.add_words(iwordv)
                                    target_vocab.add_words(owordv)
                                    tweet.set_input(iwordv)
                                    tweet.set_output(owordv)
                                    word_tweets.append(copy.deepcopy(tweet))
                                    # word_tweets.append(tweet)
            else:
                inp_i, inp_o = self.vector_repr(inp_i, inp_o, update_mappings)
                source_vocab.add_words(inp_i)
                target_vocab.add_words(inp_o)
                tweet.set_input(inp_i)
                tweet.set_output(inp_o)
                word_tweets.append(tweet)

        # store word_tweet
        if self.opt.input == 'word':
            with open('./data_word/'+self.file+"_word_tweet.txt", "w") as outfile:
                for line in tweets:
                    outfile.write(str(line))
                    outfile.write('\n')

        # store word_tweet
        if self.opt.input == 'spelling':
            with open('./data_character/'+self.file+"_word_tweet.txt", "w") as outfile:
                for line in word_tweets:
                    outfile.write(str(line))
                    outfile.write('\n')

        # print(self.opt.input)

        # print(word_tweets[0])
        # print(len(word_tweets))
        # print(source_vocab.vocab)
        tweets = word_tweets
        # print(tweets)
        if(self.opt.input == 'spelling'):
            same_tw, diff_tw = [], []
            for tweet in tweets:
                if tweet.input == tweet.output:
                    same_tw.append(tweet)
                else:
                    diff_tw.append(tweet)

        source_vocab.makeVocabulary(self.opt.vocab_size)
        source_vocab.makeLabelToIdx()
        target_vocab.makeVocabulary(self.opt.vocab_size)
        target_vocab.makeLabelToIdx()

        # for i in source_vocab.vocab:
        #     print(source_vocab.vocab[i])
        # print("---------")
        # print(source_vocab.vocab)

        # for i in range(len(source_vocab.idx_to_label)):
        #     # print(i)
        #     if(source_vocab.idx_to_label[i] != target_vocab.idx_to_label[i]):
        #         print("haha")
        # voi words model thi idx_to_label cua source vs target = nhau
        # print(source_vocab.idx_to_label[1])
        # # print(target_vocab.idx_to_label)
        # print(len(source_vocab.idx_to_label))

        if(self.opt.share_vocab):
            assert source_vocab.idx_to_label == target_vocab.idx_to_label
        return tweets, source_vocab, target_vocab

    def add_noise(self, word):
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
        op = random.randint(0, 26)
        i = random.randint(0, len(word) - 1)
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
                    # print(DataLoader.consonant_trigraphs(tmp1))
                    return word.replace(tmp1, random.choice(DataLoader.consonant_trigraphs(tmp1)))
                else:
                    return word.replace(tmp1, DataLoader.consonant_digraphs(tmp1))
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
                return word[:i] + random.choice(DataLoader.get_prox_keys(word[i])) + word[i+1:]
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
                    tmp1 = DataLoader.noise_telex(letter_i)
                    # print(tmp1)
                    r = random.random()
                    if r < 0.9:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'str':
                                # print('1')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[0])

                            else:
                                # print('2')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[0][0])
                                # return syllable_i

                        else:
                            if type(tmp1[1]).__name__ == 'str':
                                # print('3')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[1])

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
                    tmp1 = DataLoader.noise_vni(letter_i)
                    # print(tmp1)
                    r = random.random()
                    if r < 0.9:
                        rr = random.random()
                        if rr > 0.5:
                            if type(tmp1[0]).__name__ == 'str':
                                # print('1')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[0])

                            else:
                                # print('2')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[0][0])
                                # return syllable_i

                        else:
                            if type(tmp1[1]).__name__ == 'str':
                                # print('3')
                                syllable_i = syllable_i.replace(
                                    letter_i, tmp1[1])

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
                        wo = random.choice(
                            DataLoader.closely_pronunciation1(syllable_i[0]))
                        syllable[i] = wo + syllable_i[1:]
                    elif len(syllable_i) >= 2 and syllable_i[0]+syllable_i[1] in string_list2:
                        wo = random.choice(DataLoader.closely_pronunciation1(
                            syllable_i[0] + syllable_i[1]))
                        syllable[i] = wo + syllable_i[2:]
            return "_".join(syllable)
        # thay doi o cuoi
        # co mot vai truong hop rieng thoi
        # saigon phonology
        if op == 9 or op == 19 or op == 20:
            string_list1 = ['inh', 'ênh', 'iên',
                            'ươn', 'uôn', 'iêt', 'ươt', 'uôt']
            string_list2 = ['ăn', 'an', 'ân', 'ưn', 'ắt', 'ât', 'ưt', 'ôn', 'un',
                            'ât', 'ưt', 'ôn', 'un', 'ôt', 'ut']
            syllable = word.split("_")
            tmp = []
            for syllable_i in syllable:
                if len(syllable_i) >= 3 and syllable_i[len(syllable_i) - 3:] in string_list1:
                    syllable_i = syllable_i[:len(
                        syllable_i) - 3] + DataLoader.saigon_final3(str(syllable_i[len(syllable_i) - 3:]))
                    tmp.append(syllable_i)
                elif len(syllable_i) >= 2 and syllable_i[len(syllable_i) - 2:] in string_list2:
                    syllable_i = syllable_i[:len(
                        syllable_i) - 2] + DataLoader.saigon_final2(str(syllable_i[len(syllable_i) - 2:]))
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
                        wo = random.choice(DataLoader.like_pronunciation2(
                            syllable_i[0] + syllable_i[1] + syllable_i[2]))
                        syllable[i] = wo + syllable_i[3:]
                    elif len(syllable_i) >= 2 and syllable_i[0] + syllable_i[1] in string_list1:
                        wo = random.choice(DataLoader.like_pronunciation2(
                            syllable_i[0] + syllable_i[1]))
                        syllable[i] = wo + syllable_i[2:]
                    elif len(syllable_i) >= 1 in string_list2:
                        wo = random.choice(DataLoader.like_pronunciation2(syllable_i[0]))
                        syllable[i] = wo + syllable_i[1:]
            return "_".join(syllable)

        """
            thay doi vi tri dau
        """

        string_list1 = 'àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵaeiouy'
        string_list2 = ['óa', 'oá', 'òa', 'oà', 'ỏa', 'oả', 'õa', 'oã', 'ọa', 'oạ',
                        'áo', 'aó', 'ào', 'aò', 'ảo', 'aỏ', 'ão', 'aõ', 'ạo', 'aọ',
                        'éo', 'eó', 'èo', 'eò', 'ẻo', 'eỏ', 'ẽo', 'eõ', 'ẹo', 'eọ',
                        'óe', 'oé', 'òe', 'oè', 'ỏe', 'oẻ', 'õe', 'oẽ', 'ọe', 'oẹ',
                        'ái', 'aí', 'ài', 'aì', 'ải', 'aỉ', 'ãi', 'aĩ', 'ại', 'aị',
                        'ói', 'oí', 'òi', 'oì', 'ỏi', 'oỉ', 'õi', 'oĩ', 'ọi', 'oị']  # convert ve khong dau

        dict_change = {'óa': 'oá', 'òa': 'oà', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ',
                       'oá': 'óa', 'oà': 'òa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',
                       'áo': 'aó', 'ào': 'aò', 'ảo': 'aỏ', 'ão': 'aõ', 'ạo': 'aọ',
                       'aó': 'áo', 'aò': 'ào', 'aỏ': 'ảo', 'aõ': 'ão', 'aọ': 'ạo',
                       'éo': 'eó', 'èo': 'eò', 'ẻo': 'eỏ', 'ẽo': 'eõ', 'ẹo': 'eọ',
                       'eó': 'éo', 'eò': 'èo', 'eỏ': 'ẻo', 'eõ': 'ẽo', 'eọ': 'ẹo',
                       'óe': 'oé', 'òe': 'oè', 'ỏe': 'oẻ', 'õe': 'oẽ', 'ọe': 'oẹ',
                       'oé': 'óe', 'oè': 'òe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe', 'ái': 'aí', 'ài': 'aì', 'ải': 'aỉ', 'ãi': 'aĩ', 'ại': 'aị', 'aí': 'ái', 'aì': 'ài', 'aỉ': 'ải', 'aĩ': 'ãi', 'aị': 'ại', 'ói': 'oí', 'òi': 'oì', 'ỏi': 'oỉ', 'õi': 'oĩ', 'ọi': 'oị',
                       'oí': 'ói', 'oì': 'òi', 'oỉ': 'ỏi', 'oĩ': 'õi', 'oị': 'ọi'}
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
                syllable_i = syllable_i.replace(
                    tmp[0], random.choice(self.get_change_sign[tmp[0]]))
                word_add.append(syllable_i)
            else:
                tmp1 = [i for i in string_list2 if i in syllable_i]
                for letters in tmp1:
                    syllable_i = syllable_i.replace(
                        letters, dict_change[letters])
                    word_add.append(syllable_i)
        if len(word_add) != 0:
            return "_".join(word_add)
        else:
            return word


    def get_change_sign(self):
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
   
        return sign_
    @staticmethod
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
        # try:
        return array_prox[word]
        # except:
        #     return word
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def closely_pronunciation1(subword):
        init_consonants = {'l': ['n'], 'n': ['l'], 'ch': ['tr'], 'tr': ['ch'], 'x': ['s'], 's': ['x'], 'r': ['d', 'gi', 'v'],
                        'd': ['r', 'gi', 'v'], 'gi': ['r', 'd', 'v'], 'v': ['r', 'gi', 'd']}
        return init_consonants[subword]
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def like_pronunciation2(subword):
        like_pronunciation_init = {'g': ['gh'], 'gh': ['g'],
                                'c': ['q', 'k'], 'q': ['c', 'k'], 'k': ['c', 'q'],
                                'ng': ['ngh'], 'ngh': ['ng']}
        return like_pronunciation_init[subword]
    @staticmethod
    def consonant_digraphs(word):
        consonant_digraph = {'ch': 'hc', 'gh': 'hg', 'gi': 'ig', 'kh': 'hk',
                            'nh': 'hn', 'ng': 'gn', 'ph': 'hp', 'th': 'ht', 'tr': 'rt', 'qu': 'uq'}
        return consonant_digraph[word]
    @staticmethod
    def consonant_trigraphs(word):
            consonant_trigraph = {'ngh': ['gnh', 'nhg']}
            return consonant_trigraph[word]


def create_data(data, opt, vocab=None, mappings=None, file=''):
    dataload = DataLoader(
        data, vocab=vocab, mappings=mappings, opt=opt, file=file)
    vocab = {}
    vocab['src'] = dataload.source_vocab
    vocab['tgt'] = dataload.target_vocab
    return dataload.ret, vocab, dataload.mappings
    # return {}, {}, {}


def create_datasets(opt):
    train, val = read_file(opt.traindata, opt.valsplit)
    test, _ = read_file(opt.testdata)
    train_data, vocab, mappings = create_data(train, opt=opt, file="train")
    if val:
        val_data, val_vocab, mappings = create_data(
            val, opt=opt, vocab=vocab, mappings=mappings, file="val")
    else:
        val_data, val_vocab, mappings = train_data, vocab, mappings
    test_data, test_vocab, mappings = create_data(
        test, opt=opt, vocab=vocab, mappings=mappings, file="test")
    return train_data, val_data, test_data, vocab, mappings
    # # return [], [], [], [], []


# def read_file(fn, valsplit=None):
#     tweets = []
#     data = []
#     # with open(fn, 'r') as json_data:
#     #     for x in json_data:
#     #         data.append(json.loads(x))
#     # i = 0
#     with open(fn, 'r') as json_data:
#         data = json.load(json_data)
#     print(data)
#     for tweet in data:
#         # print(tweet['input'])
#         # tmp = []
#         # tmp = tweet['input']
#         # print(tmp)
#         tmp = ViTokenizer.tokenize(tweet['raw'])
#         src_tweet = tmp.split(" ")
#         # tmp = tweet['output']
#         tmp = ViTokenizer.tokenize(tweet['original'])
#         tgt_tweet = tmp.split(" ")
#         ind = tweet['id']
#         tid = tweet['tid']
#         tweets.append(Tweet(src_tweet, tgt_tweet, tid, ind))

#     #     if(len(src_tweet) != len(tgt_tweet)):
#     #         i = i + 1
#     # print(i)
#     if(valsplit):
#         random.shuffle(tweets)
#         val = tweets[:valsplit]
#         train = tweets[valsplit:]
#         return train, val
#     return tweets, []

# def read_file(fn, valsplit=None):
#     tweets = []
#     with open(fn, 'r') as json_data:
#         data = json.load(json_data)
#     # i = 0
#     for tweet in data:
#         src_tweet = tweet['input']
#         tgt_tweet = tweet['output']
#         ind = tweet['index']
#         tid = tweet['tid']
#         tweets.append(Tweet(src_tweet, tgt_tweet, tid, ind))
#     #     if(len(src_tweet) != len(tgt_tweet)):
#     #         i = i + 1
#     # print(i)
#     if(valsplit):
#         random.shuffle(tweets)
#         val = tweets[:valsplit]
#         train = tweets[valsplit:]
#         return train, val
#     return tweets, []


def read_file(fn, valsplit=None):
    tweets = []
    with open(fn, 'r') as json_data:
        data = json.load(json_data)
    # i = 0
    # data = data.encode('utf-8')

    for tweet in data:
        src_tweet = tweet['raw']
        tgt_tweet = tweet['original']
        ind = tweet['id']
        tid = tweet['tid']
        tweets.append(Tweet(src_tweet, tgt_tweet, tid, ind))
    #     if(len(src_tweet) != len(tgt_tweet)):
    #         i = i + 1
    # print(i)
    if(valsplit):
        random.shuffle(tweets)
        val = tweets[:valsplit]
        train = tweets[valsplit:]
        # print('len train = ', len(train))
        # print('len val = ', len(val))
        return train, val
    return tweets, []
