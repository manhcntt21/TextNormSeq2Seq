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
class DataLoader(object):
    def __init__(self, tweets, vocab, mappings, opt):
        self.opt = opt
        self.prox_arr = self.get_prox_keys()
        self.change_sign = self.get_change_sign()
        self.repleace_character = self.get_repleace_character()
        self.phonology_vietnamese = self.get_phonology_vietnamese()
        self.mappings = mappings if mappings else {}
        self.tweets, self.source_vocab, self.target_vocab = self.load_data(tweets)
        if(vocab):
            self.source_vocab = vocab["src"]
            self.target_vocab =  vocab["tgt"]
        self.ret = self.encode_tweets()

    def tweets_toIdx(self):
        for tweet in self.tweets:
            input = copy.deepcopy(tweet.input)
            if(self.opt.correct_unique_mappings):
                for index, wi in enumerate(input):
                    mapping = self.mappings.get(wi)
                    if mapping and len(mapping)==1 and list(mapping)[0]!=wi:
                        input[index] = list(mapping)[0]
            tweet.set_inputidx(self.source_vocab.to_indices(input, bosWord=self.opt.bos, eosWord=self.opt.eos))
            tweet.set_outputidx(self.target_vocab.to_indices(tweet.output, bosWord=self.opt.bos, eosWord=self.opt.eos))

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
                    self.mappings[inp_i[k].lower()] =  set()
                    self.mappings[inp_i[k].lower()].add(inp_o[k].lower())

        if(self.opt.self_tok==lib.constants.SELF):
            for i in range(len(inp_i)):
                if(inp_i[i].lower()==inp_o[i].lower()):
                    inp_o[i] =  self.opt.self_tok

        if(self.opt.input=='char'):
            inp_i = list('#'.join(inp_i))
            inp_o = list('#'.join(inp_o))

        return inp_i, inp_o


    def load_data(self, tweets):
        source_vocab = Dict(vocab_size=self.opt.vocab_size, bosWord=self.opt.bos, eosWord=self.opt.eos)
        target_vocab = Dict(vocab_size=self.opt.vocab_size, bosWord=self.opt.bos, eosWord=self.opt.eos)
        if(self.opt.share_vocab):
            target_vocab = source_vocab
        processor = Preprocessor()
        #for test the mappings are predefined and for all other inputs except word level we dont need them, so no updates
        update_mappings = not self.mappings and self.opt.input=='word'
        word_tweets = []
        regex  = re.compile(ur"^[aàảãáạăằẳẵắặâầẩẫấậbcdđeèẻẽéẹêềểễếệfghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợpqrstuùủũúụưừửữứựvwxyỳỷỹýỵz0123456789_]+$",re.UNICODE)
        for tweet in tweets:
            inp_i, pos_i = processor.run(tweet.input,self.opt.lowercase)
            inp_o, pos_o = processor.run(tweet.output, self.opt.lowercase)
            if(self.opt.input == 'spelling'): #character model word2word corrections
                for iword, oword in zip(inp_i, inp_o):
                    # iword = iword.encode('ascii','ignore')
                    # oword = oword.encode('ascii','ignore')
                    # iword = iword.encode('utf-8')
                    # oword = oword.encode('utf-8')
                                    # print('len cua word',len(iword))
                                    # print('word ',iword)
                    match1 = re.search(regex,iword)
                    match2 = re.search(regex,oword)
                    if iword and oword and match1 and match2:
                        if iword == oword and len(iword)>1 and len(oword)>1 and not any(c.isdigit() for c in iword) and  not any(c.isdigit() for c in oword):
                            if random.random() > 0.9 and not self.opt.data_augm:
                                continue
                        iwordv, owordv = self.vector_repr(iword, oword, update_mappings)
                        source_vocab.add_words(iwordv)
                        target_vocab.add_words(owordv)
                        tweet.set_input(iwordv)
                        tweet.set_output(owordv)
                        word_tweets.append(copy.deepcopy(tweet))
                        if(self.opt.data_augm):
                            if random.random() > (1 - self.opt.noise_ratio):
                                if iword == oword and len(iword)>1 and len(oword)>1 and not any(c.isdigit() for c in iword) and  not any(c.isdigit() for c in oword):
                                    iword = self.add_noise(iword)
                                    # aa_aa = iword
                                    # print('word sau khi add noise', unichr(int(aa_aa,16)))
                                    if(iword == '' or iword == ' '):
                                        continue
                                    iwordv, owordv = self.vector_repr(iword, oword, update_mappings)
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
        ## voi words model thi idx_to_label cua source vs target = nhau 
        # print(source_vocab.idx_to_label[1])
        # # print(target_vocab.idx_to_label)
        # print(len(source_vocab.idx_to_label))

        if(self.opt.share_vocab):
            assert source_vocab.idx_to_label == target_vocab.idx_to_label
        return tweets, source_vocab, target_vocab


    def add_noise(self, word):
        """
            There are 7 kinds of errors we can introduce for data aug:
            0) forget to "type" a char
            1) swap the placement of two chars
            2) if the word ends in u, y, s, r, extend the last char
            3) if vowel in sentence extend vowel (o, u, e, a, i)
            4-6) misplaced or missing " ' "
            9-12) keyboard errors
            7) nhung chu cai co phat am giong nhau
            8)
        """
        # try:
        #     unicode(word, "ascii")
        # except UnicodeError:
        #     word = unicode(word, "utf-8")
        # word = google.searchGoogle(word).encode("utf-8")
        # word = word.decode('uft-8')
        # print(type(word)) % unicode
        # for i in word:
        #     print(i)
        # print('')
        # word = word.encode('ascii','ignore')
        # print('len cua word',len(word))
        # print('word ',word)
        i = random.randint(0,len(word)-1)
        op = random.randint(0, 30)
        if op == 0:
            return word[:i] + word[i+1:]
        if op == 1:
            i += 1
            return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]
        if op == 2:
            # l =word[:-1]
            # if l == 'u' or l == 'y' or l == 's' or l == 'r' or l == 'a' or l == 'o' or l == 'i':
            #     return word + random.randint(1, 5) * l
            return word # ko lam gi 
        if op == 3:
            a = word.find('a')
            e = word.find('e')
            i = word.find('i')
            o = word.find('o')
            u = word.find('u')
            idx = max([a,e,i,o,u])
            if idx != -1:
                return word[:idx] +  random.randint(1, 5) * word[idx] + word[idx:]
        if op == 4:
            idx = word.find("'")
            if idx != -1:
                return word[:idx] + word[idx+1:] + word[idx]
        if op == 5:
            idx = word.find("'")
            if idx != -1:
                return word[:idx-1] + word[idx:idx+1] + word[idx-1:idx] + word[idx+1:]
        if op == 6:
            idx = word.find("'")
            if idx != -1:
                return word[:idx] + word[idx+1:]
        if op == 7 or op == 8:
            try:
                # print(op)
                return word[:i] + random.choice(self.repleace_character[word[i]]) + word[i+1:] # thay doi dau
            except:
                return word
            # if not bool(self.repleace_character[word[i]]):
            #     return word
            # 
        if op == 9 or op == 10:
            try:
                return word[:i] + random.choice(self.change_sign[word[i]]) + word[i+1:]
            except:
                return word
            # if not bool(self.change_sign[word[i]]):
            #     return word
        if op == 11 or op == 12 or op == 13 or op == 14 or op == 15 or op == 16 or op == 17 or op == 18 or op == 19 or op == 20 or op == 21 or op == 22 or \
            op == 23 or op == 24 or op == 25 or op == 26 or op == 27 or op == 28:
            if len(word) - 1 - i == 4:
                try:
                    # print('word truoc khi add noise', word)
                    return word[:i] + self.phonology_vietnamese[word[i]+word[i+1]+word[i+2]+word[i+3]+word[i+1]+word[i+4]] + word[i+1:]
                except:
                    return word
                # if not bool(self.phonology_vietnamese[word[i]+word[i+1]+word[i+2]+word[i+3]+word[i+1]+word[i+4]]):
                    # return word
            elif len(word) - 1 - i == 3:
                try:
                    # print('word truoc khi add noise', word)
                    return word[:i] + self.phonology_vietnamese[word[i]+word[i+1]+word[i+2]] + word[i+1:]
                except:
                    return word
                # if not bool(self.phonology_vietnamese[word[i]+word[i+1]+word[i+2]]):
                #     return word  
            elif len(word) - 1 - i == 2:
                try:                    
                    # print('word satruoc khi add noise', word)
                    return word[:i] + self.phonology_vietnamese[word[i]+word[i+1]] + word[i+1:]
                except :
                    return word
        try:
            return word[:i] + random.choice(self.prox_arr[word[i]]) + word[i+1:] #default is keyboard errors
        except :
            # print(word)
            return word
        
    def get_phonology_vietnamese(self):
        phonology_vietnamese = {}
        phonology_vietnamese['inh'] = ['in']
        phonology_vietnamese['ênh'] = ['ên']
        phonology_vietnamese['êch'] = ['ết']
        phonology_vietnamese['ich'] = ['ít']
        phonology_vietnamese['ăng'] = ['ăn']
        phonology_vietnamese['ang'] = ['an']
        phonology_vietnamese['âng'] = ['ân']
        phonology_vietnamese['ưng'] = ['ưn']
        phonology_vietnamese['ông'] = ['ôn']
        phonology_vietnamese['ung'] = ['un']
        phonology_vietnamese['iêc'] = ['iêt']
        phonology_vietnamese['ước'] = ['ươt']
        phonology_vietnamese['uôc'] = ['uôt']

        phonology_vietnamese['iêng'] = ['iên']
        phonology_vietnamese['ương'] = ['ươn']
        phonology_vietnamese['uông'] = ['uôn']

        phonology_vietnamese['ăc'] = ['ắt']
        phonology_vietnamese['ac'] = ['at']
        phonology_vietnamese['âc'] = ['ât']
        phonology_vietnamese['ưc'] = ['ưt']
        phonology_vietnamese['ôc'] = ['ôt']
        phonology_vietnamese['uc'] = ['ut']
        return phonology_vietnamese


    def get_change_sign(self):
        """
            # sign['a'] = ['\xc3\xa0', '\xc3\xa1', '\xc3\xa2', '\xc3\xa3', '\xe1\xba\xa1', '\xe1\xba\xa3', '\xe1\xba\xa5', '\xe1\xba\xa5', '\xe1\xba\xad', '\xe1\xba\xaf', '\xe1\xba\xb7'] # a à á â ã ạ ả ấ  ầ  ậ ắ  ặ 
            # sign['e'] = ['\xc3\xa8', '\xc3\xa9', '\xc3\xaa', '\xe1\xba\xb9' , '\xe1\xba\xbb', '\xe1\xba\xbd', '\xe1\xba\xbf', '\xe1\xbb\x81', '\xe1\xbb\x83', '\xe1\xbb\x85',' \xe1\xbb\x87'] # è é ê  ẹ ẻ ẽ ế  ề  ể  ễ  ệ 
            # sign['i'] = ['\xc3\xac', '\xc3\xad', '\xe1\xbb\x89', '\xe1\xbb\x8b'] # ì í ỉ ị 
            # sign['o'] = ['\xc3\xb2', '\xc3\xb3', '\xc3\xb4', '\xc3\xb5', '\xe1\xbb\x8d', '\xe1\xbb\x8f', '\xe1\xbb\x91', '\xe1\xbb\x93', '\xe1\xbb\x95' , '\xe1\xbb\x97', '\xe1\xbb\x99', '\xe1\xbb\x9b', '\xe1\xbb\x9d', '\xe1\xbb\x9f', '\xe1\xbb\xa1', '\xe1\xbb\xa3'] # ò ó ô  õ ọ ỏ  ố ồ  ổ ộ ớ ờ ỡ ợ 
            # sign['u'] = ['\xc3\xb9', '\xc3\xba', '\xe1\xbb\xa5', '\xe1\xbb\xa7', '\xe1\xbb\xa9', '\xe1\xbb\xab', '\xe1\xbb\xad', '\xe1\xbb\xaf', '\xe1\xbb\xb1'] # ù ú ụ ủ ứ ừ ữ ự 
            # sign['y'] = ['\xc3\xbd', '\xe1\xbb\xb3', '\xe1\xbb\xb5', '\xe1\xbb\xb7']  # ý ỳ ỵ ỷ 
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
        return sign_

    def get_repleace_character(self):
        repleace_character = {}
        repleace_character['ch'] = ['tr']
        repleace_character['tr'] = ['ch']
        repleace_character['l'] = ['n']
        repleace_character['n'] = ['l']
        repleace_character['x'] = ['s']
        repleace_character['s'] = ['x']
        repleace_character['r'] = ['d', 'gi']
        repleace_character['d'] = ['r', 'gi']
        repleace_character['gi'] = ['d', 'r']
        repleace_character['c'] = ['q', 'k']
        repleace_character['k'] = ['q', 'c']
        repleace_character['q'] = ['c', 'k']
        repleace_character['i'] = ['y']
        repleace_character['y'] = ['i']
        repleace_character['_'] = ['_']
        return repleace_character
        
    def get_prox_keys(self):
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
        return array_prox



def create_data(data, opt, vocab=None, mappings=None):
    dataload = DataLoader(data, vocab=vocab, mappings=mappings, opt=opt)
    vocab = {}
    vocab['src'] = dataload.source_vocab
    vocab['tgt'] = dataload.target_vocab
    return dataload.ret, vocab, dataload.mappings
    # return {}, {}, {}


def create_datasets(opt):
    train, val = read_file(opt.traindata, opt.valsplit)
    test, _ = read_file(opt.testdata)
    train_data, vocab, mappings = create_data(train, opt=opt)
    if val: val_data, val_vocab, mappings = create_data(val, opt=opt, vocab=vocab, mappings=mappings)
    else: val_data, val_vocab, mappings = train_data, vocab, mappings
    test_data, test_vocab, mappings = create_data(test, opt=opt, vocab=vocab, mappings=mappings)
    return train_data, val_data, test_data, vocab, mappings
    # return [], [], [], [], []




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
        return train, val
    return tweets, []
