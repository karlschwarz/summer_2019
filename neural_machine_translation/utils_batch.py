import unicodedata
import re
import jieba

import torch
from torch.utils.data import DataLoader, Dataset
import h5py 
import pandas
import random
import numpy as np

SOS_token = 0
EOS_token = 1
PAD_token = 2

def normalizeChinese(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[，。？！#]", r"", s)
    s = re.sub(r"[.!?,]", r" ", s)
    s = re.sub(r"'", r"", s)
    return s

def readLangsMand(file_path, lang1, lang2, reverse=False):
    print('Reading Lines...')
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeChinese(s) for s in l.split('\t')[:2]] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def prepareDataMand(file_path, lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangsMand(file_path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "<PAD>"}
        self.n_words = 3
        self.len_list = []
        
    def addSentence(self, sentence):
        if self.name == 'zh':
            word_list = list(jieba.cut(sentence, cut_all=False))
            for word in word_list:
                self.addWord(word)
        else:
            word_list = sentence.split(' ')
            for word in word_list:
                self.addWord(word)
        self.len_list.append(len(word_list))
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def max_length(self):
        max_len = max(self.len_list)
        return max_len
    
class ENZHDataset(Dataset):
    """
    ENZH Dataset used for NMT.
    """
    def __init__(self, input_lang, output_lang, pairs, maxlen, padding=True):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs
        self.padding = padding
        self.max_length = maxlen
        
    def __len__(self):
        num_samples = len(self.pairs)
        return num_samples
    
    def indexesFromSentence(self, lang, sentence):
        ndexes_list = []
        if lang.name == 'zh':
            word_list = [lang.word2index[word] for word in jieba.cut(sentence, cut_all=False)]
            return word_list
        else:
            return [lang.word2index[word] for word in sentence.split(' ')]
    
    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        if self.padding and self.max_length - len(indexes) > 0:
            indexes.extend([PAD_token] * (self.max_length - len(indexes)))
        return torch.tensor(indexes, dtype=torch.long)

    def tensorFromPair(self, pair):
        input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
        target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
        return (input_tensor, target_tensor)

    def __getitem__(self, global_idx):   
        pair_i = self.pairs[global_idx]
        #print(pair_i)
        input_tensor, target_tensor = self.tensorFromPair(pair_i)
        return input_tensor, target_tensor
    
def enzh_loader(file_path, batch_size, maxlen, padding=True):
    """
    This is a function used to create a LLCFS data loader.
    
    Args:
        file_path: (str) path of data file;
        batch_size: (int) batch size;
        padding: (bool) flag on padding option
        train: (bool) flag of train or valid mode;
        
    Returns:
        loader: (Dataloader) enzh Dataloader.
    """
    input_lang, output_lang, pairs = prepareDataMand(file_path, 'en', 'zh')
    num_samples = len(pairs)
    num_valid = num_samples // 10
    random.shuffle(pairs)
    pairs_train, pairs_valid = pairs[:-num_valid], pairs[-num_valid:]
    ##########################################################
    enzh_dataset_train = ENZHDataset(
        input_lang, 
        output_lang, 
        pairs_train, 
        maxlen,
        padding=padding
    )
    enzh_loader_train = torch.utils.data.DataLoader(
        enzh_dataset_train, 
        batch_size=batch_size,
        drop_last=True,
        shuffle=True, 
        num_workers=2,
        pin_memory=False,
    )   
    print(f"{num_samples-num_valid} samples in training dataset.")
    ##########################################################
    ##########################################################
    enzh_dataset_test = ENZHDataset(
        input_lang, 
        output_lang, 
        pairs_valid, 
        maxlen,
        padding=padding
    )
    enzh_loader_test = torch.utils.data.DataLoader(
        enzh_dataset_test, 
        batch_size=batch_size,
        drop_last=True,
        shuffle=True, 
        num_workers=2,
        pin_memory=False,
    )
    print(f"{num_valid} samples in valid dataset.")
    #########################################################
    return enzh_loader_train, enzh_loader_test, pairs_valid

