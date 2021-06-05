import re
import torch
import numpy as np
from tqdm.auto import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokens=None, margin=1, t=None, java_path=None):
        if tokens is None:
            self.oov = None
            self.tokens = {1: 'oov'}
        else:
            self.oov = 1
            self.tokens = tokens

        self.tokens_set = set(self.tokens.keys())

        with open(path, 'r', encoding='utf8') as file:
            print('loading data...')
            lines = file.read().split('\n\n')
            lines = [self.split_words(x) for x in lines if len(x) > 0]

            if margin > 0:
                print('removing length outliers...')
                l = [len(x[0]) for x in lines]
                b1p = np.quantile(l, margin/100, interpolation='nearest')
                t1p = np.quantile(l, (100 - margin)/100, interpolation='nearest')
                lines = [x for x in lines if b1p < len(x[0]) < t1p]

            self.lengths = np.array([len(x[0]) for x in lines])
            self.max_length = self.lengths.max()

            
            x,y = list(zip(*lines))
            
            if t is None:
                print('POS tagging...')
                print('this is going to be long...')

                from parsivar import POSTagger
                import ray
                ray.init(ignore_reinit_error=True)

                @ray.remote
                def transform(x):
                    return POSTagger(tagging_model="wapiti", jdk_variable_path=java_path).parse(x)
                futures = [transform.remote(i) for i in x]
                t = ray.get(futures)
            self.t = t

            for a,b in zip(x,self.t):
                assert len(a) == len(b)
            
            print('embedding data...')
            self.X, self.Y = self.pad_all(x, self.t, y)

    def split_words(self, line):
        tokens = [(re.sub('\sgen_.*', '',  x), 1 if x.endswith('positive') else 0) for x in line.split('\n')]
        return list(zip(*[x for x in tokens if len(x[0]) > 0]))

    def pad_all(self, x, t, y):
        data = np.zeros((len(x), self.max_length)).astype(int)
        labels = np.zeros((len(x), self.max_length))
        for i, (line, pos, label) in enumerate(zip(tqdm(x), t, y)):
            data[i,:len(line)] = self.embed(line, pos)
            labels[i,:len(label)] = np.array(label)
        return data, labels

    def embed(self, tokens, pos):
        t = [self.tokens[x] if x in self.tokens_set else self.append_token(x) for x in [x[1] for x in pos]]
        return np.array(t)

    def append_token(self, token):
        if self.oov is None:
            self.tokens[token] = len(self.tokens) + 1
            self.tokens_set.add(token)
            return len(self.tokens)
        return 1

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, index):    
        return self.lengths[index], self.X[index], self.Y[index]
