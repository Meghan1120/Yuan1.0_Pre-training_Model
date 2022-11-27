import torch,jieba
from torch.utils.data import IterableDataset,DataLoader
from torchtext.legacy.data import Field,BucketIterator,Example,Dataset

class IterDataSet(IterableDataset):
    def __init__(self,file_path):
        super().__init__()
        self.file_path = file_path
    
    def __iter__(self):
        with open(self.file_path,'r',encoding='utf-8') as file:
            for line in file:
                data = line.strip('\n')
                yield data

class GeneratorSet:
    def __init__(self,file_paths):
        self.file_paths = file_paths
        self.generators = []
        self._make_genertors()
    
    def _make_genertors(self):
        for p in self.file_paths:
            self.generators.append(IterDataSet(p))

class DataProcGenerator:
    def __init__(self,vcb=None,fix_len=None,batch_size=32):
        super().__init__()
        
        self.vocab = vcb
        self.fix_len = fix_len
        self.batch_size = batch_size
        self._build_vocab()
    
    def _tokenizer(self,text):
        tok_list = [tok for tok in jieba.cut(text)]
        return tok_list[:2048]

    def _build_vocab(self):
        self.TEXT = Field(sequential=True,
                            lower=True,
                            fix_length=self.fix_len,
                            tokenize=self._tokenizer,
                            batch_first=True)
        self.TEXT.build_vocab(self.vocab)

    def make_generator(self,data_src,label_src):
        
        fields = [('context',self.TEXT),('label',self.TEXT)]
        examps = []
        for context,label in zip(data_src,label_src):
            examp = Example.fromlist([context,label],fields=fields)
            examps.append(examp)

        train_data = Dataset(examps,fields)
        train_iter = BucketIterator(train_data,
                                    batch_size=self.batch_size,
                                    sort_key=lambda x:len(x.context))
        return train_iter
        
