class DataReader:
    def __init__(self,file_path):
        self.file_path = file_path
    
    def __iter__(self):
        with open(self.file_path,'r',encoding='utf-8',errors='replace') as file:
            for line in file:
                if line:
                    yield line.strip('\n')
                else:
                    return

class DataLoader:
    def __init__(self,generator,batch_size):
        self.generator = generator
        self.batch_size = batch_size
    
    def _dataload(self):
        res = []
        t = 0
        for e in self.generator:
            res.append(e)
            t += 1
            if t >= self.batch_size:
                break
        return res

    def __iter__(self):
        res = [1]
        while res != []:
            res = self._dataload()
            yield res
        else:
            return

p = './data/001.txt'
counts = 0
split_p = 488282//40

f = open(p,'r',encoding='utf-8')
rec = open('./split_data/001.txt','a',encoding='utf-8')
rcnt = 1
datareader = iter(DataReader(p))
dataiter = iter(DataLoader(datareader,32))
for lines in dataiter:
    item = [x for x in lines if x != '']
    counts += len(item)
    rec.write('\n' + '\n'.join(item))
    if counts >= split_p:
        counts = 0
        rcnt += 1
        new_p = f'./split_data/0{rcnt:02}.txt'
        rec = open(new_p,'a',encoding='utf-8')

