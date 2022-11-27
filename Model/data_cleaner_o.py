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


files_paths = [f'./data/0{i:02}_c.txt' for i in range(1,19)]
record_paths = [f'./clean_data/0{i:02}.txt' for i in range(1,19)]

for fp,rp in zip(files_paths,record_paths):
    datareader = iter(DataReader(fp))
    f = open(rp,'a',encoding='utf-8')
    i = 0
    dataiter = iter(DataLoader(datareader,32))
    for lines in dataiter:
        item = [x for x in lines if x != '']
        f.write('\n'+'\n'.join(item).replace('<n>',''))
        i += 32
        print('{} finished {} lines'.format(fp,i))
