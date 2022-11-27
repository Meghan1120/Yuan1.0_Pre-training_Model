import torch,datetime,time
from data_iterator import DataLoader

# Get time of an epoch
def epoch_time(start,end):
    elps = end - start
    elps_hous = elps // 3600
    elps = elps - elps_hous*3600
    elps_mins = elps // 60
    elps_secs = elps - elps_mins*60
    return elps_hous,elps_mins,elps_secs

# Get total parameters of a model
def get_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

# Trainer class
class Trainer:
    def __init__(self,batch_size,clip,optimizer,criterion,generators,data_generator,split_rate=1):
        self.optimizer = optimizer
        self.criterion  = criterion
        self.generators = generators
        self.data_generator = data_generator
        self.clip = clip
        self.batch_size = batch_size
        self.split_rate = split_rate
        self._split_gener()

    def _split_gener(self):
        split_point = int(len(self.generators)*self.split_rate) + 1
        self.train_gener = self.generators[:split_point]
        self.valid_gener = self.generators[split_point:]
        self.train_gener_len = len(self.train_gener)
        self.valid_gener_len = len(self.valid_gener)
    
    def _train_iter(self,train_iter,model):
        model.train()
        iter_loss = 0
        len_iter = len(train_iter)
        len_sm = 0

        for i,batch in enumerate(train_iter):
            x = batch.context
            y = batch.label

            self.optimizer.zero_grad()
            f = True
            if i >= 31:
                f = False
            output = model(x,f).permute(0,2,1)

            self.criterion = self.criterion.cuda(1)
            loss = self.criterion(output.cuda(1),y.cuda(1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),self.clip)
            self.optimizer.step()
            if f == False:
                iter_loss += loss.item()
                len_sm += 1
        return iter_loss,len_sm,len_iter
    
    def epoch_train(self,model,logger=None,writer=None):
        
        totalcnt = 0
        i = 0
        for gener in self.train_gener:
            epoch_loss = 0
            denom = 0
            i += 1
            start = time.time()
            dataiter = DataLoader(gener,batch_size=self.batch_size)
            for x in dataiter:
                s = time.time()
                train_iter = self.data_generator.make_generator(x,x)

                iter_loss,len_sm,len_iter = self._train_iter(train_iter,model)
                epoch_loss += iter_loss
                denom += len_sm
                totalcnt += len_iter
                
                e = time.time()
                iterl = iter_loss/len_sm if len_sm != 0 else 0
                it_hous,it_mins,it_secs = epoch_time(s,e)
                # tensorboard record information of training per iterator
                if writer != None:
                    writer.add_scalar(f'Epoch {i:02} Train/Loss',iterl,totalcnt*self.data_generator.batch_size)
                    writer.add_scalar('Total Loss process',iter_loss,totalcnt*self.data_generator.batch_size)
                info = f'training | train loss {iterl:.3f}\t| samples count {totalcnt*self.data_generator.batch_size}\t| time {it_hous:02}h {it_mins:02}m {it_secs:.3f}s'
                print(info,end='\r')
            print('                                                                                                                       ',end='\r')
            
            end = time.time()
            # record information of training per epoch
            epc_hous,epc_mins,epc_secs = epoch_time(start,end)
            info = f'epoch {i:02}\t| time {epc_hous:02}h {epc_mins:02}m {epc_secs:.3f}s\t| train loss: {epoch_loss/denom:.3f}'
            # print(info)
            if logger != None:
                logger.record_train_info(info)
        

# TrainLogger class
class TrainLogger:
    def __init__(self,log_name,dir_path='./log/'):
        self.log_name = log_name
        self.dir_path = dir_path
        self.date = self.get_date()
        self._create_file()
        self.open_file()
    
    def get_date(self):
        return str(datetime.datetime.now())

    def record_log_info(self):
        self.file.write('log name:\t{}\n'.format(self.log_name))
        self.file.write('time file created:\t{}\n\n'.format(self.date))
    
    def record_model_param(self,config,total_params,data_splite_info):
        info = f'hid_dim\t{config.hid_dim}\nn_layers\t{config.n_layers}\nn_heads\t{config.n_heads}\npf_dim\t{config.pf_dim}\ndropout\t{config.dropout}\ninput_dim\t{config.input_dim}\noutput_dim\t{config.output_dim}\npad_idx\t{config.pad_idx}\n'
        self.file.write(info+'\n')
        self.file.write('total params of model\t'+str(total_params)+'\n')
        self.file.write(data_splite_info+'\n')
        
    
    def record_train_info(self,info):
        self.file.write(info+'\n')
    
    def close_file(self):
        self.file.close()
    
    def open_file(self):
        self.file = open(self.log_path,'a',encoding='utf-8')
    
    def _create_file(self):
        self.log_path = self.dir_path + self.log_name + self.date.replace(' ','_').replace(':','-')[:-7]
        file = open(self.log_path,'w',encoding='utf-8')
        file.close()
