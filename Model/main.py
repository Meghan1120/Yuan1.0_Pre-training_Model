from data_iterator import GeneratorSet,DataProcGenerator
from train import Trainer,TrainLogger,get_model_parameters
from model import GPT
from config import Config
import jieba,torch
from torch.utils.tensorboard import SummaryWriter

# Get parameters
file_paths = [f'./split_data/0{i:02}.txt' for i in range(1,9)]
generators = GeneratorSet(file_paths).generators

load_batch_size = 32
read_batch_size = 1

# Tokenizer
vocab_path = './vocab/vocab'
f = open(vocab_path,'r',encoding='utf-8')
vocab_data = f.readlines()
f.close()
jieba.load_userdict(vocab_path)

# Data Generator
data_gener = DataProcGenerator(vocab_data,2048,batch_size=read_batch_size)

# increase parameters
input_dim = len(data_gener.TEXT.vocab)
output_dim = len(data_gener.TEXT.vocab)
pad_idx = data_gener.TEXT.vocab.stoi['<pad>']
config = Config()
config.add_info(input_dim,output_dim,pad_idx)

# torch random seed
seed = 10
torch.manual_seed(10)

# pre-training model
model = GPT(config)
optim_adam = torch.optim.Adam(model.parameters(),lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

# total parameters of model
total_params = get_model_parameters(model)
print('total parameters: ',total_params)

# Trainer
trainer = Trainer(load_batch_size,1,optim_adam,criterion,generators,data_gener)
data_splite_info = 'train generators: '+str(trainer.train_gener_len)+'\tvalid generators: '+str(trainer.valid_gener_len)
print(data_splite_info)

# trainlogger
trainlogger = TrainLogger('integry_gpu3+3_yuanformal')
trainlogger.record_log_info()
trainlogger.record_model_param(config,total_params,data_splite_info)
trainlogger.record_train_info(f'seed\t{seed}')

# tensorboard logger
writer = SummaryWriter()


# record the start time
startdate = trainlogger.get_date()
trainlogger.record_train_info('start time of training: '+ startdate)

# start training
trainer.epoch_train(model,trainlogger,writer)

# record the end time
enddate = trainlogger.get_date()
trainlogger.record_train_info('end time of training: '+ enddate)

#save the model parameters
torch.save(model.state_dict(),'./model/yuanmodel3+3.params.bar')
