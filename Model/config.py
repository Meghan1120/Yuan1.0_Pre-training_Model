class Config:
    def __init__(self) -> None:
        self.hid_dim = 3072
        self.n_layers = 40
        self.n_heads = 24
        self.pf_dim = 4*self.hid_dim
        self.dropout = 0.8
    
    def add_info(self,input_dim=None,output_dim=None,pad_idx=None):
        if input_dim != None:
            self.input_dim = input_dim
        if output_dim != None:
            self.output_dim = output_dim
        if pad_idx != None:
            self.pad_idx = pad_idx
