import torch 
import torch.nn as nn

class GRU():
    def __init__(self, n_inputs, n_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        init_w = lambda*shape : nn.Parameter(torch.randm(*shape)*sigma)
        triple = lambda : (init_w(n_inputs, n_hiddens),
                           init_w(n_hiddens, n_hiddens),
                           nn.Parameter(torch.zeros(n_hiddens)))
        self.w_xz, self.w_hz, self.b_z = triple() # Update gate
        self.w_xr, self.w_hr, self.b_r = triple() # Reset gate
        self.w_xh, self.w_hh, self.b_h = triple() # Hidden gate

    
    def forward(self, inputs, H=None):
        if H is None:
            #initial state (shape= batch_size, n_hidden)
            H = torch.zeros((input.shape[1], self.n_hiddens),
                            device=input.device)
        outputs =[]
        for x in inputs:
            Z = torch.sigmoid(torch.matmul(x, self.w_xz)+
                              torch.matmul(H, self.w_hz) + self.b_z)
            R = torch.sigmoid(torch.matmul(x, self.w_hr) + 
                              torch.matmul(H, self.w_hr) + self.b_r)
            H_t = torch.tanh(torch.matmul(x, self.w_xh) +
                             torch.matmul(R * H, self.w_hh) +self.b_h) 
            H = H_t * (1-Z) + (Z * H)