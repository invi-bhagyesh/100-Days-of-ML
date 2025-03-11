import torch
import torch.nn as nn
class LSTM():
    def __init__(self, n_input, n_hidden, n_output, sigma):
        super().__init__()
        self.save_hyperparameter()

        init_w = lambda*shape: nn.Parameter(torch.randn(*shape)*sigma)
        triple = lambda: (init_w(n_input, n_hidden),
                          init_w(n_hidden, n_hidden),
                          nn.Parameter(torch.zeros(n_hidden)))
        self.w_xi, self.w_hi, self.b_i = triple()
        self.w_xf, self.w_hf, self.b_f = triple()
        self.w_xo, self.w_ho, self.b_o = triple()
        self.w_xc, self.w_hc, self.b_c = triple() #input node


    def forward(self, inputs, H_C= None):
        if H_C is None:
            #initial state (shape= batch_size, n_hidden)
            H = torch.zeros((inputs.shape[1], self.n_hiddens), 
                            devices = inputs.device)
            C = torch.zeros((inputs.shape[1], self.n_hiddens), 
                            devices = inputs.device)
        else: 
            H, C = H_C
        outputs =[]
        for X in inputs:
            I = torch.sigmoid(torch.matmul(X, self.w_xi)+
                              torch.matmul(H, self.w_hi)+ self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.w_xf)+
                              torch.matmul(H, self.w_hf)+ self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.w_xo)+
                              torch.matmul(H, self.w_ho)+ self.b_o)
            C_t = torch.tanh(torch.matmul(X, self.w_xc)+
                               torch.matmul(H, self.w_hc)+ self.b_c)
            
            C = F * C + I * C_t
            H = O * torch .tanh(C)
            outputs.append(H)
        return outputs, (H, C)
    



