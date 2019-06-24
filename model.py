import torch
import torch.nn as nn

class NADE(nn.Module):
    """NADE for binary MNIST"""
    def __init__(self, input_dim, hidden_dim):
        super(NADE, self).__init__()
        self.D = input_dim
        self.H = hidden_dim
        self.params = nn.ParameterDict({
            "V" : nn.Parameter(torch.randn(self.D, self.H)),
            "b" : nn.Parameter(torch.zeros(self.D)),
            "W" : nn.Parameter(torch.randn(self.H, self.D)),
            "c" : nn.Parameter(torch.zeros(1, self.H)),
        })
        nn.init.xavier_normal_(self.params["V"])
        nn.init.xavier_normal_(self.params["W"])
        
    def forward(self, x):
        # a0: (B, H)
        a_d = self.params["c"].expand(x.size(0), -1)
        # Compute p(x)
        x_hat = self._cal_prob(a_d, x)
        
        return x_hat
    
    def _cal_prob(self, a_d, x, sample=False):
        """
        assert 'x = None' when sampling
        Parameters:
         - a_d : (B, H)
         - x : (B, D)
         
        Return:
         - x_hat: (B, D), estimated probability dist. of batch data
        """
        if sample:
            assert (x is None), "No input for sampling as first time"
        
        x_hat = []  # (B, 1) x D
        xs = []
        for d in range(self.D):
            # h_d: (B, H)
            h_d = torch.sigmoid(a_d)
            # p_hat: (B, H) x (H, 1) + (B, 1) = (B, 1)
            p_hat = torch.sigmoid(h_d.mm(self.params["V"][d:d+1, :].t() + self.params["b"][d:d+1]))
            x_hat.append(p_hat)
            
            if sample:
                # random sample x: (B, 1) > a_{d+1}: (B, 1) x (1, H)
                x = torch.distributions.Bernoulli(probs=p_hat).sample()
                xs.append(x)
                a_d = x.mm(self.params["W"][:, d:d+1].t()) + self.params["c"]
                
            else:
                # a_{d+1}: (B, 1) x (1, H)
                a_d = x[:, d:d+1].mm(self.params["W"][:, d:d+1].t()) + self.params["c"]
        
        # x_hat: (B, D), estimated probability dist. of batch data
        x_hat = torch.cat(x_hat, 1)
        if sample:
            xs = torch.cat(xs, 1)
            return x_hat, xs
        return x_hat
    
    def _cal_nll(self, x_hat, x):
        nll_loss = -1 * ( x*torch.log(x_hat + 1e-10) + (1-x)*torch.log(1-x_hat + 1e-10))
        return nll_loss
    
    def sample(self, n=1, only_prob=True):
        a_d = self.params["c"].expand(n, -1)  # (n, H)
        # Compute p(x)
        x_hat, xs = self._cal_prob(a_d, x=None, sample=True)
        nll_loss = self._cal_nll(x_hat, xs)
        return (x_hat, xs, nll_loss)