import torch
from app.dcrnn_model.encode import REncoder, ZEncoder, Decoder
import torch.nn as nn

class DCRNNModel(nn.Module):
    def __init__(self, x_dim, y_dim, r_dim, z_dim, device, init_func=torch.nn.init.normal_):
        super().__init__()
        self.repr_encoder = REncoder(x_dim+y_dim, r_dim) # (x,y)->r
        self.z_encoder = ZEncoder(r_dim, z_dim) # r-> mu, logvar
        self.decoder = Decoder(x_dim+z_dim, y_dim) # (x*, z) -> y*
        self.z_mu_all = 0
        self.z_logvar_all = 0
        self.z_mu_context = 0
        self.z_logvar_context = 0
        self.zs = 0
        self.zdim = z_dim
        self.device = device
    
    def data_to_z_params(self, x, y):
        """Helper to batch together some steps of the process."""
        xy = torch.cat([x,y], dim=1)
        rs = self.repr_encoder(xy)
        r_agg = rs.mean(dim=0) # Average over samples
        return self.z_encoder(r_agg) # Get mean and variance for q(z|...)
    
    def sample_z(self, mu, logvar,n=1):
        """Reparameterisation trick."""
        if n == 1:
            eps = torch.autograd.Variable(logvar.data.new(self.zdim).normal_()).to(self.device)
        else:
            eps = torch.autograd.Variable(logvar.data.new(n, self.zdim).normal_()).to(self.device)
        
        # std = torch.exp(0.5 * logvar)
        std = 0.1+ 0.9*torch.sigmoid(logvar)
        return mu + std * eps

    def KLD_gaussian(self):
        """Analytical KLD between 2 Gaussians."""
        mu_q, logvar_q, mu_p, logvar_p = self.z_mu_all, self.z_logvar_all, self.z_mu_context, self.z_logvar_context

        std_q = 0.1+ 0.9*torch.sigmoid(logvar_q)
        std_p = 0.1+ 0.9*torch.sigmoid(logvar_p)
        p = torch.distributions.Normal(mu_p, std_p)
        q = torch.distributions.Normal(mu_q, std_q)
        return torch.distributions.kl_divergence(p, q).sum()
        

    def forward(self, x_t, x_c, y_c, x_ct, y_ct):
        """
        """
        
        self.z_mu_all, self.z_logvar_all = self.data_to_z_params(x_ct, y_ct)
        self.z_mu_context, self.z_logvar_context = self.data_to_z_params(x_c, y_c)
        self.zs = self.sample_z(self.z_mu_all, self.z_logvar_all)
        return self.decoder(x_t, self.zs)