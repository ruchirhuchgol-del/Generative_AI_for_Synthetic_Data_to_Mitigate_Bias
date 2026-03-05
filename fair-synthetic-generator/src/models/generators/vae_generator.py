import torch
import torch.nn as nn
from src.core.base_module import BaseGenerator

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class VAEGenerator(BaseGenerator):
    """VAE-based synthetic data generator."""
    def __init__(self, config=None):
        super().__init__(config)
        self.input_dim = config.get('input_dim', 10)
        self.latent_dim = config.get('latent_dim', 2)
        self.model = VAE(self.input_dim, self.latent_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-3))

    def train(self, dataloader, epochs, **kwargs):
        self.model.train()
        for epoch in range(epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(batch)
                loss = self.loss_function(recon_batch, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def generate(self, num_samples, **kwargs):
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            samples = self.model.decoder(z)
        return samples
