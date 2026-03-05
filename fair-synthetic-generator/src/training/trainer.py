import torch
from tqdm import tqdm

class Trainer:
    """Standard trainer for generative models."""
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train(self, dataloader, epochs):
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Assume VAE-like loss for now
                recon_batch, mu, logvar = self.model(batch)
                loss = self.loss_function(recon_batch, batch, mu, logvar)
                
                loss.backward()
                self.optimizer.step()
                
                loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(loss=loss.item())

    def loss_function(self, recon_x, x, mu, logvar):
        # Implementation of combined MSE and KLD loss
        import torch.nn as nn
        BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
