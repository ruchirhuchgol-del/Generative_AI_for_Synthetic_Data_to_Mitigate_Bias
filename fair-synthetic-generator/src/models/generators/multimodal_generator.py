"""
Multimodal Generator
====================

Multimodal generators for joint synthetic data generation across modalities.
Supports tabular, text, and image generation with cross-modal consistency.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.generators.base_generator import (
    BaseGenerator,
    ConditionalGenerator,
    FairGenerator,
    LatentSpaceMixin,
    ReconstructionMixin
)
from src.models.generators.vae_generator import VAEGenerator, ConditionalVAEGenerator
from src.models.generators.gan_generator import GANGenerator, ConditionalGANGenerator
from src.models.generators.diffusion_generator import DiffusionGenerator, ConditionalDiffusionGenerator


class MultimodalGenerator(BaseGenerator, LatentSpaceMixin, ReconstructionMixin):
    """
    Multimodal Generator for joint tabular, text, and image generation.
    
    Architecture:
        Shared Latent z → Modality Heads → Cross-Modal Fusion → Modality Decoders
        
    Features:
    - Shared latent space across modalities
    - Cross-modal attention for consistency
    - Independent modality decoders
    - Optional encoder for reconstruction
    
    Example:
        >>> generator = MultimodalGenerator(
        ...     latent_dim=512,
        ...     tabular_decoder=tab_decoder,
        ...     text_decoder=text_decoder,
        ...     image_decoder=img_decoder,
        ...     use_cross_modal_attention=True
        ... )
        >>> outputs = generator.generate(100)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        encoder: Optional[nn.Module] = None,
        tabular_decoder: Optional[nn.Module] = None,
        text_decoder: Optional[nn.Module] = None,
        image_decoder: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
        use_cross_modal_attention: bool = True,
        fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        name: str = "multimodal_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multimodal generator.
        
        Args:
            latent_dim: Shared latent dimension
            encoder: Optional encoder for reconstruction
            tabular_decoder: Tabular decoder module
            text_decoder: Text decoder module
            image_decoder: Image decoder module
            fusion_module: Custom fusion module
            use_cross_modal_attention: Use cross-modal attention
            fusion_layers: Number of fusion layers
            num_heads: Attention heads
            dropout: Dropout rate
            name: Generator name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.encoder = encoder
        self.tabular_decoder = tabular_decoder
        self.text_decoder = text_decoder
        self.image_decoder = image_decoder
        
        # Determine modalities
        self.modalities = []
        if tabular_decoder is not None:
            self.modalities.append("tabular")
        if text_decoder is not None:
            self.modalities.append("text")
        if image_decoder is not None:
            self.modalities.append("image")
        
        # Modality-specific projection heads
        self.modality_heads = nn.ModuleDict()
        for modality in self.modalities:
            self.modality_heads[modality] = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Cross-modal fusion
        self.use_cross_modal_attention = use_cross_modal_attention
        if use_cross_modal_attention and fusion_module is None:
            self.fusion_module = CrossModalFusion(
                latent_dim=latent_dim,
                num_modalities=len(self.modalities),
                num_layers=fusion_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.fusion_module = fusion_module
            
    def sample_latent(
        self, 
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample from prior."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(n_samples, self._latent_dim, device=device)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multimodal samples.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            modalities: Subset of modalities to generate
            
        Returns:
            Dictionary of generated samples per modality
        """
        if modalities is None:
            modalities = self.modalities
            
        # Sample shared latent
        z = self.sample_latent(n_samples)
        
        # Project to modality-specific latents
        modality_latents = {}
        for modality in modalities:
            if modality in self.modality_heads:
                modality_latents[modality] = self.modality_heads[modality](z)
        
        # Apply cross-modal fusion
        if self.use_cross_modal_attention and self.fusion_module is not None:
            latent_list = [modality_latents[m] for m in modalities if m in modality_latents]
            if len(latent_list) > 1:
                fused_latents = self.fusion_module(latent_list)
                for i, modality in enumerate([m for m in modalities if m in modality_latents]):
                    modality_latents[modality] = fused_latents[i]
        
        # Decode each modality
        outputs = {}
        
        if "tabular" in modalities and self.tabular_decoder is not None:
            outputs["tabular"] = self.tabular_decoder(modality_latents["tabular"])
            
        if "text" in modalities and self.text_decoder is not None:
            outputs["text"] = self.text_decoder(modality_latents["text"])
            
        if "image" in modalities and self.image_decoder is not None:
            outputs["image"] = self.image_decoder(modality_latents["image"])
        
        return outputs
    
    def encode(
        self, 
        x: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode multimodal input to shared latent.
        
        Args:
            x: Dictionary of input tensors per modality
            
        Returns:
            Shared latent representation
        """
        if self.encoder is None:
            raise NotImplementedError("No encoder provided for reconstruction")
        return self.encoder(x, **kwargs)
    
    def decode(
        self, 
        z: torch.Tensor,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to multimodal output.
        
        Args:
            z: Latent tensor
            modalities: Modalities to decode
            
        Returns:
            Dictionary of decoded outputs
        """
        return self.generate(z.shape[0], modalities=modalities, **kwargs)
    
    def reconstruct(
        self,
        x: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Reconstruct multimodal input.
        
        Args:
            x: Dictionary of input tensors
            
        Returns:
            Tuple of (reconstructed outputs, latent)
        """
        z = self.encode(x, **kwargs)
        x_recon = self.decode(z, **kwargs)
        return x_recon, z
    
    def compute_loss(
        self,
        x: Dict[str, torch.Tensor],
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reconstruction loss.
        
        Args:
            x: Dictionary of input tensors
            return_components: Return loss components
            
        Returns:
            Total loss or dictionary
        """
        # Encode
        z = self.encode(x, **kwargs)
        
        # Decode
        x_recon = self.decode(z, **kwargs)
        
        # Compute loss per modality
        losses = {}
        total_loss = 0.0
        
        for modality in self.modalities:
            if modality in x and modality in x_recon:
                if modality == "tabular":
                    loss = self._compute_tabular_loss(x[modality], x_recon[modality])
                elif modality == "text":
                    loss = self._compute_text_loss(x[modality], x_recon[modality])
                elif modality == "image":
                    loss = self._compute_image_loss(x[modality], x_recon[modality])
                else:
                    loss = F.mse_loss(x_recon[modality], x[modality])
                
                losses[f"{modality}_loss"] = loss
                total_loss = total_loss + loss
        
        losses["total_loss"] = total_loss
        
        if return_components:
            return losses
        return total_loss
    
    def _compute_tabular_loss(
        self,
        x: Dict[str, torch.Tensor],
        x_recon: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute tabular reconstruction loss."""
        loss = 0.0
        
        if "numerical" in x and "numerical" in x_recon:
            loss = loss + F.mse_loss(x_recon["numerical"], x["numerical"])
        
        if "categorical" in x and "categorical" in x_recon:
            for key in x["categorical"]:
                if key in x_recon["categorical"]:
                    loss = loss + F.cross_entropy(
                        x_recon["categorical"][key],
                        x["categorical"][key]
                    )
        
        return loss
    
    def _compute_text_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor
    ) -> torch.Tensor:
        """Compute text reconstruction loss."""
        return F.cross_entropy(
            x_recon.reshape(-1, x_recon.shape[-1]),
            x.reshape(-1)
        )
    
    def _compute_image_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor
    ) -> torch.Tensor:
        """Compute image reconstruction loss."""
        return F.mse_loss(x_recon, x)
    
    def forward(
        self,
        x: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returns reconstruction."""
        x_recon, z = self.reconstruct(x, **kwargs)
        return x_recon


class CrossModalFusion(nn.Module):
    """
    Cross-modal attention fusion module.
    
    Enables information exchange between modalities for coherent generation.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_modalities: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        self.num_layers = num_layers
        
        # Cross-attention layers
        self.cross_attentions = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
                for _ in range(num_modalities)
            ])
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(latent_dim)
                for _ in range(num_modalities)
            ])
            for _ in range(num_layers)
        ])
        
        # Feedforward layers
        self.ff_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim * 4, latent_dim),
                    nn.Dropout(dropout)
                )
                for _ in range(num_modalities)
            ])
            for _ in range(num_layers)
        ])
        
        self.ff_norms = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(latent_dim)
                for _ in range(num_modalities)
            ])
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        latents: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Fuse modality latents through cross-attention.
        
        Args:
            latents: List of latent tensors per modality
            
        Returns:
            List of fused latent tensors
        """
        # Ensure 3D tensors
        latents_3d = []
        for z in latents:
            if z.dim() == 2:
                latents_3d.append(z.unsqueeze(1))
            else:
                latents_3d.append(z)
        
        # Apply cross-attention layers
        for layer_idx in range(self.num_layers):
            new_latents = []
            
            for i, z in enumerate(latents_3d):
                # Cross-attend to all other modalities
                cross_attn_out = z
                for j, other_z in enumerate(latents_3d):
                    if i != j:
                        attn_out, _ = self.cross_attentions[layer_idx][i](
                            z, other_z, other_z
                        )
                        cross_attn_out = cross_attn_out + attn_out
                
                # Layer norm
                z_normed = self.layer_norms[layer_idx][i](cross_attn_out)
                
                # Feedforward
                ff_out = self.ff_layers[layer_idx][i](z_normed)
                z_out = self.ff_norms[layer_idx][i](z_normed + ff_out)
                
                new_latents.append(z_out)
            
            latents_3d = new_latents
        
        # Return 2D tensors
        return [z.squeeze(1) if z.shape[1] == 1 else z for z in latents_3d]


class MultimodalVAEGenerator(VAEGenerator):
    """
    VAE-based Multimodal Generator.
    
    Combines VAE framework with multimodal decoding.
    
    Architecture:
        Encoders per modality → Fusion → (μ, log σ²) → z
        z → Cross-Modal Fusion → Modality Decoders
        
    Example:
        >>> generator = MultimodalVAEGenerator(
        ...     encoders={"tabular": tab_enc, "text": text_enc, "image": img_enc},
        ...     decoders={"tabular": tab_dec, "text": text_dec, "image": img_dec},
        ...     latent_dim=512
        ... )
    """
    
    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        decoders: Dict[str, nn.Module],
        latent_dim: int = 512,
        beta: float = 1.0,
        fusion_type: str = "attention",  # attention, concat, mean
        name: str = "multimodal_vae_generator",
        **kwargs
    ):
        """
        Initialize multimodal VAE.
        
        Args:
            encoders: Dict of encoder modules per modality
            decoders: Dict of decoder modules per modality
            latent_dim: Latent dimension
            beta: KL weight
            fusion_type: How to fuse encoder outputs
            name: Generator name
        """
        # Create combined encoder
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.fusion_type = fusion_type
        self.modalities = list(encoders.keys())
        
        # Encoder fusion
        encoder_dim = latent_dim  # Assume encoders output latent_dim
        if fusion_type == "concat":
            encoder_dim = latent_dim * len(encoders)
        
        self.encoder_fusion = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2)  # mu and logvar
        )
        
        # Modality-specific heads
        self.modality_heads = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU()
            )
            for modality in self.modalities
        })
        
        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusion(
            latent_dim=latent_dim,
            num_modalities=len(self.modalities)
        )
        
        # Initialize with dummy encoder
        super().__init__(
            encoder=nn.Identity(),  # Handled by encode method
            decoder=nn.Identity(),  # Handled by decode method
            latent_dim=latent_dim,
            beta=beta,
            name=name,
            **kwargs
        )
        
    def encode(
        self,
        x: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode multimodal input.
        
        Args:
            x: Dictionary of inputs per modality
            
        Returns:
            Tuple of (mu, logvar)
        """
        # Encode each modality
        modality_encodings = []
        for modality in self.modalities:
            if modality in x and modality in self.encoders:
                enc = self.encoders[modality](x[modality], **kwargs)
                if isinstance(enc, tuple):
                    enc = enc[0]  # Take mean if VAE encoder
                modality_encodings.append(enc)
        
        # Fuse encodings
        if self.fusion_type == "concat":
            fused = torch.cat(modality_encodings, dim=-1)
        elif self.fusion_type == "mean":
            fused = torch.stack(modality_encodings, dim=0).mean(dim=0)
        else:  # attention
            fused = torch.stack(modality_encodings, dim=1)  # (batch, num_modalities, dim)
            # Simple attention pooling
            attn_weights = F.softmax(
                torch.sum(fused, dim=-1), dim=-1
            ).unsqueeze(-1)
            fused = (fused * attn_weights).sum(dim=1)
        
        # Get mu and logvar
        params = self.encoder_fusion(fused)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        logvar = torch.clamp(logvar, -10, 10)
        
        return mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to multimodal output.
        
        Args:
            z: Latent tensor
            
        Returns:
            Dictionary of decoded outputs
        """
        # Project to modality latents
        modality_latents = {}
        for modality in self.modalities:
            modality_latents[modality] = self.modality_heads[modality](z)
        
        # Cross-modal fusion
        latent_list = [modality_latents[m] for m in self.modalities]
        fused_latents = self.cross_modal_fusion(latent_list)
        
        # Decode
        outputs = {}
        for i, modality in enumerate(self.modalities):
            outputs[modality] = self.decoders[modality](fused_latents[i], **kwargs)
        
        return outputs
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate multimodal samples."""
        z = self.sample_latent(n_samples)
        return self.decode(z, **kwargs)
    
    def compute_loss(
        self,
        x: Dict[str, torch.Tensor],
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute VAE loss for multimodal data."""
        # Encode
        mu, logvar = self.encode(x, **kwargs)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, **kwargs)
        
        # Reconstruction losses
        recon_losses = {}
        for modality in self.modalities:
            if modality in x and modality in x_recon:
                if modality == "tabular":
                    recon_losses[modality] = self._compute_tabular_loss(
                        x[modality], x_recon[modality]
                    )
                elif modality == "text":
                    recon_losses[modality] = self._compute_text_loss(
                        x[modality], x_recon[modality]
                    )
                elif modality == "image":
                    recon_losses[modality] = self._compute_image_loss(
                        x[modality], x_recon[modality]
                    )
        
        total_recon = sum(recon_losses.values())
        
        # KL loss
        kl_loss = self._compute_kl_divergence(mu, logvar)
        
        # Total loss
        total_loss = total_recon + self.beta * kl_loss
        
        if return_components:
            return {
                "total_loss": total_loss,
                "reconstruction_loss": total_recon,
                "kl_loss": kl_loss,
                **{f"{k}_recon": v for k, v in recon_losses.items()}
            }
        
        return total_loss


class MultimodalDiffusionGenerator(DiffusionGenerator):
    """
    Diffusion-based Multimodal Generator.
    
    Performs diffusion in shared latent space for coherent multimodal generation.
    
    Example:
        >>> generator = MultimodalDiffusionGenerator(
        ...     denoiser=unet_model,
        ...     decoders={"tabular": tab_dec, "text": text_dec, "image": img_dec},
        ...     latent_dim=512,
        ...     num_timesteps=1000
        ... )
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        decoders: Dict[str, nn.Module],
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        name: str = "multimodal_diffusion_generator",
        **kwargs
    ):
        """
        Initialize multimodal diffusion generator.
        
        Args:
            denoiser: Latent denoiser
            decoders: Dict of decoders per modality
            latent_dim: Latent dimension
            num_timesteps: Number of diffusion steps
            name: Generator name
        """
        super().__init__(
            denoiser=denoiser,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            name=name,
            **kwargs
        )
        
        self.decoders = nn.ModuleDict(decoders)
        self.modalities = list(decoders.keys())
        
        # Modality projection heads
        self.modality_heads = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU()
            )
            for modality in self.modalities
        })
        
        # Cross-modal fusion
        self.fusion = CrossModalFusion(
            latent_dim=latent_dim,
            num_modalities=len(self.modalities)
        )
        
    def decode(
        self,
        z: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Decode latent to multimodal output."""
        # Project to modality latents
        modality_latents = {
            modality: self.modality_heads[modality](z)
            for modality in self.modalities
        }
        
        # Cross-modal fusion
        latent_list = [modality_latents[m] for m in self.modalities]
        fused_latents = self.fusion(latent_list)
        
        # Decode
        outputs = {}
        for i, modality in enumerate(self.modalities):
            outputs[modality] = self.decoders[modality](fused_latents[i], **kwargs)
        
        return outputs
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multimodal samples via diffusion.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            progress: Show progress bar
            
        Returns:
            Dictionary of generated samples
        """
        # Sample in latent space via diffusion
        shape = (n_samples, self._latent_dim)
        z = self.p_sample_loop(shape, progress)
        
        # Decode to multimodal outputs
        return self.decode(z)


class ConditionalMultimodalGenerator(ConditionalGenerator, MultimodalGenerator):
    """
    Conditional Multimodal Generator.
    
    Supports conditioning on sensitive attributes for fair multimodal generation.
    
    Example:
        >>> generator = ConditionalMultimodalGenerator(
        ...     latent_dim=512,
        ...     tabular_decoder=tab_dec,
        ...     text_decoder=text_dec,
        ...     image_decoder=img_dec,
        ...     num_classes=10
        ... )
        >>> samples = generator.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        encoder: Optional[nn.Module] = None,
        tabular_decoder: Optional[nn.Module] = None,
        text_decoder: Optional[nn.Module] = None,
        image_decoder: Optional[nn.Module] = None,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        name: str = "conditional_multimodal_generator",
        **kwargs
    ):
        """
        Initialize conditional multimodal generator.
        
        Args:
            latent_dim: Latent dimension
            encoder: Optional encoder
            tabular_decoder: Tabular decoder
            text_decoder: Text decoder
            image_decoder: Image decoder
            num_classes: Number of classes
            condition_dim: Condition dimension
            name: Generator name
        """
        ConditionalGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        MultimodalGenerator.__init__(
            self,
            latent_dim=latent_dim,
            encoder=encoder,
            tabular_decoder=tabular_decoder,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            name=name,
            **kwargs
        )
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate conditional multimodal samples.
        
        Args:
            n_samples: Number of samples
            conditions: Conditions dict
            class_label: Class label
            condition: Continuous condition
            modalities: Modalities to generate
            
        Returns:
            Dictionary of generated samples
        """
        if conditions is not None:
            class_label = conditions.get("class_label", class_label)
            condition = conditions.get("condition", condition)
        
        # Sample latent
        z = self.sample_latent(n_samples)
        
        # Apply conditioning
        z = self.combine_latent_condition(z, class_label, condition)
        
        # Project to modality latents
        modality_latents = {}
        for modality in self.modalities:
            modality_latents[modality] = self.modality_heads[modality](z)
        
        # Cross-modal fusion
        if self.use_cross_modal_attention and self.fusion_module is not None:
            latent_list = [modality_latents[m] for m in self.modalities if m in modality_latents]
            if len(latent_list) > 1:
                fused_latents = self.fusion_module(latent_list)
                for i, modality in enumerate([m for m in self.modalities if m in modality_latents]):
                    modality_latents[modality] = fused_latents[i]
        
        # Decode
        outputs = {}
        if modalities is None:
            modalities = self.modalities
            
        if "tabular" in modalities and self.tabular_decoder is not None:
            outputs["tabular"] = self.tabular_decoder(modality_latents["tabular"])
        if "text" in modalities and self.text_decoder is not None:
            outputs["text"] = self.text_decoder(modality_latents["text"])
        if "image" in modalities and self.image_decoder is not None:
            outputs["image"] = self.image_decoder(modality_latents["image"])
        
        return outputs
    
    def generate_counterfactual(
        self,
        z: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual multimodal samples.
        
        Args:
            z: Original latent
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            
        Returns:
            Counterfactual multimodal output
        """
        return self.generate(
            z.shape[0],
            condition=target_sensitive.float(),
            **kwargs
        )


class FairMultimodalGenerator(FairGenerator, ConditionalMultimodalGenerator):
    """
    Fair Multimodal Generator for bias-aware synthetic data generation.
    
    Integrates fairness constraints across all modalities.
    
    Example:
        >>> generator = FairMultimodalGenerator(
        ...     latent_dim=512,
        ...     tabular_decoder=tab_dec,
        ...     text_decoder=text_dec,
        ...     image_decoder=img_dec,
        ...     sensitive_attributes=["gender", "race"]
        ... )
        >>> samples = generator.generate_fair(100)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        encoder: Optional[nn.Module] = None,
        tabular_decoder: Optional[nn.Module] = None,
        text_decoder: Optional[nn.Module] = None,
        image_decoder: Optional[nn.Module] = None,
        sensitive_attributes: Optional[List[str]] = None,
        fairness_weights: Optional[Dict[str, float]] = None,
        name: str = "fair_multimodal_generator",
        **kwargs
    ):
        """
        Initialize fair multimodal generator.
        
        Args:
            latent_dim: Latent dimension
            encoder: Optional encoder
            tabular_decoder: Tabular decoder
            text_decoder: Text decoder
            image_decoder: Image decoder
            sensitive_attributes: Sensitive attribute names
            fairness_weights: Fairness weights
            name: Generator name
        """
        ConditionalMultimodalGenerator.__init__(
            self,
            latent_dim=latent_dim,
            encoder=encoder,
            tabular_decoder=tabular_decoder,
            text_decoder=text_decoder,
            image_decoder=image_decoder,
            name=name,
            **kwargs
        )
        FairGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            sensitive_attributes=sensitive_attributes,
            fairness_weights=fairness_weights
        )
    
    def generate_fair(
        self,
        n_samples: int,
        target_distribution: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate fair multimodal samples.
        
        Args:
            n_samples: Number of samples
            target_distribution: Target distribution for sensitive attrs
            
        Returns:
            Fair synthetic samples
        """
        # Sample from balanced latent distribution
        z = self.sample_latent(n_samples)
        
        # Apply fairness constraints
        if target_distribution is not None:
            # Adjust latent based on target distribution
            z = self._apply_fairness_constraints(z, target_distribution)
        
        return self.decode(z, **kwargs)
    
    def _apply_fairness_constraints(
        self,
        z: torch.Tensor,
        target_distribution: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply fairness constraints to latent."""
        # Placeholder for fairness constraint implementation
        return z
    
    def compute_fairness_penalty(
        self,
        samples: Dict[str, torch.Tensor],
        sensitive: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute fairness penalty across modalities.
        
        Args:
            samples: Generated samples per modality
            sensitive: Sensitive attribute values
            
        Returns:
            Fairness penalty
        """
        if self._adversary is None or sensitive is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        penalty = 0.0
        for modality, samples_modality in samples.items():
            # Adversary tries to predict sensitive from generated samples
            pred_sensitive = self._adversary(samples_modality)
            
            if pred_sensitive.dim() > sensitive.dim():
                penalty = penalty + F.cross_entropy(pred_sensitive, sensitive)
            else:
                penalty = penalty + F.mse_loss(pred_sensitive, sensitive.float())
        
        return -self.fairness_weights.get("adversarial", 0.1) * penalty
