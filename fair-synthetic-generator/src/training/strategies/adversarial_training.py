"""
Adversarial Training Strategy for Fairness
============================================

Implements adversarial training to learn fair representations by
training an adversary to predict sensitive attributes while the main model tries to prevent this.

Training Objective:
    Generator: minimize task_loss + λ * adversary_success
    Adversary: maximize sensitive_attribute_prediction accuracy

Supported Modes:
    - gradient_reversal: GRL reverses gradients automatically
    - alternating: Alternating GAN-style mini-max optimization
    - pretrain: Pretrain adversary before main training

Example:
    >>> strategy = AdversarialTrainingStrategy(
        ...     model=generator,
        ...     adversary=adversary_network
        ...     config=AdversarialConfig(mode="gradient_reversal")
        ... )
        >>> 
        >>> for epoch in range(num_epochs):
        ...     metrics = strategy.train_epoch(dataloader, epoch)
        >>> 
        >>> metrics = strategy._compute_adversary_metrics(dataloader, epoch)
        >>> 
        >>> # Adversary metrics (when applicable)
        >>> if self.mode != AdversarialMode.ALTERNATING:
        ...     adv_metrics = strategy._compute_adversary_metrics(dataloader, epoch)
        ...     if adv_metrics:
        ...         metrics["adversary_metrics"] = adv_metrics
        
        >>> metrics = strategy._train_generator(dataloader, epoch)
        ...     self._train_adversary(dataloader, epoch)
        
        >>> # Generator metrics
        >>> gen_metrics = strategy._train_generator(dataloader, epoch)
        ...     metrics.update(gen_metrics)
        
        >>> # Combine metrics
        >>> return {**metrics, **adv_metrics}
    
    def _train_generator_alternating(
        self,
        batch: Dict[str, torch.Tensor],
        sensitive: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train generator for one step with alternating updates.
        
        Args:
            batch: Input batch
            sensitive: Sensitive attribute tensor
            
        Returns:
            Dictionary of losses
        """
        self.generator.train()
        self.adversary.eval()
        
        with torch.no_grad():
            latent = outputs.get("latent", outputs.get("z")
            
            # Generator step (for alternating)
            outputs = self.generator(batch)
            if isinstance(outputs, dict):
                gen_loss = outputs.get("loss", outputs["losses"].get("total", 0)
            else:
                gen_loss = outputs
            
            gen_loss.backward()
            
            self._step_count += 1
            gen_optimizer.step()
            
        adv_logits = self.adversary(latent.detach())
        adv_loss = F.cross_entropy(adv_logits, sensitive)
        
        accuracy = (adv_logits.argmax(dim=-1) == sensitive).float().mean().item()
            total_adv_loss += adv_loss.item()
            total_accuracy += accuracy.item()
        
        self._step_count += 1
        
        return {
            "generator_loss": gen_loss.item(),
            "adversary_loss": adv_loss.item(),
            "generator_accuracy": accuracy.item(),
            "adversary_accuracy": accuracy.item(),
        }
    
    def step_epoch(self) -> None:
        """Advance epoch counter and update GRL if scheduled."""
        self.grl.step()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        return {
            "current_epoch": self.current_epoch,
            "best_adversary_loss": min(self.history.get("best_adversary_loss", float("inf")),
            "best_generator_loss": min(self.history.get("best_generator_loss", float("inf")),
            "history": self.history
        }
