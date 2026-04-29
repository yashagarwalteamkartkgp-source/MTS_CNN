import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import os
from typing import Tuple, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model checkpoint dynamically.
    """
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0, path: str = 'checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """Saves model when validation loss decreases."""
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ModelTrainer:
    """
    Scalable training pipeline for deep learning models.
    Supports config-driven setup, SGD+Momentum, mini-batches, and GPU acceleration.
    """
    def __init__(self, model: nn.Module, config_path: str):
        self.model = model
        
        # 1. GPU Support Allocation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Initialized ModelTrainer. Targeting compute device: {self.device}")
        
        # 2. Config-driven setup
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.config = full_config.get('trainer', full_config.get('training', {}))
            
        # Ensure checkpoint directory exists
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'outputs/models')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 3. Setup Criterion & Optimizer
        class_weights = torch.tensor([1.0, 3.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.01), 
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # 4. Early Stopping Tracker
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 10), 
            verbose=True, 
            path=os.path.join(self.checkpoint_dir, 'best_model.pt')
        )
        
        # 5. History tracking for plotting
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def _compute_gradient_norm(self) -> float:
        """Computes the L2 norm of the gradients."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
        
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move to GPU/CPU
            data, target = data.to(self.device), target.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            diagnostic_weights = None
            if isinstance(output, tuple):
                logits = output[0]
                diagnostic_weights = output[1]
            else:
                logits = output
                
            loss = self.criterion(logits, target)
            
            # Backpropagation
            loss.backward()
            
            # Compute Gradient Norm
            grad_norm = self._compute_gradient_norm()
            
            self.optimizer.step()
            
            # --- DEBUG HOOKS (Triggered on first batch of epoch) ---
            if batch_idx == 0:
                logger.info(f"--- DEBUG HOOKS [Epoch {epoch} | Batch 0] ---")
                logger.info(f"Input Batch Shape: {data.shape}")
                logger.info(f"Model Output Shape: {logits.shape}")
                logger.info(f"Loss Value: {loss.item():.6f}")
                logger.info(f"Gradient L2 Norm: {grad_norm:.6f}")
                if diagnostic_weights is not None:
                    mean_weights = diagnostic_weights.mean(dim=0).detach().cpu().numpy()
                    logger.info(f"Mean Diagnostic Weights (Distribution): {mean_weights}")
                logger.info("---------------------------------------------")
            
            # Metrics Tracking
            running_loss += loss.item() * data.size(0)
            
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % self.config.get('log_freq', 10) == 0:
                logger.info(f'Train Batch: [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.4f}')
                
        epoch_loss = running_loss / len(train_loader.dataset)
        accuracy = 100. * correct / total
        return epoch_loss, accuracy

    def validate(self, val_loader) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # No gradients needed for validation (saves memory/compute)
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                if isinstance(output, tuple):
                    output = output[0]
                    
                loss = self.criterion(output, target)
                val_loss += loss.item() * data.size(0)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        accuracy = 100. * correct / total
        return val_loss, accuracy
        
    def train(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Full Training Loop."""
        epochs = self.config.get('epochs', 100)
        logger.info(f"--- Starting training loop for {epochs} epochs ---")
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n[Epoch {epoch}/{epochs}]")
            
            # Training and Validation phase
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Logging Epoch Metrics
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Check Early Stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping threshold reached! Halting training.")
                break
                
        logger.info(f"Training finalized. Best model successfully saved to: {self.early_stopping.path}")
        
        # Load best model weights before returning
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        
        return self.history
