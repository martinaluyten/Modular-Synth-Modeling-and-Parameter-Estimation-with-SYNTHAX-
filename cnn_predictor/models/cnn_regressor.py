import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        if self.dropout:
            x = self.dropout(x)
        return x


class CNNRegressor(nn.Module):
    """
    CNN for regression to predict synthesizer parameters from mel-spectrograms.
    
    Architecture:
    - Input: (B, 1, H, W) grayscale mel-spectrogram
    - Conv blocks with increasing channels
    - Global average pooling
    - Fully connected regressor head
    - Output: (B, num_params) predicted raw parameter values
    """
    
    def __init__(self, num_params=11, dropout=0.5, hidden_dim=256):
        """
        Args:
            num_params: Number of parameters to predict (default 11)
            dropout: Dropout rate for FC layers
            hidden_dim: Hidden dimension for FC layers
        """
        super(CNNRegressor, self).__init__()
        
        self.num_params = num_params
        
        # Feature extractor - 4 convolutional blocks
        # Input: (B, 1, 128, 256)
        self.features = nn.Sequential(
            # Block 1: (B, 1, 128, 256) -> (B, 32, 64, 128)
            ConvBlock(1, 32, kernel_size=3, stride=1, padding=1, dropout=0.1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1, dropout=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: (B, 32, 64, 128) -> (B, 64, 32, 64)
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, dropout=0.2),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: (B, 64, 32, 64) -> (B, 128, 16, 32)
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, dropout=0.2),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, dropout=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: (B, 128, 16, 32) -> (B, 256, 8, 16)
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, dropout=0.3),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1, dropout=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global average pooling reduces to (B, 256, 1, 1) -> (B, 256)
        # Then flatten
        
        # Regressor head
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 8, hidden_dim),
            nn.BatchNorm1d(hidden_dim),       # Stabilizes the 8,192 inputs
            nn.LeakyReLU(0.1, inplace=True),  # Prevents dead neurons
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # Stabilizes the hidden layer
            nn.LeakyReLU(0.1, inplace=True),  # Prevents dead neurons
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_params)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 1, H, W) - grayscale mel-spectrogram
        
        Returns:
            predictions: (B, num_params) - predicted raw parameter values
        """
        # Feature extraction
        x = self.features(x)
        
        # Regression head
        predictions = self.regressor(x)
        
        return predictions
    
    def get_num_params(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_params=11, dropout=0.5, hidden_dim=256, device='cuda'):
    """
    Create and return CNNRegressor model.
    
    Args:
        num_params: Number of output parameters
        dropout: Dropout rate
        hidden_dim: Hidden dimension for FC layers
        device: Device to place model on
    
    Returns:
        model: CNNRegressor instance on specified device
    """
    model = CNNRegressor(num_params=num_params, dropout=dropout, hidden_dim=hidden_dim)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    elif device == 'mps' and torch.backends.mps.is_available():
        model = model.to('mps')
    else:
        model = model.cpu()
    
    print(f"Model created with {model.get_num_params():,} parameters")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    return model
