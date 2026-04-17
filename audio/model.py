#!/usr/bin/env python3
"""
CRNN model for siren detection
Architecture: Conv blocks -> BiGRU -> Dense -> Sigmoid
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU + MaxPool"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class SirenCRNN(nn.Module):
    """
    CRNN for siren detection
    Input: (batch, 1, n_mels, time_frames)
    Output: (batch, 1) sigmoid probability
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        conv_channels: list = [32, 64, 128],
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_mels = n_mels
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for out_ch in conv_channels:
            self.conv_blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        
        # Calculate feature dimension after conv layers
        # Each conv block reduces spatial dims by 2x (pool_size=2)
        self.n_pools = len(conv_channels)
        self.freq_dim = n_mels // (2 ** self.n_pools)
        self.rnn_input_size = conv_channels[-1] * self.freq_dim
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, 1)  # *2 for bidirectional
    
    def forward(self, x):
        """
        x: (batch, 1, n_mels, time_frames)
        returns: (batch, 1) probability
        """
        batch_size = x.size(0)
        
        # Conv blocks
        for conv in self.conv_blocks:
            x = conv(x)
        
        # x: (batch, channels, freq, time)
        # Reshape for RNN: (batch, time, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels*freq)
        
        # BiGRU
        x, _ = self.gru(x)  # (batch, time, hidden*2)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, hidden*2)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)  # (batch, 1)
        x = torch.sigmoid(x)
        
        return x
    
    def predict_proba(self, x):
        """Convenience method for inference"""
        with torch.no_grad():
            return self.forward(x)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model forward pass"""
    print("=" * 60)
    print("CRNN Model Test")
    print("=" * 60)
    
    # Create model
    model = SirenCRNN(
        n_mels=128,
        conv_channels=[32, 64, 128],
        rnn_hidden=128,
        rnn_layers=2
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    n_params = count_parameters(model)
    print(f"\nTrainable parameters: {n_params:,}")
    
    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_frames = 100  # ~1 second at 10ms hop
    
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    print(f"\nInput shape: {x.shape}")
    
    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    # Test with different sequence lengths
    for frames in [50, 100, 200]:
        x = torch.randn(2, 1, n_mels, frames)
        y = model(x)
        print(f"  {frames} frames -> output: {y.shape}")
    
    print("\n[OK] Model test passed!")


if __name__ == "__main__":
    test_model()