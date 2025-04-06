import torch
import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):

    def __init__(self, channels=1):
        super(Discriminator, self).__init__()
        
        kernel_size = 5
        layer_filters = [32, 64, 128, 256]
        
        self.model = nn.Sequential()
        
        # Input layer
        in_channels = channels
        
        # Building the convolutional layers
        for i, filters in enumerate(layer_filters):
            strides = 1 if filters == layer_filters[-1] else 2
            
            # LeakyReLU followed by Conv2d
            self.model.add_module(f'leaky_relu_{i}', nn.LeakyReLU(0.2))
            self.model.add_module(
                f'conv_{i}',
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=strides,
                    padding=kernel_size // 2
                )
            )
            
            in_channels = filters
        
        # Adaptive pooling layer : to reduce the output to a fixed size
        self.model.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        # Flatten layer
        self.model.add_module('flatten', nn.Flatten())
        
        # Output layer
        self.model.add_module('dense', nn.Linear(256, 1))
        self.model.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        return self.model(x)

# Test du Discriminator
if __name__ == "__main__":
    discriminator = Discriminator()
    print(discriminator)

    # Summary du modèle
    summary(discriminator)

    # Test avec un batch de 8 images
    test_input = torch.randn(8, 1, 28, 28)
    output = discriminator(test_input)
    print(output.shape)