import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    """Build a Discriminator Model 
    Stack of LeakyReLU-Conv2d to discriminate real from fake. 
    The network does not use BatchNorm as mentioned in the original code.
    
    Arguments: 
        image_size (tuple): Size of input image (height, width)
        channels (int): Number of input channels
    """
    def __init__(self, image_size=(28, 28), channels=1):
        super(Discriminator, self).__init__()
        
        kernel_size = 5
        layer_filters = [32, 64, 128, 256]
        
        self.model = nn.Sequential()
        
        # Input layer
        in_channels = channels
        
        # Building the convolutional layers
        for i, filters in enumerate(layer_filters):
            # first 3 convolution layers use strides = 2
            # last one uses strides = 1
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
                    padding=kernel_size // 2  # same padding
                )
            )
            
            in_channels = filters
        
        # Calculate output size after convolutions
        # This depends on the input size and stride/padding configuration
        # For simplicity, we'll use adaptive pooling to handle any input size
        self.model.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        
        # Flatten layer
        self.model.add_module('flatten', nn.Flatten())
        
        # Output layer
        self.model.add_module('dense', nn.Linear(256, 1))
        self.model.add_module('sigmoid', nn.Sigmoid())
        
    def forward(self, x):
        """Forward pass
        
        Arguments:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, 1)
        """
        return self.model(x)
    
if __name__ == "__main__":
    # Test du Discriminator
    import torch
    discriminator = Discriminator()
    print(discriminator)

    # Summary du modèle
    summary(discriminator)

    # Test avec un batch de 8 images de taille 64x64 avec 1 canal (grayscale)
    test_input = torch.randn(8, 1, 28, 28)
    output = discriminator(test_input)
    print(output.shape)  # Devrait être [8, 1]