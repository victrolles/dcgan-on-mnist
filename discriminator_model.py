import torch.nn as nn
from torchinfo import summary

class Discriminator(nn.Module):
    def __init__(self, image_channels=1):
        super(Discriminator, self).__init__()
        self.layer_filters = [32, 64, 128, 256]
        self.kernel_size = 5

        # Boucle de convolution
        self.convNet = nn.Sequential()
        in_channels = image_channels

        for i, out_channels in enumerate(self.layer_filters):
            
            stride = 2 if out_channels < 200 else 1
            print(stride)
            self.convNet.add_module(f'relu_{i}', nn.LeakyReLU(0.2, True))
            self.convNet.add_module(
                f'convT_{i}', 
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding='same'
                )
            )
            in_channels = out_channels

        # Couches finales
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        # Boucle de convolution
        x = self.convNet(x)
        # Couches finales
        x = self.net(x)
        return x
    
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