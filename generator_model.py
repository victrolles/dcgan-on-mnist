import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, image_size):
        super(Generator, self).__init__()
        self.image_resize = image_size // 4
        self.layer_filters = [128, 64, 32, 1]
        self.kernel_size = 5

        # Couche dense pour transformer le vecteur latent en un tenseur 3D
        self.fc = nn.Linear(z_dim, self.image_resize * self.image_resize * self.layer_filters[0])
        
        # Boucle de convolution
        self.net = nn.Sequential()
        in_channels = self.layer_filters[0]

        for i, out_channels in enumerate(self.layer_filters):
            stride = 2 if out_channels > self.layer_filters[-2] else 1
            self.net.add_module(f'bn_{i}', nn.BatchNorm2d(in_channels))
            self.net.add_module(f'relu_{i}', nn.ReLU(True))
            self.net.add_module(
                f'convT_{i}', 
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=self.kernel_size // 2,
                    output_padding=1 if stride == 2 else 0
                )
            )
            in_channels = out_channels

        # Activation finale
        self.out_activation = nn.Sigmoid()

    def forward(self, z):
        # Couche dense
        x = self.fc(z)
        # Reshape en tenseur 3D
        x = x.view(x.size(0), self.layer_filters[0], self.image_resize, self.image_resize)
        # Boucle de convolution
        x = self.net(x)
        # Activation finale
        x = self.out_activation(x)
        return x
    
if __name__ == "__main__":
    # Test du Generator
    import torch
    z_dim = 100
    image_size = 64
    generator = Generator(z_dim, image_size)
    print(generator)

    # Test avec un vecteur latent de taille 100 et batch de 8
    test_input = torch.randn(8, z_dim)
    output = generator(test_input)
    print(output.shape)  # Devrait être [8, 1, 64, 64]