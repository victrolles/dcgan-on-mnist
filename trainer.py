import random
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from discriminator_model import Discriminator
from generator_model import Generator

class Trainer():

    def __init__(self):
        # Initialize random seeds for reproducibility
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        # Hyperparameters
        self.latent_size = 100
        self.batch_size = 16
        self.train_steps = 40000
        self.lr = 2e-4
        self.decay = 6e-8
        self.image_size = 28

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to [0, 1]
        ])
        mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.data_loader = DataLoader(mnist_data, batch_size=self.batch_size, shuffle=True)

    def initialize_models(self):
        self.discriminator = Discriminator(image_channels=1).to(self.device)
        self.generator = Generator(z_dim=self.latent_size, image_size=self.image_size).to(self.device)

    def initialize_optimizers(self):
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.lr, weight_decay=self.decay
        )
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=self.lr * 0.5, weight_decay=self.decay * 0.5
        )

    def initialize_loss(self):
        self.loss = torch.nn.BCELoss()

    def train_generator(self):
        # Set the generator in training mode and the discriminator in evaluation mode
        self.generator.train()
        self.discriminator.eval()

        # Generate random noise
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        # Generate fake images
        fake_images = self.generator(z)
        # Create labels for fake images
        fake_labels = torch.ones(self.batch_size, 1).to(self.device)
        # Compute loss
        g_loss = self.loss(self.discriminator(fake_images), fake_labels)
        # Backpropagation
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss

    def train_discriminator(self, real_images):
        # Set the discriminator in training mode
        self.discriminator.train()
        self.generator.eval()

        # Generate random noise
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        # Generate fake images
        fake_images = self.generator(z).detach()
        # Create labels for real and fake images
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        fake_labels = torch.zeros(self.batch_size, 1).to(self.device)

        # Compute loss for real images
        d_loss_real = self.loss(self.discriminator(real_images), real_labels)
        # Compute loss for fake images
        d_loss_fake = self.loss(self.discriminator(fake_images), fake_labels)
        # Total loss
        d_loss = d_loss_real + d_loss_fake

        # Backpropagation
        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss
    
    def test(self, step):
        # Créer un dossier "images" s'il n'existe pas
        os.makedirs("images", exist_ok=True)

        # Générer les images
        z = torch.randn(64, self.latent_size).to(self.device)
        fake_images = self.generator(z).cpu().detach()
        fake_images = fake_images.view(-1, 1, self.image_size, self.image_size)
        fake_images = fake_images.numpy()
        fake_images = (fake_images * 255).astype(np.uint8)

        # Créer une figure et sauvegarder l'image
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        for i in range(64):
            ax = axes[i // 8, i % 8]
            ax.imshow(fake_images[i][0], cmap='gray')
            ax.axis('off')

        # Sauvegarder l'image dans le dossier "images"
        plt.savefig(f"images/generated_images_{step}.png")
        plt.close(fig)

    def loop(self):
        for step in range(self.train_steps):
            # Load a batch of real images
            real_images, _ = next(iter(self.data_loader))
            real_images = real_images.to(self.device)

            # Train the discriminator
            d_loss = self.train_discriminator(real_images)

            # Train the generator
            g_loss = self.train_generator()

            # Print losses every 1000 steps
            if step % 100 == 0:
                self.test(step)
                print(f"Step [{step}/{self.train_steps}]")
                print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load_dataset()
    trainer.initialize_models()
    trainer.initialize_optimizers()
    trainer.initialize_loss()
    trainer.loop()