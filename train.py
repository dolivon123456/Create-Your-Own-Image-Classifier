import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from collections import OrderedDict


def load_data(data_directory):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_datasets = datasets.ImageFolder(os.path.join(data_directory, 'train'), transform=data_transforms['train'])
    valid_datasets = datasets.ImageFolder(os.path.join(data_directory, 'valid'), transform=data_transforms['valid'])

    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_datasets, batch_size=64, shuffle=False)
    return train_loader, valid_loader, train_datasets


def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture [vgg16]')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units for custom classifier')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    train_loader, valid_loader, train_datasets = load_data(args.data_directory)

    optimizer = None

    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(num_features, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(args.hidden_units, len(train_datasets.classes))),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)

    no_improve = 0
    min_valid_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                valid_loss += criterion(output, labels).item()

                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch + 1}/{args.epochs}.. "
              f"Train loss: {running_loss / len(train_loader):.3f}.. "
              f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
              f"Validation accuracy: {accuracy / len(valid_loader):.3f}")

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == 2:
                print("Early stopping due to no improvement!")
                break

        scheduler.step()

  
    checkpoint = {
        'architecture': args.arch,
        'hidden_units': args.hidden_units,
        'output_size': len(train_datasets.classes),
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_datasets.class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'classifier': model.classifier
    }

    torch.save(checkpoint, f"{args.save_dir}/checkpoint_{args.arch}.pth")


if __name__ == '__main__':
    main()
