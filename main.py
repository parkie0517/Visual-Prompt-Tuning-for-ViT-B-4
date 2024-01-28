"""
Step 1: Import Libraries
"""
import os
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR10
from tensorboardX import SummaryWriter
from CustomVPT import CustomPrompts, CustomViT # Custom class used for modifying the pre-trained ViT-B/16


def main():
    # Argument Parser = argument comprehender
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=3)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--root', type=str, default='/root/datasets/ViT_practice/cifar10') # This is where the dataset is downloaded
    parer.add_argument('--log_dir', type=str, default='./model') # define the path used for storing the saved models
    parer.add_argument('--name', type=str, default='vit_cifar10')
    parer.add_argument('--rank', type=int, default=0)
    ops = parer.parse_args()

    # CUDA setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'currently using {device}')

    """
    Step 2: Load CIFAR10 dataset
    """
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])

    test_transform_cifar = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    
    train_set = CIFAR10(root=ops.root, train=True, download=True, transform=transform_cifar)
    test_set = CIFAR10(root=ops.root, train=False, download=True, transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=ops.batch_size)
    test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=ops.batch_size)

    """
    3. Define the ViT Model
    """
    # Create the custom model
    model = CustomViT(pretrained_model = 'vit_base_patch16_224', 
                          img_size=32, 
                          patch_size=4, 
                          num_classes=10,
                          )
    model = model.to(device) # Upload the model to the specified device

    # Freeze the enocder part
    for name, param in model.named_parameters():
        if 'blocks' in name:
            param.requires_grad = False # Set the parameter untrainable
    
    # Uncomment the block below to cheeck if the freezing procedure is properly executed
    """
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
    """

    # Uncomment the block below to check if the model works well!
    """
    batch_size=10
    x = torch.randn(batch_size, 3, 32, 32)
    x = x.to(device)
    output = model(x)
    m = torch.nn.Softmax(dim=1)
    output = m(output)
    print(output.shape)
    print(output)
    """

    # uncomment the block below to print out the shapes of the model's parameters
    """
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    """


    """
    4. Training and Testing
    """

    # Set information about the training process
    criterion = nn.CrossEntropyLoss() # Softmax + cross entropy loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=ops.lr, weight_decay=5e-5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ops.lr, weight_decay=5e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)
    # os.makedirs(ops.log_dir, exist_ok=True)


    for epoch in range(1, ops.epoch+1):
        model.train() # Change to training mode
        train_total = 0
        train_correct = 0
        train_loss = 0.0

        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]

            optimizer.zero_grad() # Use this to clear the old gradients
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()


            train_loss += loss.item() * img.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_avg_loss = train_loss / train_total
        print(f"Epoch {epoch}/{ops.epoch} - Training Accuracy: {train_acc:.4f}%, Training Avg Loss: {train_avg_loss:.4f}")
        
        # TensorBoard logging for training metrics
        writer.add_scalar('Acc/train_acc', train_acc, epoch)
        writer.add_scalar('Loss/train_loss', train_avg_loss, epoch)

        """
        # Save the trained models
        save_path = os.path.join(ops.log_dir, ops.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(save_path, ops.name + '.{}.pth.tar'.format(epoch)))
        """

        # Test the model performance
        model.eval()
        correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():

            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]
                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                # first eval
                pred, idx_ = output.max(-1)
                correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()
        

        val_accuracy = 100*correct / total
        val_avg_loss = val_avg_loss / len(test_loader)
        print(f"Epoch {epoch}/{ops.epoch} - Validation Accuracy: {val_accuracy:.4f}%, Validation Avg Loss: {val_avg_loss:.4f}")
        

        scheduler.step()
        # Use tensorboard to record the validation acc and loss
        writer.add_scalar('Acc/val_acc', val_accuracy, epoch) # use add_scalar() function to write
        writer.add_scalar('Loss/val_loss', val_avg_loss, epoch)
    writer.flush()
    


if __name__ == '__main__':
    writer = SummaryWriter('./logs/test/first_test') # Write training results in './logs/' directory
    main()
    writer.close() # Must include this code when finish training results