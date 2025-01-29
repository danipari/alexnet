from tqdm import tqdm
from datetime import datetime
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = torch.nn.Conv2d(96, 256, 5, padding=2)
        self.conv3 = torch.nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(384, 256, 3, padding=1)

        # Fully connected layers
        self.fc1 = torch.nn.Linear(9216, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, num_classes)

        # Initializatin of the weights
        torch.nn.init.normal_(self.conv1.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv2.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv3.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv4.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.conv5.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.fc3.weight, mean=0.0, std=0.01)

        # Initializatin of the bias
        torch.nn.init.constant_(self.conv2.bias, 1.0)
        torch.nn.init.constant_(self.conv4.bias, 1.0)
        torch.nn.init.constant_(self.conv5.bias, 1.0)
        torch.nn.init.constant_(self.conv1.bias, 0.0)
        torch.nn.init.constant_(self.conv3.bias, 0.0)
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.constant_(self.fc2.bias, 0.0)
        torch.nn.init.constant_(self.fc3.bias, 0.0)

        # The definition of alpha in the paper and pytorch are slightly different
        self.lrn = torch.nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2.0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool -> LRN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool(x)
        
        # Conv2 -> ReLU -> Pool -> LRN
        x = self.conv2(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool(x)

        # Conv3 -> ReLU
        x = self.conv3(x)
        x = self.relu(x)
        
        # Conv4 -> ReLU
        x = self.conv4(x)
        x = self.relu(x)
        
        # Conv5 -> ReLU -> Pool
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC1 -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # FC2 -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # FC3
        x = self.fc3(x)
        return x

# transpose is use to meet the shape of the tensor C x H x W rather than H x W x C
mean_pixels = np.float32(np.load('mean_pixels.npy').transpose(2, 0, 1) / 255.0)

def meanSubstraction(x):
    return x - mean_pixels

def PCAColorAugmentation(img):
    # img is assumed to be the tensor already normalized between 0-1
    img_shape = img.shape
    # Flatten each RGB channel
    img = img.numpy().reshape(3, -1)
    # Get covariance matrix, no need to center since correlation isn't affected by centering
    img_cov = np.cov(img)
    # Get eigen values and vectors 
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    # Generate alpha
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, 0.1, 3)
    # Compute corrections
    correction = eig_vecs @ (alpha * eig_vals).T
    # Apply correction
    new_img = np.float32(img + correction.reshape(3, 1)).reshape(img_shape)
    return torch.from_numpy(new_img)


def train_one_epoch(epoch_index, writer, model, training_dataloader, optimizer, loss_fn, device):
    """
    Trains AlexNet for one epoch using mixed precision training.
    
    Args:
        epoch_index (int): Current epoch number
        writer: TensorBoard writer object
        model: AlexNet model instance
        training_dataloader: PyTorch dataloader for training data
        optimizer: SGD optimizer instance
        loss_fn: CrossEntropyLoss instance
        device: Device to train on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, average_accuracy) for the epoch
    """
    total_loss = 0.0
    running_loss = 0.0
    total_top1_error = 0.0
    total_top5_error = 0.0
    running_top1_error = 0.0
    running_top5_error = 0.0
    
    # Ensure model is in training mode
    model.train()
    
    for i, data in enumerate(tqdm(training_dataloader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate top-1 and top-5 error rates
        _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
        correct_top1 = top5_preds[:, 0] == labels
        correct_top5 = top5_preds.eq(labels.view(-1, 1)).sum(dim=1) > 0
        top1_error = 1 - correct_top1.sum().item() / labels.size(0)
        top5_error = 1 - correct_top5.sum().item() / labels.size(0)
        
        # Update running statistics
        running_loss += loss.item()
        running_top1_error += top1_error
        running_top5_error += top5_error

        # Update total statistics
        total_loss += loss.item() 
        total_top1_error += top1_error
        total_top5_error += top5_error
        
        # Log every 100 batches
        if i % 1000 == 999:
            avg_running_loss = running_loss / 1000
            avg_top1_error = 100.0 * running_top1_error / 1000
            avg_top5_error = 100.0 * running_top5_error / 1000
                        
            # Log to TensorBoard
            tb_x = epoch_index * len(training_dataloader) + i + 1
            writer.add_scalar('Loss/train_step', avg_running_loss, tb_x)
            writer.add_scalar('Top-1 error/train_step', avg_top1_error, tb_x)
            writer.add_scalar('Top-5 error/train_step', avg_top5_error, tb_x)

            print(f'  Batch {tb_x} Loss: {avg_running_loss:.4f} Top-1 error rate: {avg_top1_error:.2f}% Top-5 error rate: {avg_top5_error:.2f}%')
            
            running_loss = 0.0
            running_top1_error = 0.0
            running_top5_error = 0.0
    
    # Calculate epoch-level metrics
    avg_epoch_loss = total_loss / len(training_dataloader)
    avg_top1_error = 100.0 * total_top1_error / len(training_dataloader)
    avg_top5_error = 100.0 * total_top5_error / len(training_dataloader)
    
    return avg_epoch_loss, avg_top1_error, avg_top5_error


def validate_one_epoch(epoch_index, writer, model, validation_dataloader, loss_fn, device):
    """
    Validates AlexNet for one epoch, tracking both top-1 and top-5 error rates.
    
    Args:
        epoch_index (int): Current epoch number
        writer: TensorBoard writer object
        model: AlexNet model instance
        validation_dataloader: PyTorch dataloader for validation data
        loss_fn: CrossEntropyLoss instance
        device: Device to train on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, average_top1_error, average_top5_error) for the epoch
    """
    total_loss = 0.0
    total_top1_error = 0.0
    total_top5_error = 0.0
    running_loss = 0.0
    running_top1_error = 0.0
    running_top5_error = 0.0
    
    # Ensure model is in evaluation mode
    model.eval()
    
    with torch.no_grad():  # Disable gradient computation
        for i, data in enumerate(tqdm(validation_dataloader, desc='Validation')):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Calculate top-1 and top-5 error rates
            _, top5_preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            correct_top1 = top5_preds[:, 0] == labels
            correct_top5 = top5_preds.eq(labels.view(-1, 1)).sum(dim=1) > 0
            top1_error = 1 - correct_top1.sum().item() / labels.size(0)
            top5_error = 1 - correct_top5.sum().item() / labels.size(0)
            
            # Update running statistics
            running_loss += loss.item()
            running_top1_error += top1_error
            running_top5_error += top5_error

            # Update total statistics
            total_loss += loss.item()
            total_top1_error += top1_error
            total_top5_error += top5_error
            
            # Log every 100 batches
            if i % 100 == 99:
                avg_running_loss = running_loss / 100
                avg_top1_error = 100.0 * running_top1_error / 100
                avg_top5_error = 100.0 * running_top5_error / 100
                
                print(f'  Validation Batch {i + 1:5d} Loss: {avg_running_loss:.4f} Top-1 error: {avg_top1_error:.2f}% Top-5 error: {avg_top5_error:.2f}%')
                
                # Log to TensorBoard
                tb_x = epoch_index * len(validation_dataloader) + i + 1
                writer.add_scalar('Loss/val_step', avg_running_loss, tb_x)
                writer.add_scalar('Top-1 error/val_step', avg_top1_error, tb_x)
                writer.add_scalar('Top-5 error/val_step', avg_top5_error, tb_x)
                
                # Reset running statistics
                running_loss = 0.0
                running_top1_error = 0.0
                running_top5_error = 0.0
    
    # Calculate epoch-level metrics
    avg_epoch_loss = total_loss / len(validation_dataloader)
    avg_top1_error = 100.0 * total_top1_error / len(validation_dataloader)
    avg_top5_error = 100.0 * total_top5_error / len(validation_dataloader)
    
    return avg_epoch_loss, avg_top1_error, avg_top5_error



if __name__ == '__main__':
    os.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    alexnet = AlexNet(1000)
    # First load
    # pretrained_dict = torch.load('alexnette/model_20.pth', weights_only=True)
    # alexnet_dict = alexnet.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['fc3.weight', 'fc3.bias']}
    # alexnet_dict.update(pretrained_dict)
    # alexnet.load_state_dict(alexnet_dict)
    # Intermediate loads
    alexnet.load_state_dict(torch.load('model_55.pth', weights_only=True))
    alexnet.to(device)

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        # Transform to tensor and normalize between 0-1
        transforms.ToTensor(),
        # Apply PCA color transformation
        transforms.Lambda(PCAColorAugmentation),
        # Remove mean
        transforms.Lambda(meanSubstraction),
        # 227x227 random crop 
        transforms.RandomCrop(227),
        # Horizontal reflection with p=0.5
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    filepath = 'C://imagenet'
    print('Loading data...')
    training_data = torchvision.datasets.ImageNet(filepath, split='train', transform=transform)
    validation_data = torchvision.datasets.ImageNet(filepath, split='val', transform=transform)

    # training_data = torchvision.datasets.Imagenette('.', 'train', transform=transform)
    # validation_data = torchvision.datasets.Imagenette('.', 'val', transform=transform)

    training_dataloader = torch.utils.data.DataLoader(
        training_data, 
        batch_size=128, 
        shuffle=True, 
        num_workers=4, 
        prefetch_factor=4, 
        persistent_workers=True,
        pin_memory=True
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_data, 
        batch_size=128, 
        shuffle=False,  # not needed
        num_workers=4, 
        prefetch_factor=4, 
        persistent_workers=True,
        pin_memory=True
    )
    print('Finished loading')

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/alexnet') # .format(timestamp)
    
    examples = iter(training_dataloader)
    example_data, example_label = next(examples)
    img_grid = torchvision.utils.make_grid(example_data)
    # Add sample of images in a batch
    writer.add_image("imagenet_sample", img_grid)

    # Add the graph of the model
    writer.add_graph(alexnet, example_data.to(device))

    loss_fn = torch.nn.CrossEntropyLoss()
    # refer to https://github.com/dansuh17/alexnet-pytorch/blob/d0c1b1c52296ffcbecfbf5b17e1d1685b4ca6744/model.py#L142
    # and https://discuss.pytorch.org/t/image-recognition-alexnet-training-loss-is-not-decreasing/66885/9 
    # to justify the change
    optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.001/5, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params=alexnet.parameters(), lr=1e-6, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                            mode='min',  # Minimizing the metric (e.g., validation loss)
                                                            factor=0.2,  # Multiplicative factor of LR reduction
                                                            patience=5, # Number of epochs to wait before reducing LR
                                                            verbose=True)  # 

    EPOCHS = 120
    epoch_number = 57
    best_vloss = 1000000.0
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        # Training phase
        avg_train_loss, avg_train_top1_error, avg_train_top5_error = train_one_epoch(
            epoch_number,
            writer,
            alexnet,
            training_dataloader,
            optimizer,
            loss_fn,
            device,
        )

        # Validation phase
        avg_val_loss, avg_val_top1_error, avg_val_top5_error = validate_one_epoch(
            epoch_number,
            writer,
            alexnet,
            validation_dataloader,
            loss_fn,
            device
        )

        print(f'LOSS train {avg_train_loss:.4f} valid {avg_val_loss:.4f}')
        print(f'Top-1 Error train {avg_train_top1_error:.2f}% val {avg_val_top1_error:.2f}%')
        print(f'Top-5 Error train {avg_train_top5_error:.2f}% val {avg_val_top5_error:.2f}%')

        # Update scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Log epoch-level metrics
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training': avg_train_loss, 'Validation': avg_val_loss },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Top-1 Error',
                        { 'Training': avg_train_top1_error, 'Validation': avg_val_top1_error },
                        epoch_number + 1)
        writer.add_scalars('Training vs. Validation Top-5 Error',
                        { 'Training': avg_train_top5_error, 'Validation': avg_val_top5_error },
                        epoch_number + 1)
        writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch_number + 1)
        writer.flush()
    
        # Track best performance and save model
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_path = f'model_{epoch_number + 1}.pth'  # Save with dynamic name including the epoch number
            torch.save(alexnet.state_dict(), model_path)  # Save model

        epoch_number += 1