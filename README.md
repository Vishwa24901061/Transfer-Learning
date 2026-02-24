# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.

## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match one binary classes
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                labels = labels.float().unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: VISHWA V")
    print("Register Number:212224110062")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="661" height="225" alt="Screenshot 2026-02-24 200803" src="https://github.com/user-attachments/assets/c746b321-272c-4fad-a1b8-17bcb696523c" />
<img width="967" height="743" alt="Screenshot 2026-02-24 200750" src="https://github.com/user-attachments/assets/0b5bedee-87e0-479b-8072-04052993b969" />

### Classification Report

<img width="603" height="227" alt="Screenshot 2026-02-24 201144" src="https://github.com/user-attachments/assets/88915b94-8459-4a2e-b39a-e3a370d2bbcd" />

### Confusion Matrix
<img width="942" height="749" alt="Screenshot 2026-02-24 200904" src="https://github.com/user-attachments/assets/9bb5f8cf-8495-457f-86f7-f673edb709f7" />


### New Sample Prediction
<img width="537" height="494" alt="Screenshot 2026-02-24 200933" src="https://github.com/user-attachments/assets/adb4e4ee-f850-4c33-90b3-cf9e7cce68a5" />
<img width="551" height="514" alt="Screenshot 2026-02-24 200937" src="https://github.com/user-attachments/assets/da1c17ae-823f-443c-9bcb-d3672e06d8c4" />

## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
