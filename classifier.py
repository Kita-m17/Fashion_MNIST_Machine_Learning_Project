from torchvision import datasets
from matplotlib.pyplot import imshow
import random
import matplotlib
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn #neural network
import torch.nn.functional as F #activation functions
from torch.utils.data import random_split, DataLoader, TensorDataset
import copy
import sys
from contextlib import redirect_stdout


def load_data_fashion_mnist(download_dataset=True):
    """Download the Fashion-MNIST dataset and load it into memory."""
    if download_dataset:  
        DATA_DIR = "download" 
    else:
        DATA_DIR = "."
    train_mnist = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset)
    test_mnist = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset)
    
    return train_mnist, test_mnist

def reload_data_fashion_mnist(train_mnist, test_mnist):
    """Reload the Fashion-MNIST dataset from the specified directory."""
    # Create variables for MNIST data
    X_train = train_mnist.data.float()
    y_train = train_mnist.targets
    X_test = test_mnist.data.float()
    y_test = test_mnist.targets

    return X_train, y_train, X_test, y_test

def split_training_validation(X_test, X_train, y_train):
    """Split the training data into training and validation sets. let validation size = test_set size """

    # Sample random indices for validation
    test_size = X_test.shape[0]
    indices = np.random.choice(X_train.shape[0], test_size, replace=False)

    # Create validation set
    X_valid = X_train[indices]
    y_valid = y_train[indices]

    # Remove validation set from training set
    X_train = np.delete(X_train, indices, axis=0)
    y_train = np.delete(y_train, indices, axis=0)

    return X_train, y_train, X_valid, y_valid

def load_grayscale_image(jpg_path):
    """Convert the image to grayscale."""
    # Load a grayscale image
    img = torchvision.io.read_image(jpg_path, mode=torchvision.io.ImageReadMode.GRAY)
    img = img.squeeze() # Convert from (1,28,28) tensor to (28,28) tensorreturn img_tensor
    return img

def sigmoid(r):
    return 1 / (1 + torch.exp(-r))

def relu(x):
    return torch.max(torch.tensor(0.0), x)

def softmax(x):
    """Compute the softmax of a tensor."""
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

def accuracy(outputs, labels):
    """Compute the accuracy of the model."""
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct


def train_model(model, device, train_loader, criterion, optimizer, epoch, learning_rate):
    """Train the model using the training data."""

    #Set the model to training mode
    model.train()

    #initialise the running metrics
    run_loss = 0.0
    run_accuracy = 0.0
    total_samples = 0

    #loop through the batches in the training set
    for batch_index, (inputs, labels) in enumerate(train_loader):
        #moves data to (CPU or GPU)
        inputs, labels = inputs.to(device), labels.to(device)
        
        #computes thd loss between the predictions and true values
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward() #calc gradients
        optimizer.step() #updates parameters
        
        #calculate the loss
        run_loss += loss.item()

        # Compute correct predictions accuracies
        run_accuracy += accuracy(outputs, labels)
        total_samples += labels.size(0) #increment sample

        # Print the loss and accuracy every 200 batches
        #batch_index - counts batches from 0, 1, 2, 3...
        #print every 200 batches
        if (batch_index+1) % 200 == 0:
            avg_loss = run_loss/(batch_index+1)
            test_accuracy = 100. * run_accuracy / total_samples

            print(f"Epoch [{epoch+1}], Batch {batch_index+1}, Avg Loss: {avg_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
            
    #compute the final metric data for entire epoch
    avg_loss = run_loss/len(train_loader)
    test_accuracy = 100. * run_accuracy /total_samples

    print(f"Training set: Average loss: {avg_loss:.4f}, Accuracy: ({test_accuracy:.2f}%)")

def test_model(model, device, test_loader, criterion):
    """Test the model using the test data."""
    #sets the model to evaluation mode - disables dropout
    model.eval()

    #initialise metrics
    test_loss = 0
    test_accuracy = 0
    total = 0

    #disable the gradients
    with torch.no_grad():
        #go through test dataset
        for inputs, labels in test_loader:
            #move data to specofoed device
            inputs, labels = inputs.to(device), labels.to(device)
            
            #forward passes the model
            outputs = model(inputs)

            #computest the losss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            #computes the accuracy
            test_accuracy += accuracy(outputs, labels)
            total += labels.size(0) #increment the sample count
    
    #calculate the final metrics
    test_loss = test_loss/len(test_loader)
    t_accuracy = 100. * test_accuracy / total

    #print resultss
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: ({t_accuracy:.2f}%)")
    return test_loss, t_accuracy

def train_with_early_stopping(model, device, learning_rate, val_loader, train_loader, criterion , optimizer,  max_epochs = 50, patience = 5, val_interval=2):
    #initialise the tracking variables for eaarly stopping 
    best_val_accuracy = None
    best_model_weights = None
    best_epoch = 0
    no_improvement = 0

    #list to store the validation metrics history
    validation_acc = []
    validation_loss = []

    #training loop
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        #train the model 1 epock at a time
        train_model(model, device, train_loader, criterion, optimizer, epoch, learning_rate)

        # Only run validation every val_interval epochs
        if (epoch + 1) % val_interval == 0:
            #run validation
            val_loss, val_accuracy = test_model(model, device, val_loader, criterion)
            validation_acc.append(val_accuracy)
            validation_loss.append(val_loss)

            #check for improvement
            if best_val_accuracy is None or val_accuracy > best_val_accuracy + 0.001:
                #if imporvement found - update the model
                best_val_accuracy = val_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                no_improvement = 0
                print(f"New best model found at {epoch+1}, val_acc: {val_accuracy:.2f}%")

            else:
                #if no impovement found
                no_improvement += val_interval
                print(f"No improvement for {no_improvement} epoch(s)")

            #check for early stopping
            if no_improvement>= patience:
                print(f"\nEarly stopping: No improvement after {patience} epochs")
                break

    #loads the best model weights if the validation was performed
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch+1}")
    else:
        print("No validation run occurred. Consider reducing val_interval.")

    return validation_acc, best_val_accuracy, validation_loss, best_epoch, no_improvement

def get_class_name(val, preds, labels, class_names):
    """Gets the label of the predicted value label - returns both the truse and predicted value"""
    true_label = class_names[labels[val]]
    predicted_label = class_names[preds[val]]

    return true_label, predicted_label

def display_image(device, model, test_loader, class_names):
    """Display the image and the predicted class."""
    model.eval()
    # Visualize sample images with predictions
    samples, labels = next(iter(test_loader))
    samples, labels = samples.to(device), labels.to(device) #get first batch

    # forward pass - get model predictions
    outputs = model(samples)
    _, preds = torch.max(outputs, 1)

    #move data tp CPU
    samples = samples.cpu().view(-1, 28, 28)  # Reshape to (batch_size, height, width)
    labels = labels.cpu()
    preds = preds.cpu()
    
    #ready to plot: create a 3x3 grid
    _, axes = plt.subplots(3, 3, figsize=(10, 10))

    #loop through each subplot
    for i, ax in enumerate(axes.ravel()):
        #get predicted and true vlass labels
        true_label, predicted_label = get_class_name(i, preds, labels, class_names)
        #display image
        ax.imshow(samples[i], cmap='gray')
        ax.set_title(f'Label: {true_label}, Prediction: {predicted_label}')
        ax.axis('off') #remove axis ticks
    #configure image
    plt.tight_layout()
    plt.savefig(f'results.png')
    plt.show()

def predict_Image(model, image_path, device, class_names):
    """Load a JPEG image, preprocess it, and predict its class using the trained model"""
    img =  Image.open(image_path).convert('L') #load grayscale image
    img = img.resize((28, 28)) #convert from (1, 28, 28) to (28, 28) tensor
    
    #convert pil to tensor
    img_tensor = torchvision.transforms.functional.to_tensor(img)
    #add batch dim to tensor
    img_tensor = img_tensor.unsqueeze(0)

    #ensure tensor is a float
    img_tensor = img_tensor.float() 

    #move tensor to device
    img_tensor = img_tensor.to(device)

    #save image
    img.save("processed_preview.png")
    model.eval()

    #disable gradient calculation
    with torch.no_grad():

        #forward pass - get model prediction
        output = model(img_tensor)
        #get prediction
        _, prediction = torch.max(output, 1)

    #get prediction label
    return class_names[prediction.item()]

def plot_accuracy_loss(validation_acc, best_epoch, val_interval, best_acc, val_loss):
    #ploot the validation accuracy
    plt.figure(figsize=(8, 4))
    plt.title('Validation accuracy. Dot denotes best accuracy.')
    plt.plot(validation_acc, label='Validation accuracy')
    plt.plot(best_epoch // val_interval,  best_acc, 'bo', label='Best accuracy')
    plt.grid(True)
    plt.savefig("Validation_Accuracy.png")
    plt.show()

    # Plot validation loss
    plt.figure(figsize=(8, 4))
    plt.title('Validation Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Validation Interval')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("Validation_loss.png")
    plt.show()

class FNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNNet, self).__init__()
        self.input_layer = nn.Linear(input_size, 512) #input layer
        self.bn1 = nn.BatchNorm1d(512)

        self.hidden_layer1 = nn.Linear(512, 256) #hidden layer 1
        self.bn2 = nn.BatchNorm1d(256)

        self.hidden_layer2 = nn.Linear(256,hidden_size) #hidden layer 2
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.hidden_layer3 = nn.Linear(hidden_size,hidden_size) #hidden layer 3
        self.bn4 = nn.BatchNorm1d(hidden_size)

        self.output_layer = nn.Linear(hidden_size, output_size) #output layer
        self.dropout = nn.Dropout(p=0.5) #dropout layer

    def forward(self, x):
        x = x.view(-1, 28*28) #flatten the image
        x = F.relu(self.bn1(self.input_layer(x))) #ReLU activation - first layer
        x = self.dropout(x) #apply dropout after activation

        x = F.tanh(self.bn2(self.hidden_layer1(x))) #Tanh activation - second layer
        x = self.dropout(x) #apply dropout after activation

        x = F.tanh(self.bn3(self.hidden_layer2(x))) #Tanh activation - second layer
        x = self.dropout(x) #apply dropout after activation

        x = F.tanh(self.bn4(self.hidden_layer3(x))) #Tanh activation - second layer
        x = self.dropout(x) #apply dropout after activation

        x = self.output_layer(x) #output layer - final layer
        x = F.log_softmax(x, dim=1) #classification
        return x
    
def main():
    # define the hyperparameters
    num_classes = 10
    # num_features = 784
    patience = 3
    num_epochs = 20
    epochs_not_impove = 0
    learning_rate = 0.0001
    hidden_size = 128
    input_size = 28 * 28
    batch_size = 64
    val_interval = 1
    torch.manual_seed(42)
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print("Pytorch Output...")
    #define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Fashion-MNIST dataset
    train_mnist, test_mnist = load_data_fashion_mnist()
    X_train, y_train, X_test, y_test = reload_data_fashion_mnist(train_mnist, test_mnist)


    # Normalise input data
    X_train = X_train/255.0
    X_test = X_test/255.0

    # Flatten X_train and X_test
    X_train = X_train.view(X_train.size(0), -1)
    X_test = X_test.view(X_test.size(0), -1)

    # Split the training data into training and validation sets
    # validation_size = 10000
    # train_size = 50000
    X_train, y_train, X_valid, y_valid  = split_training_validation(X_test, X_train, y_train)

    test_data =  TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    train_data =  TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    validation_data =  TensorDataset(X_valid, y_valid)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    num_features = X_train.shape[1]  # Store number of features for later use (784)

    model = FNNet(input_size=num_features, hidden_size=hidden_size, output_size=num_classes)
    model = model.to(device)  # Move the model to the device (GPU or CPU)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    # Train the model - with early stopping and validate it
    with open ('log.txt', 'w') as f:
        with redirect_stdout(f):
            validation_acc, best_acc, val_loss, best_epoch, no_improvement = train_with_early_stopping(model=model, device=device, learning_rate=learning_rate, val_loader=validation_loader, train_loader=train_loader, criterion=criterion , optimizer=optimizer,  max_epochs = 50, patience = 5, val_interval=val_interval)

    # Test the  final model using the test data
    test_model(model, device, test_loader, criterion)

    # Test the model using the test data - display some validation images and predictions
    display_image(device, model, test_loader, class_names)

    #plot the validation accuracies and loss
    plot_accuracy_loss(validation_acc=validation_acc, best_epoch=best_epoch, val_interval=val_interval, best_acc=best_acc, val_loss=val_loss)
    
    #done training the model
    print("Done!")

    #test the model on external jpegd - see if it classifies well
    filepath = input("Please enter a filepath:")
    while(filepath.lower() != "exit"):
        try:
            predicted = predict_Image(model, filepath, device, class_names)
            print(f'Classifier: {predicted}')
        except Exception as e:
            print(f"Error: {e}")
        filepath = input("Please enter a filepath:")
    

if __name__ == "__main__":
    main()