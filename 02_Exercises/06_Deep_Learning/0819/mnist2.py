import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = Path(__file__).resolve().parent / "mnist.pth"
DATA_DIR = Path(__file__).resolve().parent / "data"

# --- 2. Model Definition ---
class ImageClassifier(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=500, num_class=10):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- 3. Data Loading ---
def get_mnist_dataloader(train=True, batch_size=64):
    """Loads the MNIST dataset and returns a DataLoader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=DATA_DIR, train=train, transform=transform, download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_mnist_test_dataset():
    """Loads the MNIST test dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)


# --- 4. Model Operations ---
def load_model(path):
    """Loads a pre-trained model from the given path."""
    model = ImageClassifier().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    return model

def predict(model, image_tensor):
    """Performs inference on a single image tensor."""
    with torch.no_grad():
        # The model expects a batch, so we add a dimension if it's not there.
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = F.softmax(output, dim=1)
        
        # Get the top prediction
        confidence, predicted_class = torch.max(probabilities, 1)
        
        return predicted_class.item(), confidence.item()

# --- 5. Training and Evaluation (for script execution) ---
def train_model_script(model, train_loader, epochs=10, learning_rate=0.001):
    """Trains the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train() # Set model to training mode

    print("Starting model training...")
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    print("Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def evaluate_model_script(model, test_loader):
    """Evaluates the model on the test set."""
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Set Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    # This block runs when the script is executed directly
    # e.g., `python mnist2.py`
    
    # 1. Prepare data
    train_loader = get_mnist_dataloader(train=True)
    test_loader = get_mnist_dataloader(train=False)

    # 2. Create model
    model_to_train = ImageClassifier().to(DEVICE)

    # 3. Train the model
    train_model_script(model_to_train, train_loader, epochs=10) # Reduced epochs for quick run

    # 4. Evaluate the model
    evaluate_model_script(model_to_train, test_loader)
