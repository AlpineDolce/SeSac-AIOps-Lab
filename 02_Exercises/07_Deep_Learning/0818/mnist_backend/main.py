from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import random
from pathlib import Path

# Import functions and classes from the refactored model.py
from model import load_model, predict

# --- 1. Configuration and Setup ---

# Use pathlib for robust path management
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
MODEL_PATH = BASE_DIR / "trained_model_weights.pth"
DATA_DIR = BASE_DIR / "data"

# Ensure static directory exists
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- 2. Load Model and Data at Startup ---

# Load the trained model using the helper function from model.py
# This is done once when the application starts.
try:
    model = load_model(MODEL_PATH)
    print("PyTorch 모델 로드 완료.")
except FileNotFoundError:
    print(f"에러: 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
    print("model.py를 직접 실행하여 모델을 먼저 학습하고 저장해주세요. (예: python model.py)")
    model = None

# Helper function to prepare the dataset
def get_mnist_test_dataset():
    """Loads the MNIST test dataset."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)

test_dataset = get_mnist_test_dataset() if model else None
if test_dataset:
    print("MNIST 테스트 데이터셋 준비 완료.")


# --- 3. Helper Functions ---

def prepare_image_for_response(image_tensor):
    """Converts a tensor to a base64 encoded string for HTML display."""
    from torchvision.transforms import ToPILImage
    # Denormalize for display: image = image * std + mean
    display_image = image_tensor * 0.5 + 0.5
    # Convert tensor to PIL Image
    pil_image = ToPILImage()(display_image)
    
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# --- 4. API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict_digit():
    """
    Selects a random image from the MNIST test set,
    predicts the digit, and returns the result.
    """
    if not model or not test_dataset:
        return {"error": "Model or data not loaded. Please check server logs."}

    # 1. Get a random image and its label from the test dataset
    idx = random.randint(0, len(test_dataset) - 1)
    image_tensor, true_label = test_dataset[idx]

    # 2. Perform inference using the predict function from model.py
    # The model expects a batch, so we add a dimension.
    predicted_digit = predict(model, image_tensor.unsqueeze(0))

    # 3. Prepare the image for display in the browser
    img_str = prepare_image_for_response(image_tensor)

    # 4. Return the results
    return {
        "image": img_str,
        "true_label": true_label,
        "predicted_digit": predicted_digit
    }

# To run this app:
# uvicorn main:app --reload

