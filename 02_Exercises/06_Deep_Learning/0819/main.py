from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import random
from pathlib import Path
from torchvision import transforms

# Import functions and classes from the refactored mnist2.py
import mnist2 as model_handler

# --- 1. Configuration and Setup ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
# Use the model path defined in the model script for consistency
MODEL_PATH = model_handler.MODEL_SAVE_PATH

# Ensure static directory exists
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- 2. Load Model and Data at Startup ---
model = None
test_dataset = None

# Check if the model file exists before trying to load
if MODEL_PATH.exists():
    try:
        model = model_handler.load_model(MODEL_PATH)
        test_dataset = model_handler.get_mnist_test_dataset()
        print("PyTorch model and test dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading model or data: {e}")
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please train and save the model first by running: python mnist2.py")

# --- 3. Helper Functions ---
def prepare_image_for_response(image_tensor):
    """Converts a tensor to a base64 encoded string for HTML display."""
    from torchvision.transforms import ToPILImage
    # Denormalize for display: image = image * std + mean
    # The model uses Normalize((0.5,), (0.5,)), so we reverse it.
    display_image = image_tensor * 0.5 + 0.5
    pil_image = ToPILImage()(display_image)
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_uploaded_image(image_bytes):
    """Converts uploaded image bytes to a tensor suitable for the model."""
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("L") # Convert to grayscale
    
    # Define the same transformations as used for the training data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    return image_tensor

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
    predicted_digit, confidence = model_handler.predict(model, image_tensor)

    # 3. Prepare the image for display in the browser
    img_str = prepare_image_for_response(image_tensor)

    # 4. Return the results
    return {
        "image": img_str,
        "true_label": true_label,
        "predicted_digit": predicted_digit,
        "confidence": f"{confidence * 100:.2f}"
    }

@app.post("/upload")
async def upload_and_predict(file: UploadFile = File(...)):
    """
    Receives an uploaded image, predicts the digit, and returns the result.
    """
    if not model:
        return {"error": "Model not loaded. Please check server logs."}

    # 1. Read and process the uploaded image
    contents = await file.read()
    try:
        image_tensor = process_uploaded_image(contents)
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # 2. Perform inference
    predicted_digit, confidence = model_handler.predict(model, image_tensor)

    # 3. Prepare the uploaded image for display
    img_str = base64.b64encode(contents).decode()

    # 4. Return the results
    return {
        "image": img_str,
        "predicted_digit": predicted_digit,
        "confidence": f"{confidence * 100:.2f}"
    }

# To run this app:
# uvicorn main:app --reload