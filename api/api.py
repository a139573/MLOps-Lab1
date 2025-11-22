# Import the libraries, classes and functions
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

# from pydantic import BaseModel
from mylib.inference import predict_img_class, resize_image

# Create an instance of FastAPI
app = FastAPI(
    title="API of the Image Predictor using FastAPI",
    description="API to identify the image in a picture between cat, dog or fox",
    version="0.1.0",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


# Initial endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


# # Input class (with Pydantic) to define the input arguments of the calculator
# class CalcRequest(BaseModel):
#     operation: str
#     a: float
#     b: float


# Main endpoint to perform the artihmetical operations using the input class defined with Pydantic
@app.post("/predict")
async def predict(img: UploadFile = File('cat.jpg')):
    """
    It predicts the animal in the picture uploaded by the user
    """
    result = predict_img_class(img)
    return {"result": result}

@app.post("/resize")
async def resize(img: UploadFile = File('cat.jpg'), width: int = Form(), height: int = Form()):
    """
    It resizes the uploaded image to the width and height specified by the user
    """
    image_bytes = await img.read()
    result_bytes = resize_image(image_bytes, width, height)
    return Response(content=result_bytes, media_type="image/jpeg")


# Entry point (for direct execution only)
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)