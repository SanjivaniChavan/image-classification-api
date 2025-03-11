# Image Classification API

This project classifies Fashion MNIST images using a pre-trained MobileNetV2 model.

## Setup Instructions

1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
4. Access the Swagger UI at `http://localhost:8000/docs` for API testing.

## Docker Instructions

1. Build the Docker image:
   ```bash
   docker build -t image-classification-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 image-classification-api
   ```

## Sample API Response

```json
{
    "predicted_class": 3
}
```