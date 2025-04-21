# Food Classification and Nutrition Information App

This application allows you to upload food images, classify them using AI models, and get nutritional information for the identified food.

## Features

- Upload images of food
- Select between two different AI models for classification:
  - Custom CNN (Keras)
  - Vision Transformer (PyTorch)
- View top 5 predictions for the uploaded image
- Get detailed nutrition information for the detected food

## Required Models

Make sure you have the following model files in the project directory:
- `custom_cnn_final.keras` - Keras CNN model
- `vit_best_food101.pth` - PyTorch Vision Transformer model

## Setup and Installation

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```
   cd aifinal
   streamlit run app.py
   ```

3. Access the application in your web browser at http://localhost:8501

## Usage

1. Select your model from the sidebar.
2. Upload a food image using the file uploader.
3. The application will display:
   - The uploaded image
   - Top 5 predictions with confidence scores
   - Nutritional information for the top prediction (if available in the database)

## Notes

- The nutrition database contains a limited set of foods. Not all Food101 classes have corresponding nutrition data.
- For best results, use clear images of individual food items. 