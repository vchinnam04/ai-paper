import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import io
from nutrition_database import NutritionDatabase

# Fix model paths with absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(current_dir, "custom_cnn_final.keras")
VIT_MODEL_PATH = os.path.join(current_dir, "vit_best_food101.pth")

# Load PyTorch ViT model (using the same approach as in vit_test.py)
def load_vit_model(model_path):
    try:
        # Follow the exact pattern from vit_test.py
        num_classes = 101
        model = models.vit_b_16(pretrained=False)
        model.heads = nn.Linear(model.heads[0].in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ViT model: {e}")
        return None

# Load CNN model
def load_cnn_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading CNN model: {e}")
        return None

# Preprocess image for CNN
def preprocess_image_cnn(image, img_size=(224, 224)):
    img = image.resize(img_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Preprocess image for ViT (use the same transforms as in vit_test.py)
def preprocess_image_vit(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Get Food101 class names
def get_class_names():
    # Hardcoded list of Food101 classes
    return [
        'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
        'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
        'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
        'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
        'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
        'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
        'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
        'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
        'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
        'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
        'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
        'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
        'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
        'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
        'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
        'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
        'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
        'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
        'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
        'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
    ]

# Predict using CNN model
def predict_cnn(model, image, class_names, top_k=5):
    try:
        preprocessed_img = preprocess_image_cnn(image)
        predictions = model.predict(preprocessed_img)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [(class_names[i], float(predictions[i])) for i in top_indices]
    except Exception as e:
        st.error(f"Error predicting with CNN: {e}")
        return []

# Predict using ViT model
def predict_vit(model, image, class_names, top_k=5):
    try:
        preprocessed_img = preprocess_image_vit(image)
        with torch.no_grad():
            outputs = model(preprocessed_img)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)
        return [(class_names[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]
    except Exception as e:
        st.error(f"Error predicting with ViT: {e}")
        return []

# Function to display predictions and nutrition info
def display_predictions_and_nutrition(image, predictions, nutrition_db):
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Food Image', use_column_width=True)
    
    with col2:
        # Display predictions
        st.subheader("Predictions")
        for i, (food, prob) in enumerate(predictions):
            st.write(f"{i+1}. {food.replace('_', ' ').title()}: {prob:.2%}")
        
        # Get nutrition for the top prediction
        top_food = predictions[0][0].replace('_', ' ')
        st.subheader(f"Nutrition Information for {top_food.title()}")
        
        # Try to get nutrition data
        nutrition_data = nutrition_db.get_food_item(top_food)
        
        if nutrition_data:
            # Display nutrition data in a more visual way
            nutrition = nutrition_data['nutrition']
            
            # Create three columns for macros
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Calories", f"{nutrition['calories']} kcal")
            with c2:
                st.metric("Protein", f"{nutrition['protein']}g")
            with c3:
                st.metric("Carbs", f"{nutrition['carbs']}g")
            
            # Create more columns for other nutrition info
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Fat", f"{nutrition['fat']}g")
            with c2:
                if nutrition['fiber']:
                    st.metric("Fiber", f"{nutrition['fiber']}g")
            with c3:
                if nutrition['sugar']:
                    st.metric("Sugar", f"{nutrition['sugar']}g")
            
            # Vitamins and minerals if available
            if nutrition['vitamins'] and len(nutrition['vitamins']) > 0:
                st.subheader("Vitamins")
                for vitamin, amount in nutrition['vitamins'].items():
                    st.write(f"- {vitamin}: {amount}mg")
            
            if nutrition['minerals'] and len(nutrition['minerals']) > 0:
                st.subheader("Minerals")
                for mineral, amount in nutrition['minerals'].items():
                    st.write(f"- {mineral}: {amount}mg")
        else:
            st.info("Nutrition data not available for this food item.")
            st.write("This app includes nutrition data for a limited set of foods. "
                     "The Food101 dataset includes many foods not yet in our nutrition database.")

def main():
    st.set_page_config(page_title="Food Classification & Nutrition", layout="wide")
    
    st.title("Food Classification and Nutrition Information")
    
    # Sidebar for model selection
    st.sidebar.title("Model Options")
    model_option = st.sidebar.radio(
        "Select Model",
        ["Custom CNN (Keras)", "Vision Transformer (PyTorch)"]
    )
    
    # Initialize nutrition database
    nutrition_db = NutritionDatabase()
    
    # Load class names
    class_names = get_class_names()
    
    # Mode selection (Demo or Upload)
    mode = st.radio("Select Mode", ["Upload Your Image", "Try Demo Images"])
    
    # Demo mode
    if mode == "Try Demo Images":
        st.subheader("Demo Mode: Select a sample image")
        
        # Get available demo images
        demo_images = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                demo_images.append(file)
                
        # If we have test_images folder with example images
        if os.path.exists('test_images'):
            for file in os.listdir('test_images'):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    demo_images.append(os.path.join('test_images', file))
        
        # If no demo images found, use these specific images if they exist
        if not demo_images:
            sample_images = ['pizza.jpg', 'fries.jpeg', 'taco.jpeg']
            existing_samples = [img for img in sample_images if os.path.exists(img)]
            if existing_samples:
                demo_images = existing_samples
        
        if demo_images:
            selected_demo = st.selectbox("Select a sample image:", demo_images)
            
            if st.button("Classify Demo Image"):
                with st.spinner('Processing...'):
                    # Load and process the selected demo image
                    try:
                        image = Image.open(selected_demo).convert('RGB')
                        
                        # Run prediction based on selected model
                        if model_option == "Custom CNN (Keras)":
                            with st.status("Loading CNN model..."):
                                model = load_cnn_model(CNN_MODEL_PATH)
                            
                            if model:
                                with st.status("Making predictions..."):
                                    predictions = predict_cnn(model, image, class_names)
                        else:  # ViT
                            with st.status("Loading ViT model..."):
                                model = load_vit_model(VIT_MODEL_PATH)
                            
                            if model:
                                with st.status("Making predictions..."):
                                    predictions = predict_vit(model, image, class_names)
                        
                        # Display results if predictions were made
                        if model and predictions:
                            display_predictions_and_nutrition(image, predictions, nutrition_db)
                        else:
                            st.error("Failed to make predictions. Please check model files.")
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
        else:
            st.warning("No demo images found. Please upload your own image.")
            mode = "Upload Your Image"
    
    # Upload mode
    if mode == "Upload Your Image":
        st.subheader("Upload an image of food to get nutritional information")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file).convert('RGB')
                
                # Model prediction
                with st.spinner('Classifying...'):
                    if model_option == "Custom CNN (Keras)":
                        with st.status("Loading CNN model..."):
                            model = load_cnn_model(CNN_MODEL_PATH)
                        
                        if model:
                            with st.status("Making predictions..."):
                                predictions = predict_cnn(model, image, class_names)
                    else:  # ViT
                        with st.status("Loading ViT model..."):
                            model = load_vit_model(VIT_MODEL_PATH)
                        
                        if model:
                            with st.status("Making predictions..."):
                                predictions = predict_vit(model, image, class_names)
                    
                    # Display results if predictions were made
                    if model and predictions:
                        display_predictions_and_nutrition(image, predictions, nutrition_db)
                    else:
                        st.error("Failed to make predictions. Please check model files.")
            except Exception as e:
                st.error(f"Error processing image: {e}")

    # App information and instructions
    with st.expander("About this App"):
        st.write("""
        This app uses deep learning models trained on the Food101 dataset to classify food images.
        After classification, it displays nutritional information for the detected food.
        
        - The **Custom CNN** model is a Keras-based model that uses a custom architecture with multiple convolutional layers.
        - The **Vision Transformer (ViT)** model uses a transformer-based architecture for image classification.
        
        *Note: The nutrition database contains data for a limited set of foods. Not all foods in Food101 have nutritional data available.*
        """)

if __name__ == "__main__":
    main() 