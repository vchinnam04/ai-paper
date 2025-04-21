import os
import argparse
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from nutrition_database import NutritionDatabase
import sys

#CNN
def load_cnn_model(model_path):
    """Load the Keras CNN model."""
    if not os.path.exists(model_path):
        print(f"Error: CNN model not found at {model_path}")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return None

#VIT
def load_vit_model(model_path):
    """Load the PyTorch ViT model."""
    if not os.path.exists(model_path):
        print(f"Error: ViT model not found at {model_path}")
        return None
    try:
        num_classes = 101
        model = models.vit_b_16(weights=None)
        model.heads = nn.Linear(model.heads[0].in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None

def preprocess_image_cnn(image_path):
    """Preprocess image for CNN."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0), img
    except Exception as e:
        print(f"Error preprocessing image for CNN: {e}")
        return None, None

def preprocess_image_vit(image_path):
    """Preprocess image for ViT."""
    try:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0), img
    except Exception as e:
        print(f"Error preprocessing image for ViT: {e}")
        return None, None

def get_class_names():
    """Get Food101 class names."""
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

def predict_cnn(model, image, class_names, top_k=5):
    """Predict using CNN model."""
    try:
        predictions = model.predict(image)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        return [(class_names[i], float(predictions[i])) for i in top_indices]
    except Exception as e:
        print(f"Error predicting with CNN: {e}")
        return []

def predict_vit(model, image, class_names, top_k=5):
    """Predict using ViT model."""
    try:
        with torch.no_grad():
            outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)
        return [(class_names[idx.item()], prob.item()) for prob, idx in zip(top_probs, top_indices)]
    except Exception as e:
        print(f"Error predicting with ViT: {e}")
        return []

def display_nutrition_info(food_name, nutrition_db):
    """Display nutrition information for a food item."""
    nutrition_data = nutrition_db.get_food_item(food_name.replace('_', ' '))
    
    if nutrition_data:
        nutrition = nutrition_data['nutrition']
        print(f"\nNutrition Information for {food_name.replace('_', ' ').title()}:")
        print(f"Calories: {nutrition['calories']} kcal")
        print(f"Protein: {nutrition['protein']}g")
        print(f"Carbs: {nutrition['carbs']}g")
        print(f"Fat: {nutrition['fat']}g")
        
        if nutrition['fiber']:
            print(f"Fiber: {nutrition['fiber']}g")
        if nutrition['sugar']:
            print(f"Sugar: {nutrition['sugar']}g")
        if nutrition['sodium']:
            print(f"Sodium: {nutrition['sodium']}mg")
        
        # Vitamins
        if nutrition['vitamins'] and len(nutrition['vitamins']) > 0:
            print("\nVitamins:")
            for vitamin, amount in nutrition['vitamins'].items():
                print(f"- {vitamin}: {amount}mg")
        
        # Minerals
        if nutrition['minerals'] and len(nutrition['minerals']) > 0:
            print("\nMinerals:")
            for mineral, amount in nutrition['minerals'].items():
                print(f"- {mineral}: {amount}mg")
    else:
        print(f"\nNutrition data not available for {food_name.replace('_', ' ')}.")

def main():
    parser = argparse.ArgumentParser(description='Classify food images and display nutrition info')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, choices=['cnn', 'vit'], default='cnn', 
                        help='Model to use (cnn or vit)')
    parser.add_argument('--cnn_model', type=str, default='custom_cnn_final.keras',
                        help='Path to CNN model file')
    parser.add_argument('--vit_model', type=str, default='vit_best_food101.pth',
                        help='Path to ViT model file')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file {args.image} not found.")
        return
    
    # Get class names
    class_names = get_class_names()
    
    # Initialize nutrition database
    nutrition_db = NutritionDatabase()
    
    print(f"Classifying image {args.image} using {args.model.upper()} model...")
    
    if args.model == 'cnn':
        model = load_cnn_model(args.cnn_model)
        if model is None:
            return
        
        processed_img, original_img = preprocess_image_cnn(args.image)
        if processed_img is None:
            return
        
        predictions = predict_cnn(model, processed_img, class_names)
    else:  # vit
        model = load_vit_model(args.vit_model)
        if model is None:
            return
        
        processed_img, original_img = preprocess_image_vit(args.image)
        if processed_img is None:
            return
        
        predictions = predict_vit(model, processed_img, class_names)
    
    # Display results
    if predictions:
        print(f"\nTop 5 predictions:")
        for i, (food, prob) in enumerate(predictions):
            print(f"{i+1}. {food.replace('_', ' ').title()}: {prob:.2%}")
        
        # Display nutrition for the top prediction
        top_food = predictions[0][0]
        display_nutrition_info(top_food, nutrition_db)
    else:
        print("Failed to make predictions.")

if __name__ == "__main__":
    main() 