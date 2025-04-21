import json
import os
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

class NutritionDatabase:
    def __init__(self, db_path: str = "nutrition_data.json"):
        """Initialize the nutrition database.
        
        Args:
            db_path (str): Path to the JSON file storing nutrition data
        """
        self.db_path = db_path
        self.data = self._load_database()
        
    def _load_database(self) -> Dict:
        """Load the database from file or create a new one if it doesn't exist."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {
            "food_items": {},
            "categories": [],
            "last_updated": None
        }
    
    def _save_database(self):
        """Save the current database to file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def add_food_item(self, 
                      name: str, 
                      category: str,
                      calories: float,
                      protein: float,
                      carbs: float,
                      fat: float,
                      fiber: Optional[float] = None,
                      sugar: Optional[float] = None,
                      sodium: Optional[float] = None,
                      vitamins: Optional[Dict[str, float]] = None,
                      minerals: Optional[Dict[str, float]] = None):
        """Add a new food item to the database.
        
        Args:
            name (str): Name of the food item
            category (str): Food category (e.g., 'fruits', 'vegetables', 'meat')
            calories (float): Calories per 100g
            protein (float): Protein content in grams per 100g
            carbs (float): Carbohydrate content in grams per 100g
            fat (float): Fat content in grams per 100g
            fiber (float, optional): Fiber content in grams per 100g
            sugar (float, optional): Sugar content in grams per 100g
            sodium (float, optional): Sodium content in mg per 100g
            vitamins (Dict[str, float], optional): Vitamin content
            minerals (Dict[str, float], optional): Mineral content
        """
        food_item = {
            "name": name.lower(),
            "category": category.lower(),
            "nutrition": {
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat,
                "fiber": fiber,
                "sugar": sugar,
                "sodium": sodium,
                "vitamins": vitamins or {},
                "minerals": minerals or {}
            }
        }
        
        self.data["food_items"][name.lower()] = food_item
        if category.lower() not in self.data["categories"]:
            self.data["categories"].append(category.lower())
        self.data["last_updated"] = pd.Timestamp.now().isoformat()
        self._save_database()
    
    def get_food_item(self, name: str) -> Optional[Dict]:
        """Retrieve a food item by name.
        
        Args:
            name (str): Name of the food item to retrieve
            
        Returns:
            Optional[Dict]: Food item data if found, None otherwise
        """
        return self.data["food_items"].get(name.lower())
    
    def get_category_items(self, category: str) -> List[Dict]:
        """Get all food items in a specific category.
        
        Args:
            category (str): Category to filter by
            
        Returns:
            List[Dict]: List of food items in the category
        """
        category = category.lower()
        return [
            item for item in self.data["food_items"].values()
            if item["category"] == category
        ]
    
    def search_food_items(self, query: str) -> List[Dict]:
        """Search for food items by name.
        
        Args:
            query (str): Search query
            
        Returns:
            List[Dict]: List of matching food items
        """
        query = query.lower()
        return [
            item for item in self.data["food_items"].values()
            if query in item["name"]
        ]
    
    def get_nutrition_summary(self, food_items: List[str]) -> Dict:
        """Calculate total nutrition for a list of food items.
        
        Args:
            food_items (List[str]): List of food item names
            
        Returns:
            Dict: Summary of total nutrition values
        """
        summary = {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "fiber": 0,
            "sugar": 0,
            "sodium": 0
        }
        
        for item_name in food_items:
            item = self.get_food_item(item_name)
            if item:
                for key in summary:
                    if key in item["nutrition"] and item["nutrition"][key] is not None:
                        summary[key] += item["nutrition"][key]
        
        return summary
    
    def export_to_csv(self, output_path: str):
        """Export the database to a CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        df = pd.DataFrame.from_dict(self.data["food_items"], orient='index')
        df.to_csv(output_path)
    
    def import_from_csv(self, input_path: str):
        """Import food items from a CSV file.
        
        Args:
            input_path (str): Path to the CSV file
        """
        df = pd.read_csv(input_path, index_col=0)
        for _, row in df.iterrows():
            self.add_food_item(
                name=row['name'],
                category=row['category'],
                calories=row['nutrition']['calories'],
                protein=row['nutrition']['protein'],
                carbs=row['nutrition']['carbs'],
                fat=row['nutrition']['fat'],
                fiber=row['nutrition'].get('fiber'),
                sugar=row['nutrition'].get('sugar'),
                sodium=row['nutrition'].get('sodium'),
                vitamins=row['nutrition'].get('vitamins', {}),
                minerals=row['nutrition'].get('minerals', {})
            )

if __name__ == "__main__":
    # Create a new database
    db = NutritionDatabase()
    
    # Add some example food items
    db.add_food_item(
        name="Apple",
        category="fruits",
        calories=52,
        protein=0.3,
        carbs=14,
        fat=0.2,
        fiber=2.4,
        sugar=10.4,
        sodium=1,
        vitamins={"C": 4.6, "B6": 0.041},
        minerals={"Potassium": 107}
    )
    
    db.add_food_item(
        name="Chicken Breast",
        category="meat",
        calories=165,
        protein=31,
        carbs=0,
        fat=3.6,
        sodium=74,
        vitamins={"B6": 0.6, "B12": 0.3},
        minerals={"Potassium": 256, "Phosphorus": 210}
    )
    
    # Test retrieval
    apple = db.get_food_item("apple")
    print("Apple nutrition info:", apple)
    
    # Test category search
    fruits = db.get_category_items("fruits")
    print("Fruits in database:", fruits)
    
    # Test nutrition summary
    summary = db.get_nutrition_summary(["apple", "chicken breast"])
    print("Nutrition summary:", summary)
