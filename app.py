import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid
import os
import time
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import base64
from io import BytesIO
import json
import re

# MongoDB imports with URL parsing
import pymongo
from bson import ObjectId
from urllib.parse import urlparse, urlunparse, quote_plus

load_dotenv()

app = Flask(__name__)
CORS(app)

# Production Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-super-secret-jwt-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

jwt = JWTManager(app)

# MongoDB Configuration with URL encoding fix
def create_mongodb_client():
    """Create MongoDB client with properly encoded credentials"""
    try:
        raw_uri = os.getenv('MONGODB_URI')
        if not raw_uri:
            raise Exception("MONGODB_URI environment variable not set")
        
        parsed = urlparse(raw_uri)
        username = quote_plus(parsed.username) if parsed.username else ""
        password = quote_plus(parsed.password) if parsed.password else ""
        
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{username}:{password}@{host}{port}"
        
        encoded_uri = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
        
        print(f"🔗 Connecting to MongoDB...")
        client = pymongo.MongoClient(encoded_uri)
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        return client
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

# Create MongoDB client
client = create_mongodb_client()
db = client['foodguard']

# Collections
users_collection = db['users']
allergies_collection = db['allergies'] 
scan_history_collection = db['scan_history']

# Create indexes for better performance
try:
    users_collection.create_index("email", unique=True)
    allergies_collection.create_index("user_id")
    scan_history_collection.create_index("user_id")
    print("✅ MongoDB indexes created")
except Exception as e:
    print(f"⚠️ Index creation warning: {e}")

# Global variables for multi-model pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_pipeline = []

# Gemini AI Ingredient Detector Class
class GeminiIngredientDetector:
    def __init__(self, api_key):
        """Initialize Gemini for ingredient detection"""
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Test the connection
            test_response = self.model.generate_content("Hello")
            print("✅ Gemini AI Ingredient Detector loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
        
    def detect_ingredients(self, image):
        """Use Gemini to identify ingredients in food image"""
        try:
            # Convert image to appropriate format
            if isinstance(image, str):  # File path
                with open(image, 'rb') as f:
                    image_data = f.read()
            else:  # PIL Image
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                image_data = buffer.getvalue()
            
            # Create image part for Gemini
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_data).decode('utf-8')
                }
            ]
            
            # Specialized prompt for comprehensive ingredient detection
            prompt = """
            Analyze this food image and identify ALL visible ingredients, components, and food items with high accuracy.
            
            IMPORTANT: Only identify ingredients you can clearly see. Do not guess or assume.
            
            Focus on detecting:
            1. Primary ingredients (proteins like paneer, chicken, mutton, fish, eggs)
            2. Dairy products (milk, cheese, butter, ghee, yogurt, cream)
            3. Vegetables and fruits (onions, tomatoes, garlic, ginger, leafy greens, etc.)
            4. Grains and cereals (rice, wheat, bread, naan, roti)
            5. Legumes and pulses (dal, chickpeas, lentils, beans)
            6. Nuts and seeds (almonds, cashews, sesame, etc.)
            7. Spices and herbs (turmeric, cumin, coriander, mint, cilantro, etc.)
            8. Oils and fats
            
            Be very specific and conservative:
            - Only report what you can actually see in the image
            - Don't identify paneer unless you're absolutely certain it's visible
            - Be specific: if you see ghee, say "ghee" not "butter"
            - If you see specific vegetables, name them individually
            - Use confidence scores between 0.5-0.95 (be realistic)
            
            Return your response as a JSON array in this exact format:
            [
                {"name": "ingredient_name", "confidence": 0.85, "category": "protein"},
                {"name": "another_ingredient", "confidence": 0.75, "category": "dairy"}
            ]
            
            Categories: protein, dairy, vegetable, grain, spice, nut, oil, fruit, legume, processed, herb, other
            
            Only return the JSON array, no other text.
            """
            
            # Generate content with image
            response = self.model.generate_content([prompt] + image_parts)
            
            # Parse JSON response
            try:
                # Clean up response text
                response_text = response.text.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    ingredients_data = json.loads(json_str)
                else:
                    # Try parsing the entire response as JSON
                    ingredients_data = json.loads(response_text)
                
                detected_ingredients = []
                for item in ingredients_data:
                    if isinstance(item, dict) and 'name' in item:
                        confidence = float(item.get('confidence', 0.7))
                        
                        # Cap confidence at reasonable levels
                        confidence = min(confidence, 0.95)
                        
                        # Skip very low confidence detections
                        if confidence < 0.5:
                            continue
                            
                        detected_ingredients.append({
                            'name': item['name'].lower().strip(),
                            'confidence': confidence,
                            'category': item.get('category', 'unknown'),
                            'model_source': 'gemini_vision',
                            'detection_type': 'ai_vision_analysis'
                        })
                
                return detected_ingredients
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"JSON parsing failed: {e}")
                # Fallback: extract ingredients from text response
                return self._extract_ingredients_from_text(response.text)
            
        except Exception as e:
            print(f"Error in Gemini detection: {e}")
            return []
    
    def _extract_ingredients_from_text(self, text):
        """Fallback method to extract ingredients from text response"""
        ingredients = []
        
        # Common Indian and general ingredients
        common_ingredients = [
            'paneer', 'cottage cheese', 'tomato', 'onion', 'garlic', 'ginger', 
            'turmeric', 'cumin', 'coriander', 'mint', 'cilantro', 'spinach',
            'potato', 'cauliflower', 'peas', 'carrot', 'bell pepper', 'chili',
            'rice', 'wheat', 'bread', 'naan', 'roti', 'dal', 'lentils',
            'chickpeas', 'chicken', 'mutton', 'fish', 'eggs', 'milk', 'ghee',
            'oil', 'butter', 'yogurt', 'cream', 'cheese', 'almonds', 'cashews'
        ]
        
        text_lower = text.lower()
        
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                # Avoid duplicates
                if not any(existing['name'] == ingredient for existing in ingredients):
                    ingredients.append({
                        'name': ingredient,
                        'confidence': 0.6,  # Conservative confidence for text extraction
                        'category': self._categorize_ingredient(ingredient),
                        'model_source': 'gemini_text_extraction',
                        'detection_type': 'text_analysis'
                    })
        
        return ingredients
    
    def _categorize_ingredient(self, ingredient):
        """Categorize ingredients for better organization"""
        categories = {
            'protein': ['paneer', 'cottage cheese', 'chicken', 'mutton', 'fish', 'eggs'],
            'dairy': ['milk', 'ghee', 'butter', 'yogurt', 'cream', 'cheese'],
            'vegetable': ['tomato', 'onion', 'garlic', 'ginger', 'spinach', 'potato', 'cauliflower', 'peas', 'carrot', 'bell pepper'],
            'spice': ['turmeric', 'cumin', 'coriander', 'chili'],
            'herb': ['mint', 'cilantro'],
            'grain': ['rice', 'wheat', 'bread', 'naan', 'roti'],
            'legume': ['dal', 'lentils', 'chickpeas'],
            'nut': ['almonds', 'cashews'],
            'oil': ['oil']
        }
        
        for category, items in categories.items():
            if ingredient.lower() in items:
                return category
        
        return 'other'

# YOLOv8 Paneer Detector Class
class YOLOv8PaneerDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize YOLOv8 paneer detector with your trained best.pt model"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        print(f"🥛 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class mapping for your trained model
        self.class_names = ['Paneer', 'mint']  # Based on your training data
        
        print(f"✅ YOLOv8 Paneer Detector loaded on {self.device}")
        
    def detect_ingredients(self, image):
        """Detect paneer and mint in image using YOLOv8"""
        try:
            # Run inference
            results = self.model(image)
            
            detected_ingredients = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf)
                        cls = int(box.cls)
                        
                        # Cap confidence at 100% and ensure it meets threshold
                        conf = min(conf, 1.0)
                        
                        if conf >= self.confidence_threshold and cls < len(self.class_names):
                            class_name = self.class_names[cls].lower()
                            bbox = box.xyxy.cpu().numpy().flatten().tolist()
                            
                            detected_ingredients.append({
                                'name': class_name,
                                'confidence': conf,
                                'bbox': bbox,
                                'model_source': 'yolov8_paneer_detector',
                                'class_id': cls,
                                'detection_type': 'object_detection'
                            })
            
            return detected_ingredients
            
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            return []

# Your existing model classes
class FoodAllergenDetector(nn.Module):
    def __init__(self, num_classes):
        super(FoodAllergenDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def load_multi_model_pipeline():
    """Load three-model pipeline: YOLOv8 + Gemini + General Model"""
    global models_pipeline
    
    try:
        models_pipeline = []
        
        # Model 1: YOLOv8 Paneer & Mint Detector (PRIMARY)
        yolov8_model_path = "best.pt"  
        
        # Try multiple possible paths
        possible_paths = [
            yolov8_model_path,
            "best.pt",
            "food_detectors/paneer_mint_yolov8/weights/best.pt",
            os.path.join("food_detectors", "paneer_mint_yolov8", "weights", "best.pt")
        ]
        
        yolo_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Use higher confidence threshold to reduce false positives
                    yolov8_detector = YOLOv8PaneerDetector(path, confidence_threshold=0.7)
                    
                    models_pipeline.append({
                        'name': 'yolov8_paneer_detector',
                        'model': yolov8_detector,
                        'ingredients': ['paneer', 'mint'],
                        'weight': 0.9,  # Slightly reduced weight
                        'specialty': 'yolov8_paneer'
                    })
                    
                    print(f"✅ YOLOv8 Paneer Detector loaded from: {path}")
                    yolo_loaded = True
                    break
                    
                except Exception as e:
                    print(f"❌ Failed to load YOLOv8 from {path}: {e}")
                    continue
        
        if not yolo_loaded:
            print("⚠️  YOLOv8 model not found in any expected location")
        
        # Model 2: Gemini Vision AI (SECONDARY)
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            try:
                gemini_detector = GeminiIngredientDetector(gemini_api_key)
                
                models_pipeline.append({
                    'name': 'gemini_vision',
                    'model': gemini_detector,
                    'ingredients': [],  # Dynamic - Gemini can identify any ingredient
                    'weight': 0.8,  # Moderate weight for Gemini
                    'specialty': 'ai_vision'
                })
                
                print("✅ Gemini Vision AI loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load Gemini: {e}")
        else:
            print("⚠️  GEMINI_API_KEY not found in environment variables")
        
        # Model 3: Your existing general food model (FALLBACK)
        if os.path.exists('food_detector.pth') and os.path.exists('pytorch_ingredients.pkl'):
            try:
                with open('pytorch_ingredients.pkl', 'rb') as f:
                    ingredients_list_1 = pickle.load(f)
                
                model_1 = FoodAllergenDetector(len(ingredients_list_1))
                model_1.load_state_dict(torch.load('food_detector.pth', map_location=device))
                model_1.to(device)
                model_1.eval()
                
                models_pipeline.append({
                    'name': 'general_food_model',
                    'model': model_1,
                    'ingredients': ingredients_list_1,
                    'weight': 0.7,  # Lower weight since others are primary
                    'specialty': 'general'
                })
                print("✅ General food model loaded as fallback")
            except Exception as e:
                print(f"❌ Failed to load general food model: {e}")
        
        print(f"🚀 Multi-model pipeline loaded with {len(models_pipeline)} models")
        return True
        
    except Exception as e:
        print(f"❌ Error loading multi-model pipeline: {e}")
        return False

def validate_allergen_match(allergen, ingredient):
    """Prevent obviously incorrect matches"""
    
    # Define invalid combinations
    invalid_combinations = [
        # Dairy cross-contamination prevention
        ('cheese', 'ghee'),
        ('cheese', 'butter'),  
        ('butter', 'paneer'),
        ('butter', 'ghee'),
        ('ghee', 'cheese'),
        ('ghee', 'butter'),
        
        # Prevent nut confusion
        ('nuts', 'coconut'),
        ('tree nuts', 'coconut'),
        
        # Prevent grain confusion
        ('wheat', 'rice'),
        ('gluten', 'rice'),
        
        # Prevent spice confusion
        ('nut', 'nutmeg'),  # nutmeg is a spice, not a nut
    ]
    
    allergen_lower = allergen.lower()
    ingredient_lower = ingredient.lower()
    
    for invalid_allergen, invalid_ingredient in invalid_combinations:
        if (invalid_allergen == allergen_lower and invalid_ingredient in ingredient_lower) or \
           (invalid_ingredient == allergen_lower and invalid_allergen in ingredient_lower):
            print(f"🚫 Blocked invalid match: {allergen} -> {ingredient}")
            return False
    
    return True

def enhanced_allergen_matching(detected_ingredients, user_allergies):
    """Fixed allergen matching with proper specificity"""
    allergen_warnings = []
    
    for ingredient in detected_ingredients:
        ingredient_name_lower = ingredient['name'].lower().strip()
        
        for user_allergy in user_allergies:
            allergen_lower = user_allergy['allergen_name'].lower().strip()
            
            match_found = False
            match_type = 'no_match'
            match_confidence = 0.0
            
            # 1. EXACT MATCHING (highest priority)
            if ingredient_name_lower == allergen_lower:
                match_found = True
                match_type = 'exact_match'
                match_confidence = 1.0
            
            # 2. SPECIFIC INGREDIENT MATCHING
            elif allergen_lower == 'paneer':
                paneer_variants = ['paneer', 'cottage cheese', 'indian cottage cheese', 'fresh cheese']
                if any(variant in ingredient_name_lower for variant in paneer_variants):
                    match_found = True
                    match_type = 'paneer_variant'
                    match_confidence = 0.95
            
            elif allergen_lower == 'cheese':
                # Only match actual cheese types, NOT ghee, butter, or paneer
                cheese_types = ['cheese', 'cheddar', 'mozzarella', 'parmesan', 'gouda', 'swiss']
                if any(cheese_type in ingredient_name_lower for cheese_type in cheese_types):
                    # Exclude dairy products that aren't cheese
                    if not any(exclude in ingredient_name_lower for exclude in ['ghee', 'butter', 'paneer']):
                        match_found = True
                        match_type = 'cheese_variant'
                        match_confidence = 0.9
            
            elif allergen_lower == 'butter':
                if 'butter' in ingredient_name_lower and 'ghee' not in ingredient_name_lower:
                    match_found = True
                    match_type = 'butter_match'
                    match_confidence = 0.9
            
            elif allergen_lower == 'ghee':
                if 'ghee' in ingredient_name_lower:
                    match_found = True
                    match_type = 'ghee_match'
                    match_confidence = 0.95
            
            # 3. BROAD CATEGORY MATCHING
            elif allergen_lower in ['dairy', 'milk allergy', 'lactose intolerance', 'lactose']:
                dairy_products = ['milk', 'cheese', 'butter', 'ghee', 'cream', 'yogurt', 'paneer', 'curd', 'whey']
                if any(dairy_prod in ingredient_name_lower for dairy_prod in dairy_products):
                    match_found = True
                    match_type = 'broad_dairy_match'
                    match_confidence = 0.8
            
            elif 'nut' in allergen_lower or allergen_lower in ['tree nuts', 'nuts']:
                tree_nuts = ['almond', 'cashew', 'walnut', 'hazelnut', 'pecan', 'pistachio', 'macadamia', 'brazil nut']
                if any(nut in ingredient_name_lower for nut in tree_nuts):
                    match_found = True
                    match_type = 'tree_nut_match'
                    match_confidence = 0.85
            
            elif allergen_lower == 'peanut':
                if 'peanut' in ingredient_name_lower or 'groundnut' in ingredient_name_lower:
                    match_found = True
                    match_type = 'peanut_match'
                    match_confidence = 0.9
            
            # 4. CONSERVATIVE SUBSTRING MATCHING
            elif len(allergen_lower) > 4:  # Only for longer allergen names
                if allergen_lower in ingredient_name_lower:
                    # Additional validation to prevent false matches
                    if validate_allergen_match(allergen_lower, ingredient_name_lower):
                        match_found = True
                        match_type = 'substring_match'
                        match_confidence = 0.7
            
            # Add warning if valid match found
            if match_found and match_confidence > 0.5:
                # Adjust confidence based on detection confidence
                final_confidence = (ingredient['confidence'] + match_confidence) / 2
                
                allergen_warnings.append({
                    'allergen': user_allergy['allergen_name'],
                    'ingredient': ingredient['name'],
                    'confidence': final_confidence,
                    'match_confidence': match_confidence,
                    'severity': user_allergy['severity'],
                    'match_type': match_type,
                    'bbox': ingredient.get('bbox'),
                    'detection_method': ingredient.get('detection_type', 'unknown'),
                    'model_source': ingredient.get('model_source', 'unknown')
                })
    
    # Remove duplicate warnings
    seen = set()
    unique_warnings = []
    for warning in allergen_warnings:
        key = (warning['allergen'].lower(), warning['ingredient'].lower())
        if key not in seen:
            seen.add(key)
            unique_warnings.append(warning)
    
    return unique_warnings

def multi_model_predict(image, confidence_threshold=0.4):
    """Enhanced prediction with better confidence handling and duplicate removal"""
    if not models_pipeline:
        raise Exception("Multi-model pipeline not loaded")
    
    combined_predictions = {}
    model_results = {}
    
    for model_info in models_pipeline:
        model_name = model_info['name']
        try:
            if model_info['specialty'] == 'yolov8_paneer':
                # YOLOv8 Object Detection with better confidence handling
                yolo_detector = model_info['model']
                detections = yolo_detector.detect_ingredients(image)
                weight = model_info['weight']
                
                model_results['yolo'] = len(detections)
                
                for detection in detections:
                    ingredient = detection['name']
                    confidence = min(detection['confidence'] * weight, 1.0)  # Cap at 100%
                    
                    # Add stricter confidence threshold for paneer to reduce false positives
                    if ingredient.lower() == 'paneer' and confidence < 0.8:
                        print(f"⚠️ Low confidence paneer detection skipped: {confidence:.2f}")
                        continue
                    
                    # Only add if confidence is reasonable
                    if confidence >= confidence_threshold:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': confidence,
                            'bbox': detection.get('bbox'),
                            'model_source': detection['model_source'],
                            'detection_type': detection['detection_type']
                        }
            
            elif model_info['specialty'] == 'ai_vision':
                # Gemini Vision AI with better filtering
                gemini_detector = model_info['model']
                detections = gemini_detector.detect_ingredients(image)
                weight = model_info['weight']
                
                model_results['gemini'] = len(detections)
                
                for detection in detections:
                    ingredient = detection['name']
                    confidence = min(detection['confidence'] * weight, 1.0)  # Cap at 100%
                    
                    # Skip very low confidence detections
                    if confidence < 0.4:
                        continue
                    
                    # Special handling for commonly over-detected ingredients
                    if ingredient.lower() in ['paneer', 'cheese'] and confidence < 0.6:
                        print(f"⚠️ Low confidence {ingredient} detection skipped: {confidence:.2f}")
                        continue
                    
                    # Avoid duplicates - only add if not detected by YOLOv8 or confidence is significantly higher
                    if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence'] + 0.1:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': confidence,
                            'category': detection.get('category', 'unknown'),
                            'model_source': detection['model_source'],
                            'detection_type': detection['detection_type']
                        }
            
            elif model_info['specialty'] == 'general':
                # General model with conservative approach
                model = model_info['model']
                ingredients = model_info['ingredients']
                weight = model_info['weight']
                
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if isinstance(image, str):
                    image_pil = Image.open(image).convert('RGB')
                else:
                    image_pil = image
                
                image_tensor = transform(image_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    predictions = outputs.cpu().numpy()[0]
                
                # Use higher threshold for general model to reduce noise
                detected_indices = np.where(predictions > max(confidence_threshold, 0.5))[0]
                model_results['general'] = len(detected_indices)
                
                for idx in detected_indices:
                    if idx < len(ingredients):
                        ingredient = ingredients[idx]
                        confidence = min(float(predictions[idx]) * weight, 1.0)
                        
                        # Only add if not already detected with higher confidence
                        if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence']:
                            combined_predictions[ingredient] = {
                                'name': ingredient,
                                'confidence': confidence,
                                'model_source': model_info['name'],
                                'detection_type': 'classification'
                            }
                            
        except Exception as e:
            print(f"Error in model {model_name}: {e}")
            model_results[model_name.split('_')[0]] = 0
            continue
    
    # Convert to list and sort by confidence
    detected_ingredients = list(combined_predictions.values())
    detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Remove very similar ingredients to reduce duplicates
    filtered_ingredients = []
    for ingredient in detected_ingredients:
        ingredient_name = ingredient['name'].lower()
        
        # Check if similar ingredient already exists
        is_duplicate = False
        for existing in filtered_ingredients:
            existing_name = existing['name'].lower()
            
            # Check for obvious duplicates or very similar names
            if (ingredient_name == existing_name or 
                (len(ingredient_name) > 4 and ingredient_name in existing_name) or
                (len(existing_name) > 4 and existing_name in ingredient_name)):
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_ingredients.append(ingredient)
    
    print(f"🔍 Model Results: {model_results}")
    print(f"🧹 Filtered {len(detected_ingredients) - len(filtered_ingredients)} duplicate ingredients")
    
    return filtered_ingredients[:25]  # Top 25 unique ingredients

# MongoDB Helper Functions
def get_user_allergies(user_id):
    """Get user allergies from MongoDB"""
    allergies = list(allergies_collection.find({"user_id": user_id}))
    return allergies

def save_scan_to_history(user_id, detected_ingredients, allergen_warnings, is_safe, confidence_score):
    """Save scan to MongoDB history"""
    scan_doc = {
        "user_id": user_id,
        "detected_ingredients": detected_ingredients,
        "allergen_warnings": allergen_warnings,
        "is_safe": is_safe,
        "confidence_score": confidence_score,
        "created_at": datetime.utcnow()
    }
    result = scan_history_collection.insert_one(scan_doc)
    return str(result.inserted_id)

@app.route('/')
def health_check():
    """Health check endpoint for Render and keep-alive"""
    gemini_available = any(model.get('specialty') == 'ai_vision' for model in models_pipeline)
    yolo_available = any(model.get('specialty') == 'yolov8_paneer' for model in models_pipeline)
    
    return jsonify({
        'status': 'healthy',
        'message': 'FoodGuard API backend with Gemini AI is running',
        'models_loaded': len(models_pipeline),
        'mongodb_connected': True,
        'yolov8_available': yolo_available,
        'gemini_available': gemini_available,
        'ai_enhanced': gemini_available,
        'fixes_applied': 'false_positive_reduction',
        'timestamp': datetime.utcnow().isoformat()
    })

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print(f"📝 Registration attempt for: {data.get('email', 'unknown')}")
        
        required_fields = ['email', 'password', 'first_name', 'last_name']
        if not all(field in data for field in required_fields):
            missing_fields = [f for f in required_fields if f not in data]
            print(f"❌ Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        if len(data['password']) < 6:
            print(f"❌ Password too short for {data['email']}")
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        email = data['email'].lower().strip()
        print(f"📧 Processing email: {email}")
        
        # Check if user already exists
        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            print(f"❌ Email already exists: {email}")
            return jsonify({'error': 'Email already registered'}), 409
        
        # Create user document
        user_doc = {
            "email": email,
            "password_hash": generate_password_hash(data['password']),
            "first_name": data['first_name'].strip(),
            "last_name": data['last_name'].strip(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        # Insert user into MongoDB
        try:
            result = users_collection.insert_one(user_doc)
            user_id = str(result.inserted_id)
            print(f"✅ User inserted with ID: {user_id}")
        except pymongo.errors.DuplicateKeyError as e:
            print(f"❌ Duplicate key error during insert: {e}")
            return jsonify({'error': 'Email already registered (duplicate key)'}), 409
        except Exception as e:
            print(f"❌ Database insert error: {e}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        
        # Create access token
        access_token = create_access_token(identity=user_id)
        
        response_data = {
            'message': 'Account created successfully',
            'access_token': access_token,
            'user': {
                'id': user_id,
                'email': email,
                'first_name': data['first_name'],
                'last_name': data['last_name']
            }
        }
        
        print(f"🎉 Registration completed successfully for: {email}")
        return jsonify(response_data), 201
        
    except Exception as e:
        print(f"❌ Unexpected registration error: {str(e)}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"🔐 Login attempt for: {data.get('email', 'unknown')}")
        
        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400
        
        email = data['email'].lower().strip()
        user = users_collection.find_one({"email": email, "is_active": True})
        
        if user and check_password_hash(user['password_hash'], data['password']):
            access_token = create_access_token(identity=str(user['_id']))
            print(f"✅ Login successful for: {email}")
            return jsonify({
                'access_token': access_token,
                'user': {
                    'id': str(user['_id']),
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name']
                }
            })
        
        print(f"❌ Invalid credentials for: {email}")
        return jsonify({'error': 'Invalid email or password'}), 401
        
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user allergies
        allergies = list(allergies_collection.find({"user_id": user_id}))
        allergy_list = [{
            'name': allergy['allergen_name'], 
            'severity': allergy['severity'], 
            'notes': allergy.get('notes', '')
        } for allergy in allergies]
        
        # Get scan count
        total_scans = scan_history_collection.count_documents({"user_id": user_id})
        
        return jsonify({
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'first_name': user['first_name'],
                'last_name': user['last_name'],
                'created_at': user['created_at'].isoformat()
            },
            'allergies': allergy_list,
            'total_scans': total_scans
        })
        
    except Exception as e:
        return jsonify({'error': 'Failed to get profile'}), 500

@app.route('/api/profile/allergies', methods=['POST'])
@jwt_required()
def update_allergies():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if 'allergies' not in data:
        return jsonify({'error': 'Missing allergies data'}), 400
    
    try:
        # Clear existing allergies
        allergies_collection.delete_many({"user_id": user_id})
        
        # Add new allergies
        for allergy_data in data['allergies']:
            if 'name' not in allergy_data:
                continue
                
            allergy_doc = {
                "user_id": user_id,
                "allergen_name": allergy_data['name'].lower().strip(),
                "severity": allergy_data.get('severity', 'moderate'),
                "notes": allergy_data.get('notes', ''),
                "created_at": datetime.utcnow()
            }
            allergies_collection.insert_one(allergy_doc)
        
        return jsonify({'message': 'Allergies updated successfully'})
        
    except Exception as e:
        return jsonify({'error': 'Failed to update allergies'}), 500

@app.route('/api/analyze-food', methods=['POST'])
@jwt_required()
def analyze_food():
    user_id = get_jwt_identity()
    
    if not models_pipeline:
        return jsonify({'error': 'Multi-model pipeline not available. Please check server logs.'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Load user's allergies from MongoDB
        user_allergies = get_user_allergies(user_id)
        
        # Save temporary image for processing
        temp_filename = f"temp_{user_id}_{int(time.time())}.jpg"
        image_file.save(temp_filename)
        
        try:
            # Process with enhanced multi-model pipeline (YOLOv8 + Gemini + fallback)
            detected_ingredients = multi_model_predict(temp_filename)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        if not detected_ingredients:
            return jsonify({
                'scan_id': None,
                'ingredients': [],
                'allergen_warnings': [],
                'is_safe': True,
                'message': 'No ingredients detected. Try a clearer image or different angle.',
                'user_allergies_count': len(user_allergies)
            })
        
        # Enhanced allergen matching with fixed logic
        allergen_warnings = enhanced_allergen_matching(detected_ingredients, user_allergies)
        
        avg_confidence = np.mean([ing['confidence'] for ing in detected_ingredients]) if detected_ingredients else 0.0
        
        # Count detections by model type
        yolo_detections = len([ing for ing in detected_ingredients if ing.get('detection_type') == 'object_detection'])
        gemini_detections = len([ing for ing in detected_ingredients if 'gemini' in ing.get('model_source', '')])
        general_detections = len([ing for ing in detected_ingredients if ing.get('model_source') == 'general_food_model'])
        
        # Save scan to MongoDB history
        scan_id = save_scan_to_history(
            user_id=user_id,
            detected_ingredients=detected_ingredients,
            allergen_warnings=allergen_warnings,
            is_safe=len(allergen_warnings) == 0,
            confidence_score=float(avg_confidence)
        )
        
        return jsonify({
            'scan_id': scan_id,
            'ingredients': detected_ingredients,
            'allergen_warnings': allergen_warnings,
            'is_safe': len(allergen_warnings) == 0,
            'confidence_score': float(avg_confidence),
            'user_allergies_count': len(user_allergies),
            'models_used': len(models_pipeline),
            'model_breakdown': {
                'yolov8_detections': yolo_detections,
                'gemini_detections': gemini_detections,
                'general_detections': general_detections,
                'total_detections': len(detected_ingredients)
            },
            'message': 'Enhanced YOLOv8 + Gemini AI + General Model analysis completed',
            'ai_enhanced': gemini_detections > 0,
            'false_positive_reduction': True
        })
        
    except Exception as e:
        print(f"Multi-model analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/scan-history', methods=['GET'])
@jwt_required()
def get_scan_history():
    user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 50)
    
    skip = (page - 1) * per_page
    
    # Get scans with pagination
    scans_cursor = scan_history_collection.find({"user_id": user_id}) \
        .sort("created_at", -1) \
        .skip(skip) \
        .limit(per_page)
    
    scans = list(scans_cursor)
    total = scan_history_collection.count_documents({"user_id": user_id})
    
    scan_history = [{
        'id': str(scan['_id']),
        'ingredients': scan.get('detected_ingredients', []),
        'warnings': scan.get('allergen_warnings', []),
        'is_safe': scan.get('is_safe', True),
        'confidence': scan.get('confidence_score', 0.0),
        'created_at': scan['created_at'].isoformat()
    } for scan in scans]
    
    return jsonify({
        'scans': scan_history,
        'total': total,
        'pages': (total + per_page - 1) // per_page,
        'current_page': page,
        'has_next': skip + per_page < total,
        'has_prev': page > 1
    })

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    return jsonify({
        'models_loaded': len(models_pipeline),
        'models': [
            {
                'name': model_info['name'],
                'weight': model_info['weight'],
                'specialty': model_info['specialty'],
                'ingredients_count': len(model_info['ingredients']) if isinstance(model_info.get('ingredients'), list) else 'dynamic'
            } for model_info in models_pipeline
        ],
        'device': str(device),
        'yolov8_available': any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline),
        'gemini_available': any(model['specialty'] == 'ai_vision' for model in models_pipeline),
        'general_model_available': any(model['specialty'] == 'general' for model in models_pipeline),
        'false_positive_fixes': True,
        'enhanced_allergen_matching': True
    })

# Debug endpoints
@app.route('/api/debug/db-info')
def debug_db_info():
    return jsonify({
        'database_name': db.name,
        'collections': db.list_collection_names(),
        'users_collection_name': users_collection.name,
        'server_info': client.server_info()['version']
    })

@app.route('/api/debug/users-count', methods=['GET'])
def debug_users_count():
    try:
        count = users_collection.count_documents({})
        recent_users = list(users_collection.find({}).sort("created_at", -1).limit(5))
        
        # Remove sensitive data
        for user in recent_users:
            user['_id'] = str(user['_id'])
            user.pop('password_hash', None)
        
        return jsonify({
            'total_users': count,
            'recent_users': recent_users,
            'collection_name': users_collection.name,
            'database_name': db.name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/db-verify', methods=['GET'])
def debug_db_verify():
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        
        # Get database info
        db_stats = db.command('dbStats')
        
        # Get all collections and their stats
        collections_info = {}
        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            count = collection.count_documents({})
            collections_info[collection_name] = {
                'count': count,
                'sample_docs': list(collection.find({}).limit(2))
            }
            
            # Convert ObjectId to string for JSON serialization
            for doc in collections_info[collection_name]['sample_docs']:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'password_hash' in doc:
                    doc.pop('password_hash')  # Remove sensitive data
        
        # Test write operation
        test_doc = {
            'test_write': True,
            'timestamp': datetime.utcnow(),
            'test_id': str(uuid.uuid4())
        }
        
        test_result = db['test_collection'].insert_one(test_doc)
        db['test_collection'].delete_one({'_id': test_result.inserted_id})
        
        return jsonify({
            'mongodb_status': 'connected',
            'database_name': db.name,
            'server_version': client.server_info()['version'],
            'db_size_mb': round(db_stats['dataSize'] / (1024*1024), 2),
            'collections': collections_info,
            'write_test': 'successful',
            'connection_string_set': bool(os.getenv('MONGODB_URI')),
            'gemini_api_key_set': bool(os.getenv('GEMINI_API_KEY')),
            'total_users': users_collection.count_documents({}),
            'indexes_created': True,
            'false_positive_fixes': True
        })
        
    except Exception as e:
        return jsonify({
            'mongodb_status': 'error',
            'error': str(e),
            'connection_string_set': bool(os.getenv('MONGODB_URI')),
            'gemini_api_key_set': bool(os.getenv('GEMINI_API_KEY'))
        }), 500

def initialize_app():
    """Initialize MongoDB connections and load enhanced multi-model pipeline"""
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        # Load enhanced pipeline
        pipeline_loaded = load_multi_model_pipeline()
        
        if pipeline_loaded:
            gemini_loaded = any(model['specialty'] == 'ai_vision' for model in models_pipeline)
            yolo_loaded = any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline)
            
            print("🚀 Enhanced FoodGuard API Server initialized successfully!")
            print(f"   - YOLOv8 Loaded: {'✅' if yolo_loaded else '❌'}")
            print(f"   - Gemini AI Loaded: {'✅' if gemini_loaded else '❌'}")
            print(f"   - Total Models: {len(models_pipeline)}")
            print(f"   - False Positive Fixes: ✅")
        else:
            print("⚠️  Server started but some models may not be loaded.")
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")

if __name__ == '__main__':
    initialize_app()
    
    print("🍽️ Enhanced FoodGuard API Server with False Positive Fixes Starting...")
    print("📝 Required files and environment variables:")
    print("   - best.pt (YOLOv8 model)")
    print("   - MONGODB_URI (MongoDB connection string)")
    print("   - GEMINI_API_KEY (Google Gemini AI API key)")
    print("   - JWT_SECRET_KEY (for authentication)")
    print()
    print("🎯 Enhanced Model Pipeline:")
    print("   1. YOLOv8: Stricter paneer detection (80% threshold)")
    print("   2. Gemini AI: Conservative ingredient identification")
    print("   3. General Model: Fallback classification")
    print()
    print("🔧 Key Fixes Applied:")
    print("   ✅ Fixed 'cheese detected in ghee' false positive")
    print("   ✅ Reduced paneer over-detection with higher thresholds")
    print("   ✅ Enhanced allergen matching specificity")
    print("   ✅ Confidence capping (max 100%)")
    print("   ✅ Better duplicate ingredient filtering")
    print("   ✅ Invalid allergen match prevention")
    print()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
