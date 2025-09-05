import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pickle
import requests
from pathlib import Path
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid
import os
import time
import sys
from dotenv import load_dotenv
import logging

# MongoDB imports with URL parsing
import pymongo
from bson import ObjectId
from urllib.parse import urlparse, urlunparse, quote_plus

load_dotenv()

app = Flask(__name__)

# Enhanced CORS Configuration
CORS(app, 
     origins=[
         "https://foodguard-eight.vercel.app",
         "https://foodguard-frontend.vercel.app", 
         "http://localhost:3000",
         "http://127.0.0.1:3000"
     ],
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
     supports_credentials=False,
     max_age=86400
)

# Add explicit OPTIONS handler
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin', '*'))
        response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,Origin,X-Requested-With")
        response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS,HEAD")
        response.headers.add('Access-Control-Max-Age', '86400')
        return response

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in [
        "https://foodguard-eight.vercel.app",
        "https://foodguard-frontend.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', "Content-Type,Authorization,Accept,Origin,X-Requested-With")
    response.headers.add('Access-Control-Allow-Methods', "GET,PUT,POST,DELETE,OPTIONS,HEAD")
    return response

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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
mlb_encoder = None

# Model URLs Configuration - FIXED PATHS
MODEL_URLS = {
    'yolov8': os.environ.get('YOLOV8_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/best.pt'),
    'food_detector': os.environ.get('FOOD_DETECTOR_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/food_detector.pth'),
    'ingredients_list': os.environ.get('INGREDIENTS_LIST_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/pytorch_ingredients.pkl'),
    'label_binarizer': os.environ.get('MLB_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/mlb.pkl')
}

# FIXED: Use models/ directory consistently
MODEL_PATHS = {
    'yolov8': 'models/best.pt',
    'food_detector': 'models/food_detector.pth', 
    'ingredients_list': 'models/pytorch_ingredients.pkl',
    'label_binarizer': 'models/mlb.pkl'
}

def download_model_file(url, local_path, model_name):
    """Download a single model file from GitHub with enhanced error handling"""
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        print(f"✅ {model_name} already exists locally ({file_size/1024/1024:.1f}MB)")
        if file_size > 1024:  # At least 1KB
            return True
        else:
            print(f"⚠️ {model_name} file seems corrupted, re-downloading...")
            os.remove(local_path)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"📥 Downloading {model_name} from GitHub (attempt {attempt + 1}/{max_retries})...")
            print(f"URL: {url}")
            
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with longer timeout and retries
            response = requests.get(url, stream=True, timeout=600)  # 10 minutes timeout
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):  # Larger chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024) == 0:  # Print every MB
                                print(f"📊 {model_name} progress: {progress:.1f}%")
            
            file_size = os.path.getsize(local_path)
            print(f"\n✅ {model_name} downloaded successfully ({file_size/1024/1024:.1f}MB)")
            
            # Enhanced integrity check
            if file_size < 1024:
                print(f"❌ {model_name} file corrupted (too small: {file_size} bytes)")
                os.remove(local_path)
                continue
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download attempt {attempt + 1} failed for {model_name}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏰ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            continue
        except Exception as e:
            print(f"❌ Unexpected error downloading {model_name}: {e}")
            break
    
    print(f"💥 Failed to download {model_name} after {max_retries} attempts")
    if os.path.exists(local_path):
        os.remove(local_path)
    return False

def download_all_models():
    """Download all required models from GitHub"""
    print("🚀 Starting model downloads from GitHub...")
    
    success_count = 0
    total_models = len(MODEL_URLS)
    
    for model_key, url in MODEL_URLS.items():
        local_path = MODEL_PATHS[model_key]
        model_name = model_key.replace('_', ' ').title()
        
        if download_model_file(url, local_path, model_name):
            success_count += 1
        else:
            print(f"💥 Failed to download {model_name}")
    
    print(f"\n📊 Download Summary: {success_count}/{total_models} models downloaded")
    
    if success_count >= 1:  # At least one model downloaded
        print("✅ At least one model downloaded successfully!")
        return True
    else:
        print("❌ No models downloaded successfully!")
        return False

# YOLOv8 Paneer Detector Class
class YOLOv8PaneerDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize YOLOv8 paneer detector"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        print(f"🥛 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = ['Paneer', 'mint']
        print(f"✅ YOLOv8 Paneer Detector loaded on {self.device}")
        
    def detect_ingredients(self, image):
        """Detect paneer and mint in image using YOLOv8"""
        try:
            results = self.model(image, verbose=False)
            detected_ingredients = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf)
                        cls = int(box.cls)
                        
                        if conf >= self.confidence_threshold and cls < len(self.class_names):
                            class_name = self.class_names[cls].lower()
                            bbox = box.xyxy.cpu().numpy().flatten().tolist()
                            
                            detected_ingredients.append({
                                'name': class_name,
                                'confidence': conf,
                                'bbox': bbox,
                                'model_source': 'yolov8_paneer_detector',
                                'class_id': cls
                            })
            
            return detected_ingredients
            
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            return []

# Food Allergen Detector Class
class FoodAllergenDetector(nn.Module):
    def __init__(self, num_classes):
        super(FoodAllergenDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def load_multi_model_pipeline():
    """Load all models with enhanced error handling"""
    global models_pipeline, mlb_encoder
    
    try:
        print("🚀 Starting multi-model pipeline initialization...")
        
        # Download models first
        download_success = download_all_models()
        
        models_pipeline = []
        mlb_encoder = None
        successful_models = 0
        
        # Load MLB Label Binarizer
        mlb_path = MODEL_PATHS['label_binarizer']
        if os.path.exists(mlb_path):
            try:
                print(f"🏷️ Loading label binarizer from: {mlb_path}")
                with open(mlb_path, 'rb') as f:
                    mlb_encoder = pickle.load(f)
                print(f"✅ Label binarizer loaded with {len(mlb_encoder.classes_)} classes")
            except Exception as e:
                print(f"❌ Failed to load MLB: {e}")
        
        # Load YOLOv8 Model
        yolov8_path = MODEL_PATHS['yolov8']
        if os.path.exists(yolov8_path):
            try:
                print(f"🥛 Loading YOLOv8 model from: {yolov8_path}")
                yolov8_detector = YOLOv8PaneerDetector(yolov8_path, confidence_threshold=0.3)
                
                # Test the model
                print("🧪 Testing YOLOv8 model...")
                dummy_img = Image.new('RGB', (640, 640), color='white')
                test_results = yolov8_detector.detect_ingredients(dummy_img)
                print(f"✅ YOLOv8 test successful: {len(test_results)} detections")
                
                models_pipeline.append({
                    'name': 'yolov8_paneer_detector',
                    'model': yolov8_detector,
                    'ingredients': ['paneer', 'mint'],
                    'weight': 1.2,
                    'specialty': 'yolov8_paneer',
                    'source': 'github'
                })
                successful_models += 1
                print(f"✅ YOLOv8 Paneer Detector loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load YOLOv8 model: {e}")
        
        # Load General Food Model
        food_detector_path = MODEL_PATHS['food_detector']
        ingredients_path = MODEL_PATHS['ingredients_list']
        
        if os.path.exists(food_detector_path) and os.path.exists(ingredients_path):
            try:
                print(f"🍽️ Loading general food model from: {food_detector_path}")
                
                with open(ingredients_path, 'rb') as f:
                    ingredients_list = pickle.load(f)
                
                model = FoodAllergenDetector(len(ingredients_list))
                model.load_state_dict(torch.load(food_detector_path, map_location=device))
                model.to(device)
                model.eval()
                
                models_pipeline.append({
                    'name': 'general_food_model',
                    'model': model,
                    'ingredients': ingredients_list,
                    'weight': 0.4,
                    'specialty': 'general',
                    'source': 'github',
                    'mlb': mlb_encoder
                })
                successful_models += 1
                print(f"✅ General food model loaded successfully ({len(ingredients_list)} ingredients)")
                
            except Exception as e:
                print(f"❌ Failed to load general food model: {e}")
        
        # Add fallback if no models loaded
        if not models_pipeline:
            print("⚠️ No models loaded, adding basic fallback...")
            basic_ingredients = ['paneer', 'cheese', 'milk', 'dairy', 'nuts', 'wheat', 'soy']
            
            models_pipeline.append({
                'name': 'basic_fallback',
                'model': None,
                'ingredients': basic_ingredients,
                'weight': 0.1,
                'specialty': 'fallback',
                'source': 'built_in'
            })
            print("✅ Basic fallback model added")
        
        final_status = len(models_pipeline) > 0
        
        print(f"🚀 Multi-model pipeline loaded with {len(models_pipeline)} models")
        print(f"📊 Successful model loads: {successful_models}")
        
        # Display final status
        yolov8_available = any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline)
        general_available = any(model['specialty'] == 'general' for model in models_pipeline)
        fallback_available = any(model['specialty'] == 'fallback' for model in models_pipeline)
        
        print(f"📊 Final Model Status:")
        print(f"   - YOLOv8 Paneer Detector: {'✅ Ready' if yolov8_available else '❌ Not loaded'}")
        print(f"   - General Food Model: {'✅ Ready' if general_available else '❌ Not loaded'}")
        print(f"   - Basic Fallback: {'✅ Ready' if fallback_available else '❌ Not loaded'}")
        print(f"   - MLB Label Decoder: {'✅ Ready' if mlb_encoder else '❌ Not loaded'}")
        
        return final_status
        
    except Exception as e:
        print(f"❌ Error in multi-model pipeline loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def multi_model_predict(image, confidence_threshold=0.2):
    """Enhanced prediction with better fallback handling"""
    if not models_pipeline:
        raise Exception("Multi-model pipeline not loaded")
    
    combined_predictions = {}
    successful_predictions = 0
    
    for model_info in models_pipeline:
        try:
            print(f"🔍 Running prediction with {model_info['name']}")
            
            if model_info['specialty'] == 'yolov8_paneer':
                yolo_detector = model_info['model']
                detections = yolo_detector.detect_ingredients(image)
                weight = model_info['weight']
                
                for detection in detections:
                    ingredient = detection['name']
                    confidence = detection['confidence'] * weight
                    
                    if 'paneer' in ingredient.lower():
                        confidence = min(confidence * 1.3, 1.0)
                    
                    if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence']:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': confidence,
                            'bbox': detection.get('bbox'),
                            'model_source': detection['model_source'],
                            'detection_type': 'object_detection'
                        }
                
                successful_predictions += 1
                print(f"✅ YOLOv8 prediction successful: {len(detections)} detections")
            
            elif model_info['specialty'] == 'general':
                model = model_info['model']
                ingredients = model_info['ingredients']
                weight = model_info['weight']
                
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
                
                detected_indices = np.where(predictions > confidence_threshold)[0]
                for idx in detected_indices:
                    if idx < len(ingredients):
                        ingredient = ingredients[idx]
                        confidence = float(predictions[idx]) * weight
                        
                        if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence']:
                            combined_predictions[ingredient] = {
                                'name': ingredient,
                                'confidence': confidence,
                                'model_source': model_info['name'],
                                'detection_type': 'classification'
                            }
                
                successful_predictions += 1
                print(f"✅ General model prediction successful: {len(detected_indices)} detections")
                
            elif model_info['specialty'] == 'fallback':
                print("🆘 Using fallback detection method")
                fallback_ingredients = ['paneer', 'dairy', 'cheese']
                
                for ingredient in fallback_ingredients:
                    if ingredient not in combined_predictions:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': 0.1,
                            'model_source': model_info['name'],
                            'detection_type': 'fallback'
                        }
                
                successful_predictions += 1
                print(f"⚠️ Fallback prediction completed")
            
        except Exception as e:
            print(f"❌ Error in model {model_info['name']}: {e}")
            continue
    
    if successful_predictions == 0:
        print("❌ All model predictions failed")
        return []
    
    detected_ingredients = list(combined_predictions.values())
    detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"✅ Final prediction completed: {len(detected_ingredients)} ingredients")
    return detected_ingredients[:25]

def get_user_allergies(user_id):
    """Get user allergies from MongoDB"""
    try:
        allergies = list(allergies_collection.find({"user_id": user_id}))
        return allergies
    except Exception as e:
        print(f"Failed to get user allergies: {e}")
        return []

def save_scan_to_history(user_id, detected_ingredients, allergen_warnings, is_safe, confidence_score):
    """Save scan to MongoDB history"""
    try:
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
    except Exception as e:
        print(f"Failed to save scan history: {e}")
        return None

# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(415)
def unsupported_media_type(e):
    return jsonify({
        'error': 'Unsupported Media Type',
        'message': 'Content-Type must be application/json'
    }), 415

# Health check route
@app.route('/')
def health_check():
    """Health check endpoint for Render and keep-alive"""
    return jsonify({
        'status': 'healthy',
        'message': 'FoodGuard API backend is running',
        'models_loaded': len(models_pipeline),
        'mongodb_connected': True,
        'yolov8_available': any(model.get('specialty') == 'yolov8_paneer' for model in models_pipeline),
        'mlb_available': mlb_encoder is not None,
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
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        email = data['email'].lower().strip()
        
        if users_collection.find_one({"email": email}):
            return jsonify({'error': 'Email already registered'}), 409
        
        user_doc = {
            "email": email,
            "password_hash": generate_password_hash(data['password']),
            "first_name": data['first_name'].strip(),
            "last_name": data['last_name'].strip(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "is_active": True
        }
        
        try:
            result = users_collection.insert_one(user_doc)
            user_id = str(result.inserted_id)
            print(f"✅ User inserted with ID: {user_id}")
            
            # Verify user was saved
            saved_user = users_collection.find_one({"_id": result.inserted_id})
            if not saved_user:
                return jsonify({'error': 'User creation verification failed'}), 500
            
        except pymongo.errors.DuplicateKeyError:
            return jsonify({'error': 'Email already registered'}), 409
        except Exception as e:
            print(f"❌ Database insert error: {e}")
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        
        access_token = create_access_token(identity=user_id)
        
        return jsonify({
            'message': 'Account created successfully',
            'access_token': access_token,
            'user': {
                'id': user_id,
                'email': email,
                'first_name': data['first_name'],
                'last_name': data['last_name']
            }
        }), 201
        
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not all(k in data for k in ['email', 'password']):
            return jsonify({'error': 'Missing email or password'}), 400
        
        email = data['email'].lower().strip()
        user = users_collection.find_one({"email": email, "is_active": True})
        
        if user and check_password_hash(user['password_hash'], data['password']):
            access_token = create_access_token(identity=str(user['_id']))
            return jsonify({
                'access_token': access_token,
                'user': {
                    'id': str(user['_id']),
                    'email': user['email'],
                    'first_name': user['first_name'],
                    'last_name': user['last_name']
                }
            })
        
        return jsonify({'error': 'Invalid email or password'}), 401
        
    except Exception as e:
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    
    try:
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        allergies = list(allergies_collection.find({"user_id": user_id}))
        allergy_list = [{
            'name': allergy['allergen_name'], 
            'severity': allergy['severity'], 
            'notes': allergy.get('notes', '')
        } for allergy in allergies]
        
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
        allergies_collection.delete_many({"user_id": user_id})
        
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
    
    try:
        app.logger.info(f"[DEBUG] Starting analyze_food for user: {user_id}")
        
        if not models_pipeline:
            app.logger.error("[ERROR] Multi-model pipeline is empty")
            
            # Try to reload models
            reload_success = load_multi_model_pipeline()
            
            if not reload_success or not models_pipeline:
                return jsonify({
                    'error': 'Multi-model pipeline not available. Please try again in a few minutes.',
                    'details': 'Server is initializing AI models. This may take several minutes on first startup.',
                    'emergency_mode': True
                }), 503
        
        app.logger.info(f"[DEBUG] Models pipeline loaded: {len(models_pipeline)} models")
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        user_allergies = get_user_allergies(user_id)
        
        temp_filename = f"temp_{user_id}_{int(time.time())}.jpg"
        
        try:
            image_file.save(temp_filename)
            detected_ingredients = multi_model_predict(temp_filename, confidence_threshold=0.15)
            
        except Exception as e:
            app.logger.error(f"[ERROR] Model prediction failed: {str(e)}")
            return jsonify({
                'error': 'AI model processing failed',
                'details': str(e)
            }), 500
            
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        # Process allergen warnings
        allergen_warnings = []
        for ingredient in detected_ingredients:
            ingredient_name_lower = ingredient['name'].lower()
            
            for user_allergy in user_allergies:
                allergen_lower = user_allergy['allergen_name'].lower()
                
                paneer_variants = ['paneer', 'cottage cheese', 'indian cheese', 'fresh cheese']
                dairy_terms = ['dairy', 'milk', 'cheese', 'curd', 'butter']
                
                if allergen_lower == 'paneer' and any(variant in ingredient_name_lower for variant in paneer_variants):
                    allergen_warnings.append({
                        'allergen': user_allergy['allergen_name'],
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy['severity'],
                        'match_type': 'paneer_variant',
                        'bbox': ingredient.get('bbox'),
                        'detection_method': ingredient.get('detection_type', 'unknown')
                    })
                elif any(dairy_term in allergen_lower for dairy_term in dairy_terms) and 'paneer' in ingredient_name_lower:
                    allergen_warnings.append({
                        'allergen': user_allergy['allergen_name'],
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy['severity'],
                        'match_type': 'dairy_match',
                        'bbox': ingredient.get('bbox'),
                        'detection_method': ingredient.get('detection_type', 'unknown')
                    })
                elif (allergen_lower in ingredient_name_lower or 
                      ingredient_name_lower in allergen_lower):
                    allergen_warnings.append({
                        'allergen': user_allergy['allergen_name'],
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy['severity'],
                        'match_type': 'standard',
                        'bbox': ingredient.get('bbox'),
                        'detection_method': ingredient.get('detection_type', 'unknown')
                    })
        
        # Remove duplicates
        seen = set()
        unique_warnings = []
        for warning in allergen_warnings:
            key = (warning['allergen'], warning['ingredient'])
            if key not in seen:
                seen.add(key)
                unique_warnings.append(warning)
        
        allergen_warnings = unique_warnings
        avg_confidence = np.mean([ing['confidence'] for ing in detected_ingredients]) if detected_ingredients else 0.0
        yolo_detections = len([ing for ing in detected_ingredients if ing.get('detection_type') == 'object_detection'])
        
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
            'yolo_detections': yolo_detections,
            'total_detections': len(detected_ingredients),
            'message': 'Analysis completed successfully'
        })
        
    except Exception as e:
        app.logger.error(f"[ERROR] Analyze food failed: {str(e)}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/scan-history', methods=['GET'])
@jwt_required()
def get_scan_history():
    user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 50)
    
    skip = (page - 1) * per_page
    
    try:
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
    
    except Exception as e:
        return jsonify({'error': 'Failed to get scan history'}), 500

@app.route('/api/pipeline-status', methods=['GET'])
def pipeline_status():
    return jsonify({
        'models_loaded': len(models_pipeline),
        'models': [
            {
                'name': model_info['name'],
                'weight': model_info['weight'],
                'specialty': model_info['specialty'],
                'source': model_info.get('source', 'unknown'),
                'ingredients_count': len(model_info.get('ingredients', [])),
                'mlb_included': 'mlb' in model_info
            } for model_info in models_pipeline
        ],
        'device': str(device),
        'yolov8_available': any(model.get('specialty') == 'yolov8_paneer' for model in models_pipeline),
        'mlb_available': mlb_encoder is not None
    })

# Debug endpoints
@app.route('/api/debug/models-status')
def debug_models_status():
    """Check status of downloaded models"""
    try:
        model_status = {}
        
        for model_key, local_path in MODEL_PATHS.items():
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                model_status[model_key] = {
                    'downloaded': True,
                    'size_mb': round(file_size / 1024 / 1024, 2),
                    'path': local_path
                }
            else:
                model_status[model_key] = {
                    'downloaded': False,
                    'size_mb': 0,
                    'path': local_path
                }
        
        return jsonify({
            'models': model_status,
            'pipeline_loaded': len(models_pipeline),
            'mlb_encoder_loaded': mlb_encoder is not None,
            'total_models_available': len([m for m in model_status.values() if m['downloaded']]),
            'model_urls': MODEL_URLS
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reload-models', methods=['POST'])
def reload_models():
    """Manually reload models"""
    try:
        success = load_multi_model_pipeline()
        return jsonify({
            'success': success,
            'models_loaded': len(models_pipeline),
            'message': 'Model reload completed' if success else 'Model reload failed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health-detailed', methods=['GET'])
def health_detailed():
    """Detailed health check"""
    try:
        mongodb_status = True
        try:
            client.admin.command('ping')
        except:
            mongodb_status = False
        
        model_files_status = {}
        for model_key, path in MODEL_PATHS.items():
            model_files_status[model_key] = {
                'exists': os.path.exists(path),
                'size_mb': round(os.path.getsize(path) / 1024 / 1024, 2) if os.path.exists(path) else 0
            }
        
        overall_status = 'healthy' if (mongodb_status and len(models_pipeline) > 0) else 'degraded'
        
        return jsonify({
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'mongodb': {
                'connected': mongodb_status,
                'database': db.name
            },
            'models': {
                'total_loaded': len(models_pipeline),
                'yolov8_available': any(model.get('specialty') == 'yolov8_paneer' for model in models_pipeline),
                'mlb_available': mlb_encoder is not None
            },
            'model_files': model_files_status,
            'system': {
                'device': str(device),
                'pytorch_version': torch.__version__
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

def initialize_app():
    """Initialize app with model downloads"""
    try:
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        print("🤖 Initializing ML models from GitHub...")
        pipeline_loaded = load_multi_model_pipeline()
        
        if pipeline_loaded:
            print("🚀 FoodGuard API initialized successfully!")
            return True
        else:
            print("⚠️  Server started with warnings - limited functionality")
            return False
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

if __name__ == '__main__':
    print("🍽️ FoodGuard API Server Starting...")
    
    # Initialize the app
    init_success = initialize_app()
    
    if init_success:
        print("✅ Server initialization completed successfully")
    else:
        print("⚠️  Server started with warnings")
        print("💡 Visit /api/reload-models (POST) to retry model loading")
    
    print(f"🚀 Server ready at http://0.0.0.0:{int(os.environ.get('PORT', 5000))}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
