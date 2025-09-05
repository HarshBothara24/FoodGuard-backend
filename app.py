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

# Enhanced CORS Configuration for Production
CORS_ORIGINS = [
    "https://foodguard-eight.vercel.app",  # Your Vercel deployment
    "https://foodguard-frontend.vercel.app",  # Alternative Vercel URLs
    "http://localhost:3000",  # Local development
    "http://127.0.0.1:3000",  # Alternative local
    "https://localhost:3000"  # HTTPS local
]

# Temporary CORS fix for testing (use specific origins in production)
CORS(app, 
     origins=["*"],  # Allow all origins temporarily
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False  # Set to False when using origins=["*"]
)

# Add explicit OPTIONS handler for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

# Add response headers for all requests
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in CORS_ORIGINS:
        response.headers.add('Access-Control-Allow-Origin', origin)
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Production Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-super-secret-jwt-key')

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
mlb_encoder = None  # Global MLB encoder

# Model URLs Configuration - ALL 4 FILES
MODEL_URLS = {
    'yolov8': os.environ.get('YOLOV8_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/best.pt'),
    'food_detector': os.environ.get('FOOD_DETECTOR_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/food_detector.pth'),
    'ingredients_list': os.environ.get('INGREDIENTS_LIST_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/pytorch_ingredients.pkl'),
    'label_binarizer': os.environ.get('MLB_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/mlb.pkl')
}

# Local paths where all files will be saved
MODEL_PATHS = {
    'yolov8': 'best.pt',
    'food_detector': 'food_detector.pth', 
    'ingredients_list': 'pytorch_ingredients.pkl',
    'label_binarizer': 'mlb.pkl'
}

def download_model_file(url, local_path, model_name):
    """Download a single model file from GitHub with progress tracking"""
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        print(f"✅ {model_name} already exists locally ({file_size/1024/1024:.1f}MB)")
        return True
    
    try:
        print(f"📥 Downloading {model_name} from GitHub...")
        print(f"URL: {url}")
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress tracking
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"📊 {model_name} progress: {progress:.1f}%", end='\r')
        
        file_size = os.path.getsize(local_path)
        print(f"\n✅ {model_name} downloaded successfully ({file_size/1024/1024:.1f}MB)")
        
        # Basic integrity check
        if file_size < 1024:  # Less than 1KB suggests error
            print(f"❌ {model_name} file seems corrupted (too small)")
            os.remove(local_path)
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
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
    
    if success_count == total_models:
        print("🎉 All models downloaded successfully!")
        return True
    elif success_count > 0:
        print("⚠️  Some models downloaded, continuing with available models...")
        return True
    else:
        print("❌ No models downloaded successfully!")
        return False

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

# Your existing model classes (for fallback)
class FoodAllergenDetector(nn.Module):
    def __init__(self, num_classes):
        super(FoodAllergenDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def load_multi_model_pipeline():
    """Load all models including MLB label decoder after downloading from GitHub"""
    global models_pipeline, mlb_encoder
    
    try:
        # First, download all models from GitHub
        if not download_all_models():
            print("💥 Model download failed, but continuing with any available models...")
        
        models_pipeline = []
        mlb_encoder = None
        
        # Load MLB Label Binarizer (CRITICAL for decoding)
        mlb_path = MODEL_PATHS['label_binarizer']
        if os.path.exists(mlb_path):
            try:
                print(f"🏷️ Loading label binarizer from: {mlb_path}")
                with open(mlb_path, 'rb') as f:
                    mlb_encoder = pickle.load(f)
                print(f"✅ Label binarizer loaded with {len(mlb_encoder.classes_)} classes")
            except Exception as e:
                print(f"❌ Failed to load MLB: {e}")
                print("⚠️  Continuing without MLB decoder...")
        
        # Model 1: YOLOv8 Paneer Detector (PRIMARY)
        yolov8_path = MODEL_PATHS['yolov8']
        
        if os.path.exists(yolov8_path):
            try:
                print(f"🥛 Loading YOLOv8 model from: {yolov8_path}")
                yolov8_detector = YOLOv8PaneerDetector(yolov8_path, confidence_threshold=0.5)
                
                # Test the model with a dummy inference
                print("🧪 Testing YOLOv8 model...")
                dummy_img = Image.new('RGB', (640, 640), color='white')
                test_results = yolov8_detector.detect_ingredients(dummy_img)
                print(f"✅ YOLOv8 test successful: {len(test_results)} detections")
                
                models_pipeline.append({
                    'name': 'yolov8_paneer_detector',
                    'model': yolov8_detector,
                    'ingredients': ['paneer', 'mint'],
                    'weight': 1.0,
                    'specialty': 'yolov8_paneer',
                    'source': 'github'
                })
                
                print(f"✅ YOLOv8 Paneer Detector loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load YOLOv8 model: {e}")
        
        # Model 2: General Food Detector (FALLBACK)
        food_detector_path = MODEL_PATHS['food_detector']
        ingredients_path = MODEL_PATHS['ingredients_list']
        
        if os.path.exists(food_detector_path) and os.path.exists(ingredients_path):
            try:
                print(f"🍽️ Loading general food model from: {food_detector_path}")
                
                # Load ingredients list
                with open(ingredients_path, 'rb') as f:
                    ingredients_list = pickle.load(f)
                
                # Load PyTorch model
                model = FoodAllergenDetector(len(ingredients_list))
                model.load_state_dict(torch.load(food_detector_path, map_location=device))
                model.to(device)
                model.eval()
                
                models_pipeline.append({
                    'name': 'general_food_model',
                    'model': model,
                    'ingredients': ingredients_list,
                    'weight': 0.3,
                    'specialty': 'general',
                    'source': 'github',
                    'mlb': mlb_encoder  # Include MLB for this model
                })
                
                print(f"✅ General food model loaded successfully ({len(ingredients_list)} ingredients)")
                
            except Exception as e:
                print(f"❌ Failed to load general food model: {e}")
        
        if not models_pipeline:
            print("💥 No models loaded successfully!")
            return False
        
        print(f"🚀 Multi-model pipeline loaded with {len(models_pipeline)} models from GitHub")
        
        # Display final status
        yolov8_available = any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline)
        general_available = any(model['specialty'] == 'general' for model in models_pipeline)
        
        print(f"📊 Final Model Status:")
        print(f"   - YOLOv8 Paneer Detector: {'✅ Ready' if yolov8_available else '❌ Not loaded'}")
        print(f"   - General Food Model: {'✅ Ready' if general_available else '❌ Not loaded'}")
        print(f"   - MLB Label Decoder: {'✅ Ready' if mlb_encoder else '❌ Not loaded'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in multi-model pipeline loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def multi_model_predict(image, confidence_threshold=0.3):
    """Enhanced prediction with MLB label decoding"""
    if not models_pipeline:
        raise Exception("Multi-model pipeline not loaded")
    
    combined_predictions = {}
    
    for model_info in models_pipeline:
        try:
            if model_info['specialty'] == 'yolov8_paneer':
                # YOLOv8 Object Detection (no MLB needed)
                yolo_detector = model_info['model']
                detections = yolo_detector.detect_ingredients(image)
                weight = model_info['weight']
                
                for detection in detections:
                    ingredient = detection['name']
                    confidence = detection['confidence'] * weight
                    
                    # Boost confidence for paneer (main allergen concern)
                    if 'paneer' in ingredient.lower():
                        confidence *= 1.2  # 20% boost for paneer detection
                    
                    # Keep highest confidence and add detection metadata
                    if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence']:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': confidence,
                            'bbox': detection.get('bbox'),
                            'model_source': detection['model_source'],
                            'detection_type': 'object_detection'
                        }
            
            elif model_info['specialty'] == 'general':
                # General Food Model with optional MLB decoding
                model = model_info['model']
                ingredients = model_info['ingredients']
                mlb = model_info.get('mlb')  # Get MLB decoder
                weight = model_info['weight']
                
                # Preprocess image for your existing model
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
                
                # Use MLB to decode multi-label predictions if available
                if mlb is not None:
                    try:
                        # Convert predictions to binary labels
                        binary_predictions = (predictions > confidence_threshold).astype(int)
                        
                        # Only decode if there are any positive predictions
                        if binary_predictions.sum() > 0:
                            # Decode using MLB
                            decoded_labels = mlb.inverse_transform([binary_predictions])
                            
                            for label in decoded_labels[0]:
                                if label in ingredients:
                                    idx = ingredients.index(label)
                                    confidence = float(predictions[idx]) * weight
                                    
                                    combined_predictions[label] = {
                                        'name': label,
                                        'confidence': confidence,
                                        'model_source': 'general_food_model_mlb',
                                        'detection_type': 'multilabel_classification'
                                    }
                    except Exception as e:
                        print(f"MLB decoding error: {e}")
                        # Fallback to standard approach
                        pass
                
                # Standard approach (with or without MLB failure)
                detected_indices = np.where(predictions > confidence_threshold)[0]
                for idx in detected_indices:
                    if idx < len(ingredients):
                        ingredient = ingredients[idx]
                        confidence = float(predictions[idx]) * weight
                        
                        # Only add if not already detected by YOLOv8 or MLB, or confidence is higher
                        if ingredient not in combined_predictions or confidence > combined_predictions[ingredient]['confidence']:
                            combined_predictions[ingredient] = {
                                'name': ingredient,
                                'confidence': confidence,
                                'model_source': model_info['name'],
                                'detection_type': 'classification'
                            }
                            
        except Exception as e:
            print(f"Error in model {model_info['name']}: {e}")
            continue
    
    # Convert to list and sort by confidence
    detected_ingredients = list(combined_predictions.values())
    detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)
    
    return detected_ingredients[:25]  # Top 25 ingredients

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

# Error handlers
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error occurred'}), 500

@app.errorhandler(415)
def unsupported_media_type(e):
    return jsonify({
        'error': 'Unsupported Media Type',
        'message': 'Content-Type must be application/json',
        'received_content_type': request.content_type
    }), 415

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"500 error: {str(e)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

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
            'model_urls': {k: v for k, v in MODEL_URLS.items()}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/db-info')
def debug_db_info():
    return jsonify({
        'database_name': db.name,
        'collections': db.list_collection_names(),
        'users_collection_name': users_collection.name,
        'server_info': client.server_info()['version']
    })

@app.route('/api/test-models', methods=['GET'])
def test_models():
    """Test model loading and basic functionality"""
    try:
        models_status = {
            'models_loaded': len(models_pipeline),
            'device': str(device),
            'models': []
        }
        
        for model_info in models_pipeline:
            models_status['models'].append({
                'name': model_info['name'],
                'specialty': model_info['specialty'],
                'ingredients_count': len(model_info.get('ingredients', []))
            })
        
        return jsonify(models_status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        
        print(f"💾 Attempting to save user document: {user_doc['email']}")
        
        # Insert user into MongoDB
        try:
            result = users_collection.insert_one(user_doc)
            user_id = str(result.inserted_id)
            print(f"✅ User inserted with ID: {user_id}")
            
            # Verify user was actually saved
            saved_user = users_collection.find_one({"_id": result.inserted_id})
            if not saved_user:
                print(f"❌ User verification failed for {email}")
                return jsonify({'error': 'User creation verification failed'}), 500
            
            print(f"✅ User verification successful: {saved_user['email']}")
            
        except pymongo.errors.DuplicateKeyError:
            print(f"❌ Duplicate key error for {email}")
            return jsonify({'error': 'Email already registered'}), 409
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
        import traceback
        traceback.print_exc()
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
        print(f"❌ Profile fetch error: {str(e)}")
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
        print(f"❌ Allergies update error: {str(e)}")
        return jsonify({'error': 'Failed to update allergies'}), 500

@app.route('/api/analyze-food', methods=['POST'])
@jwt_required()
def analyze_food():
    user_id = get_jwt_identity()
    
    try:
        app.logger.info(f"[DEBUG] Starting analyze_food for user: {user_id}")
        
        if not models_pipeline:
            app.logger.error("[ERROR] Multi-model pipeline not available")
            return jsonify({'error': 'Multi-model pipeline not available. Please check server logs.'}), 500
        
        app.logger.info(f"[DEBUG] Models pipeline loaded: {len(models_pipeline)} models")
        
        if 'image' not in request.files:
            app.logger.error("[ERROR] No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            app.logger.error("[ERROR] No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        app.logger.info(f"[DEBUG] Image received: {image_file.filename}")
        
        # Load user allergies
        app.logger.info(f"[DEBUG] Loading allergies for user: {user_id}")
        user_allergies = get_user_allergies(user_id)
        app.logger.info(f"[DEBUG] Found {len(user_allergies)} allergies")
        
        # Save temporary image
        temp_filename = f"temp_{user_id}_{int(time.time())}.jpg"
        app.logger.info(f"[DEBUG] Saving temp file: {temp_filename}")
        
        try:
            image_file.save(temp_filename)
            app.logger.info(f"[DEBUG] Temp file saved successfully")
            
            if not os.path.exists(temp_filename):
                app.logger.error(f"[ERROR] Temp file not created: {temp_filename}")
                return jsonify({'error': 'Failed to save image'}), 500
            
            file_size = os.path.getsize(temp_filename)
            app.logger.info(f"[DEBUG] Temp file size: {file_size} bytes")
            
            # Process with models
            app.logger.info("[DEBUG] Starting model prediction")
            detected_ingredients = multi_model_predict(temp_filename)
            app.logger.info(f"[DEBUG] Model prediction completed: {len(detected_ingredients)} ingredients detected")
            
        except Exception as e:
            app.logger.error(f"[ERROR] Model prediction failed: {str(e)}", exc_info=True)
            return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500
            
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    app.logger.info(f"[DEBUG] Temp file cleaned up: {temp_filename}")
            except Exception as e:
                app.logger.warning(f"[WARNING] Failed to clean up temp file: {e}")
        
        if not detected_ingredients:
            return jsonify({
                'scan_id': None,
                'ingredients': [],
                'allergen_warnings': [],
                'is_safe': True,
                'message': 'No ingredients detected. Try a clearer image.',
                'user_allergies_count': len(user_allergies)
            })
        
        # Enhanced allergen matching
        app.logger.info("[DEBUG] Processing allergen warnings")
        allergen_warnings = []
        for ingredient in detected_ingredients:
            ingredient_name_lower = ingredient['name'].lower()
            
            for user_allergy in user_allergies:
                allergen_lower = user_allergy['allergen_name'].lower()
                
                # Enhanced matching for paneer/dairy
                paneer_variants = ['paneer', 'cottage cheese', 'indian cheese', 'fresh cheese']
                dairy_terms = ['dairy', 'milk', 'cheese', 'curd', 'butter']
                
                # Check for paneer variants
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
                    continue
                
                # Check for dairy terms
                if any(dairy_term in allergen_lower for dairy_term in dairy_terms) and 'paneer' in ingredient_name_lower:
                    allergen_warnings.append({
                        'allergen': user_allergy['allergen_name'],
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy['severity'],
                        'match_type': 'dairy_match',
                        'bbox': ingredient.get('bbox'),
                        'detection_method': ingredient.get('detection_type', 'unknown')
                    })
                    continue
                
                # Standard matching
                if (allergen_lower in ingredient_name_lower or 
                    ingredient_name_lower in allergen_lower or
                    any(word in ingredient_name_lower.split() for word in allergen_lower.split())):
                    
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
        
        # Count YOLOv8 detections
        yolo_detections = len([ing for ing in detected_ingredients if ing.get('detection_type') == 'object_detection'])
        
        # Save scan to MongoDB history
        scan_id = save_scan_to_history(
            user_id=user_id,
            detected_ingredients=detected_ingredients,
            allergen_warnings=allergen_warnings,
            is_safe=len(allergen_warnings) == 0,
            confidence_score=float(avg_confidence)
        )
        
        app.logger.info("[DEBUG] Analysis completed successfully")
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
            'message': 'YOLOv8 enhanced multi-model analysis completed'
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
    
    except Exception as e:
        print(f"❌ Scan history error: {str(e)}")
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
                'ingredients_count': len(model_info['ingredients']) if 'ingredients' in model_info else 0
            } for model_info in models_pipeline
        ],
        'device': str(device),
        'yolov8_available': any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline),
        'mlb_available': mlb_encoder is not None
    })

def initialize_app():
    """Initialize app with model downloads from GitHub and MongoDB"""
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        
        # Download and load models from GitHub
        print("🤖 Initializing ML models from GitHub...")
        pipeline_loaded = load_multi_model_pipeline()
        
        if pipeline_loaded:
            print("🚀 YOLOv8-Enhanced FoodGuard API with GitHub models initialized successfully!")
        else:
            print("⚠️  Server started but model loading failed. API will have limited functionality.")
            
    except Exception as e:
        print(f"❌ Initialization error: {e}")

if __name__ == '__main__':
    initialize_app()
    
    print("🍽️ GitHub-Enhanced FoodGuard API Server Starting...")
    print("📝 Model files will be downloaded from GitHub:")
    print("   - best.pt (YOLOv8 model)")
    print("   - food_detector.pth (General food model)")
    print("   - pytorch_ingredients.pkl (Ingredients list)")
    print("   - mlb.pkl (Label binarizer)")
    print()
    print("🎯 YOLOv8 Model Performance: mAP50 = 76.8%")
    print("🔍 Detection capabilities: Paneer + Mint with bounding boxes + Multi-label classification")
    print("🍃 Database: MongoDB with collections: users, allergies, scan_history")
    print("🚀 Ready for production deployment on Render!")
    print()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
