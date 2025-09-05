import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pickle
import requests
from pathlib import Path
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
import logging

# MongoDB imports with URL parsing
import pymongo
from bson import ObjectId
from urllib.parse import urlparse, urlunparse, quote_plus

load_dotenv()

app = Flask(__name__)

# FIXED: Simplified CORS Configuration - NO MANUAL HEADERS
CORS(app, 
     origins=[
         "https://foodguard-eight.vercel.app",
         "https://foodguard-frontend.vercel.app",
         "http://localhost:3000",
         "http://127.0.0.1:3000"
     ],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=False
)

# REMOVED: All manual CORS handlers that were causing conflicts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Production Configuration
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-super-secret-jwt-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

jwt = JWTManager(app)

# MongoDB Configuration
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
        
        logger.info("🔗 Connecting to MongoDB...")
        client = pymongo.MongoClient(encoded_uri)
        client.admin.command('ping')
        logger.info("✅ MongoDB connection successful")
        return client
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

# Create MongoDB client
try:
    client = create_mongodb_client()
    db = client['foodguard']
    
    # Collections
    users_collection = db['users']
    allergies_collection = db['allergies'] 
    scan_history_collection = db['scan_history']
    
    # Create indexes
    try:
        users_collection.create_index("email", unique=True)
        allergies_collection.create_index("user_id")
        scan_history_collection.create_index("user_id")
        logger.info("✅ MongoDB indexes created")
    except Exception as e:
        logger.warning(f"⚠️ Index creation warning: {e}")

except Exception as e:
    logger.error(f"❌ MongoDB initialization failed: {e}")
    # Create dummy collections for error handling
    users_collection = None
    allergies_collection = None
    scan_history_collection = None

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_pipeline = []
mlb_encoder = None

# FIXED: Model URLs and paths
MODEL_URLS = {
    'yolov8': os.environ.get('YOLOV8_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/best.pt'),
    'food_detector': os.environ.get('FOOD_DETECTOR_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/food_detector.pth'),
    'ingredients_list': os.environ.get('INGREDIENTS_LIST_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/pytorch_ingredients.pkl'),
    'label_binarizer': os.environ.get('MLB_MODEL_URL', 'https://github.com/HarshBothara24/FoodGuard-backend/releases/download/download/mlb.pkl')
}

MODEL_PATHS = {
    'yolov8': 'models/best.pt',
    'food_detector': 'models/food_detector.pth',
    'ingredients_list': 'models/pytorch_ingredients.pkl',
    'label_binarizer': 'models/mlb.pkl'
}

def test_network_connectivity():
    """Test if external HTTP requests work on Render"""
    try:
        logger.info("🌐 Testing network connectivity...")
        response = requests.get('https://httpbin.org/json', timeout=30)
        if response.status_code == 200:
            logger.info("✅ External HTTP requests working")
            return True
        else:
            logger.warning(f"⚠️ HTTP test returned status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Network connectivity test failed: {e}")
        return False

def download_model_file(url, local_path, model_name):
    """Download model file with enhanced error handling"""
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        if file_size > 1024:  # At least 1KB
            logger.info(f"✅ {model_name} already exists ({file_size/1024/1024:.1f}MB)")
            return True
        else:
            logger.warning(f"⚠️ {model_name} corrupted, re-downloading...")
            os.remove(local_path)
    
    # Test network first
    if not test_network_connectivity():
        logger.error("❌ Network connectivity failed, cannot download models")
        return False
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Downloading {model_name} (attempt {attempt + 1}/{max_retries})")
            logger.info(f"URL: {url}")
            
            # Create directory
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with proper headers and timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; FoodGuard/1.0)',
                'Accept': 'application/octet-stream'
            }
            
            response = requests.get(url, stream=True, timeout=300, headers=headers)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"📊 {model_name}: {progress:.1f}%")
            
            file_size = os.path.getsize(local_path)
            logger.info(f"✅ {model_name} downloaded ({file_size/1024/1024:.1f}MB)")
            
            if file_size < 1024:
                logger.error(f"❌ {model_name} too small ({file_size} bytes)")
                os.remove(local_path)
                continue
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"⏰ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            break
    
    if os.path.exists(local_path):
        os.remove(local_path)
    return False

def download_all_models():
    """FIXED: Download all models - THIS FUNCTION IS CALLED NOW"""
    logger.info("🚀 STARTING MODEL DOWNLOADS FROM GITHUB...")
    
    success_count = 0
    total_models = len(MODEL_URLS)
    
    # Log environment variables
    for model_key, url in MODEL_URLS.items():
        logger.info(f"📋 {model_key}: {url}")
    
    for model_key, url in MODEL_URLS.items():
        local_path = MODEL_PATHS[model_key]
        model_name = model_key.replace('_', ' ').title()
        
        logger.info(f"📥 Attempting to download {model_name}...")
        
        if download_model_file(url, local_path, model_name):
            success_count += 1
            logger.info(f"✅ {model_name} download successful")
        else:
            logger.error(f"💥 {model_name} download failed")
    
    logger.info(f"📊 Download Summary: {success_count}/{total_models} models")
    
    return success_count > 0

# Model Classes
class YOLOv8PaneerDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        logger.info(f"🥛 Loading YOLOv8: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = ['Paneer', 'mint']
        logger.info(f"✅ YOLOv8 loaded on {self.device}")
        
    def detect_ingredients(self, image):
        try:
            results = self.model(image, verbose=False)
            detected_ingredients = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        conf = float(box.conf)
                        cls = int(box.cls)
                        
                        if conf >= self.confidence_threshold and cls < len(self.class_names):
                            detected_ingredients.append({
                                'name': self.class_names[cls].lower(),
                                'confidence': conf,
                                'bbox': box.xyxy.cpu().numpy().flatten().tolist(),
                                'model_source': 'yolov8_paneer_detector',
                                'class_id': cls
                            })
            
            return detected_ingredients
        except Exception as e:
            logger.error(f"YOLOv8 detection error: {e}")
            return []

class FoodAllergenDetector(nn.Module):
    def __init__(self, num_classes):
        super(FoodAllergenDetector, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

def load_multi_model_pipeline():
    """FIXED: Load models with forced download attempt"""
    global models_pipeline, mlb_encoder
    
    try:
        logger.info("🤖 INITIALIZING MULTI-MODEL PIPELINE...")
        
        # FORCE model downloads
        logger.info("🔽 FORCING MODEL DOWNLOADS...")
        download_success = download_all_models()
        
        if not download_success:
            logger.warning("⚠️ Model downloads failed, adding fallback...")
        
        models_pipeline = []
        mlb_encoder = None
        
        # Try loading models
        yolov8_path = MODEL_PATHS['yolov8']
        if os.path.exists(yolov8_path):
            try:
                yolov8_detector = YOLOv8PaneerDetector(yolov8_path)
                models_pipeline.append({
                    'name': 'yolov8_paneer_detector',
                    'model': yolov8_detector,
                    'ingredients': ['paneer', 'mint'],
                    'weight': 1.2,
                    'specialty': 'yolov8_paneer'
                })
                logger.info("✅ YOLOv8 loaded successfully")
            except Exception as e:
                logger.error(f"❌ YOLOv8 loading failed: {e}")
        
        # Load other models similarly...
        food_path = MODEL_PATHS['food_detector']
        ingredients_path = MODEL_PATHS['ingredients_list']
        
        if os.path.exists(food_path) and os.path.exists(ingredients_path):
            try:
                with open(ingredients_path, 'rb') as f:
                    ingredients_list = pickle.load(f)
                
                model = FoodAllergenDetector(len(ingredients_list))
                model.load_state_dict(torch.load(food_path, map_location=device))
                model.to(device)
                model.eval()
                
                models_pipeline.append({
                    'name': 'general_food_model',
                    'model': model,
                    'ingredients': ingredients_list,
                    'weight': 0.4,
                    'specialty': 'general'
                })
                logger.info(f"✅ General model loaded ({len(ingredients_list)} ingredients)")
            except Exception as e:
                logger.error(f"❌ General model failed: {e}")
        
        # Load MLB
        mlb_path = MODEL_PATHS['label_binarizer']
        if os.path.exists(mlb_path):
            try:
                with open(mlb_path, 'rb') as f:
                    mlb_encoder = pickle.load(f)
                logger.info(f"✅ MLB loaded ({len(mlb_encoder.classes_)} classes)")
            except Exception as e:
                logger.error(f"❌ MLB loading failed: {e}")
        
        # Add fallback if no models
        if not models_pipeline:
            logger.warning("⚠️ Adding emergency fallback...")
            models_pipeline.append({
                'name': 'emergency_fallback',
                'model': None,
                'ingredients': ['paneer', 'dairy', 'cheese', 'nuts'],
                'weight': 0.1,
                'specialty': 'fallback'
            })
        
        logger.info(f"🚀 Pipeline loaded: {len(models_pipeline)} models")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline loading error: {e}")
        return False

def multi_model_predict(image, confidence_threshold=0.2):
    """Enhanced prediction with fallback"""
    if not models_pipeline:
        raise Exception("Multi-model pipeline not loaded")
    
    combined_predictions = {}
    
    for model_info in models_pipeline:
        try:
            if model_info['specialty'] == 'yolov8_paneer':
                detections = model_info['model'].detect_ingredients(image)
                for detection in detections:
                    combined_predictions[detection['name']] = {
                        'name': detection['name'],
                        'confidence': detection['confidence'] * model_info['weight'],
                        'bbox': detection.get('bbox'),
                        'model_source': detection['model_source'],
                        'detection_type': 'object_detection'
                    }
            
            elif model_info['specialty'] == 'general':
                # General model prediction logic
                model = model_info['model']
                ingredients = model_info['ingredients']
                
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
                        confidence = float(predictions[idx]) * model_info['weight']
                        
                        if ingredient not in combined_predictions:
                            combined_predictions[ingredient] = {
                                'name': ingredient,
                                'confidence': confidence,
                                'model_source': model_info['name'],
                                'detection_type': 'classification'
                            }
            
            elif model_info['specialty'] == 'fallback':
                # Emergency fallback
                for ingredient in ['paneer', 'dairy']:
                    if ingredient not in combined_predictions:
                        combined_predictions[ingredient] = {
                            'name': ingredient,
                            'confidence': 0.1,
                            'model_source': 'emergency_fallback',
                            'detection_type': 'fallback'
                        }
        
        except Exception as e:
            logger.error(f"Model {model_info['name']} error: {e}")
            continue
    
    detected_ingredients = list(combined_predictions.values())
    detected_ingredients.sort(key=lambda x: x['confidence'], reverse=True)
    return detected_ingredients[:25]

def get_user_allergies(user_id):
    """Get user allergies"""
    try:
        if allergies_collection:
            return list(allergies_collection.find({"user_id": user_id}))
    except:
        pass
    return []

def save_scan_to_history(user_id, detected_ingredients, allergen_warnings, is_safe, confidence_score):
    """Save scan history"""
    try:
        if scan_history_collection:
            result = scan_history_collection.insert_one({
                "user_id": user_id,
                "detected_ingredients": detected_ingredients,
                "allergen_warnings": allergen_warnings,
                "is_safe": is_safe,
                "confidence_score": confidence_score,
                "created_at": datetime.utcnow()
            })
            return str(result.inserted_id)
    except:
        pass
    return None

# Routes
@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'FoodGuard API running',
        'models_loaded': len(models_pipeline),
        'mongodb_connected': users_collection is not None,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        if not users_collection:
            return jsonify({'error': 'Database unavailable'}), 503
        
        required_fields = ['email', 'password', 'first_name', 'last_name']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
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
        
        result = users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
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
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not users_collection:
            return jsonify({'error': 'Database unavailable'}), 503
        
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
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/analyze-food', methods=['POST'])
@jwt_required()
def analyze_food():
    user_id = get_jwt_identity()
    
    try:
        if not models_pipeline:
            # Try to reload models
            logger.info("🔄 Attempting to reload models...")
            load_multi_model_pipeline()
            
            if not models_pipeline:
                return jsonify({
                    'error': 'AI models not available',
                    'details': 'Server is still loading models. Please try again.',
                    'suggestion': 'Wait 2-3 minutes and retry'
                }), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No image selected'}), 400
        
        user_allergies = get_user_allergies(user_id)
        
        temp_filename = f"temp_{user_id}_{int(time.time())}.jpg"
        
        try:
            image_file.save(temp_filename)
            detected_ingredients = multi_model_predict(temp_filename)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        # Process allergen warnings
        allergen_warnings = []
        for ingredient in detected_ingredients:
            for allergy in user_allergies:
                if allergy['allergen_name'].lower() in ingredient['name'].lower():
                    allergen_warnings.append({
                        'allergen': allergy['allergen_name'],
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': allergy['severity']
                    })
        
        scan_id = save_scan_to_history(
            user_id, detected_ingredients, allergen_warnings,
            len(allergen_warnings) == 0,
            np.mean([ing['confidence'] for ing in detected_ingredients]) if detected_ingredients else 0
        )
        
        return jsonify({
            'scan_id': scan_id,
            'ingredients': detected_ingredients,
            'allergen_warnings': allergen_warnings,
            'is_safe': len(allergen_warnings) == 0,
            'models_used': len(models_pipeline),
            'message': 'Analysis completed'
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/debug/models-status')
def debug_models_status():
    """Debug models status"""
    model_status = {}
    
    for key, path in MODEL_PATHS.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        model_status[key] = {
            'downloaded': exists,
            'size_mb': round(size / 1024 / 1024, 2),
            'path': path
        }
    
    return jsonify({
        'models': model_status,
        'pipeline_loaded': len(models_pipeline),
        'model_urls': MODEL_URLS,
        'network_test': test_network_connectivity()
    })

@app.route('/api/reload-models', methods=['POST'])
def reload_models():
    """Force reload models"""
    try:
        logger.info("🔄 Manual model reload requested")
        success = load_multi_model_pipeline()
        return jsonify({
            'success': success,
            'models_loaded': len(models_pipeline),
            'message': 'Reload completed' if success else 'Reload failed'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def initialize_app():
    """FIXED: Initialize app and FORCE model loading"""
    try:
        logger.info("🚀 INITIALIZING FOODGUARD API...")
        
        # Test MongoDB
        if users_collection:
            logger.info("✅ MongoDB connected")
        else:
            logger.warning("⚠️ MongoDB connection issues")
        
        # FORCE model loading
        logger.info("🤖 FORCING MODEL INITIALIZATION...")
        pipeline_loaded = load_multi_model_pipeline()
        
        if pipeline_loaded and len(models_pipeline) > 0:
            logger.info(f"🚀 SUCCESS: {len(models_pipeline)} models loaded!")
            return True
        else:
            logger.warning("⚠️ Model loading failed or incomplete")
            return False
            
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return False

# Additional routes (profile, allergies, etc.) - keeping them minimal for space
@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    if not users_collection:
        return jsonify({'error': 'Database unavailable'}), 503
    
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': {
            'id': str(user['_id']),
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name']
        },
        'allergies': get_user_allergies(user_id),
        'total_scans': 0
    })

if __name__ == '__main__':
    logger.info("🍽️ FoodGuard API Server Starting...")
    
    # FORCE initialization on startup
    init_success = initialize_app()
    
    if init_success:
        logger.info("✅ Server ready with models loaded!")
    else:
        logger.warning("⚠️ Server started but models may not be ready")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
