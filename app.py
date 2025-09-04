# app.py - Multi-Model Pipeline with YOLOv8 best.pt Integration
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision.models as torch_models
from ultralytics import YOLO  # Added for YOLOv8
from PIL import Image
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid
import os
import time

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JWT_SECRET_KEY'] = 'your-super-secret-jwt-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///foodguard.db')

# Handle PostgreSQL URL format for Render
db_url = app.config['SQLALCHEMY_DATABASE_URI']
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
jwt = JWTManager(app)
db = SQLAlchemy(app)

# Global variables for multi-model pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_pipeline = []

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
    """Load multiple models with YOLOv8 as primary paneer detector"""
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
        
        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    yolov8_detector = YOLOv8PaneerDetector(path, confidence_threshold=0.5)
                    
                    models_pipeline.append({
                        'name': 'yolov8_paneer_detector',
                        'model': yolov8_detector,
                        'ingredients': ['paneer', 'mint'],
                        'weight': 1.0,  # Primary model gets full weight
                        'specialty': 'yolov8_paneer'
                    })
                    
                    print(f"✅ YOLOv8 Paneer Detector loaded from: {path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"❌ Failed to load YOLOv8 from {path}: {e}")
                    continue
        
        if not model_loaded:
            print("⚠️  YOLOv8 model not found in any expected location")
            print("   Please ensure best.pt is in the correct directory")
        
        # Model 2: Your existing general food model (FALLBACK)
        if os.path.exists('food_detector.pth') and os.path.exists('pytorch_ingredients.pkl'):
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
                'weight': 0.3,  # Lower weight since YOLOv8 is primary
                'specialty': 'general'
            })
            print("✅ General food model loaded as fallback")
        
        print(f"🚀 Multi-model pipeline loaded with {len(models_pipeline)} models")
        return True
        
    except Exception as e:
        print(f"❌ Error loading multi-model pipeline: {e}")
        return False

def multi_model_predict(image, confidence_threshold=0.3):
    """Enhanced prediction using YOLOv8 + fallback models"""
    if not models_pipeline:
        raise Exception("Multi-model pipeline not loaded")
    
    combined_predictions = {}
    
    for model_info in models_pipeline:
        try:
            if model_info['specialty'] == 'yolov8_paneer':
                # YOLOv8 Object Detection
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
                # Your existing general food model
                model = model_info['model']
                ingredients = model_info['ingredients']
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
                
                detected_indices = np.where(predictions > confidence_threshold)[0]
                
                for idx in detected_indices:
                    if idx < len(ingredients):
                        ingredient = ingredients[idx]
                        confidence = float(predictions[idx]) * weight
                        
                        # Only add if not already detected by YOLOv8 or confidence is higher
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

# Enhanced analyze_food endpoint with YOLOv8
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
        # Load user's allergies
        user_allergies = UserAllergy.query.filter_by(user_id=user_id).all()
        
        # Save temporary image for YOLOv8 processing
        temp_filename = f"temp_{user_id}_{int(time.time())}.jpg"
        image_file.save(temp_filename)
        
        try:
            # Process with multi-model pipeline (YOLOv8 + fallback)
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
                'message': 'No ingredients detected. Try a clearer image.',
                'user_allergies_count': len(user_allergies)
            })
        
        # Enhanced allergen matching
        allergen_warnings = []
        for ingredient in detected_ingredients:
            ingredient_name_lower = ingredient['name'].lower()
            
            for user_allergy in user_allergies:
                allergen_lower = user_allergy.allergen_name.lower()
                
                # Enhanced matching for paneer/dairy
                paneer_variants = ['paneer', 'cottage cheese', 'indian cheese', 'fresh cheese']
                dairy_terms = ['dairy', 'milk', 'cheese', 'curd', 'butter']
                
                # Check for paneer variants
                if allergen_lower == 'paneer' and any(variant in ingredient_name_lower for variant in paneer_variants):
                    allergen_warnings.append({
                        'allergen': user_allergy.allergen_name,
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy.severity,
                        'match_type': 'paneer_variant',
                        'bbox': ingredient.get('bbox'),
                        'detection_method': ingredient.get('detection_type', 'unknown')
                    })
                    continue
                
                # Check for dairy terms
                if any(dairy_term in allergen_lower for dairy_term in dairy_terms) and 'paneer' in ingredient_name_lower:
                    allergen_warnings.append({
                        'allergen': user_allergy.allergen_name,
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy.severity,
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
                        'allergen': user_allergy.allergen_name,
                        'ingredient': ingredient['name'],
                        'confidence': ingredient['confidence'],
                        'severity': user_allergy.severity,
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
        
        # Save scan to history
        scan = ScanHistory(
            user_id=user_id,
            detected_ingredients=detected_ingredients,
            allergen_warnings=allergen_warnings,
            is_safe=len(allergen_warnings) == 0,
            confidence_score=float(avg_confidence)
        )
        db.session.add(scan)
        db.session.commit()
        
        return jsonify({
            'scan_id': scan.id,
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
        print(f"Multi-model analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# Pipeline status endpoint
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
        'yolov8_available': any(model['specialty'] == 'yolov8_paneer' for model in models_pipeline)
    })

# Database Models (same as before)
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    allergies = db.relationship('UserAllergy', backref='user', lazy=True, cascade='all, delete-orphan')
    scan_history = db.relationship('ScanHistory', backref='user', lazy=True, cascade='all, delete-orphan')

class UserAllergy(db.Model):
    __tablename__ = 'user_allergies'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    allergen_name = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(20), default='moderate')
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'allergen_name'),)

class ScanHistory(db.Model):
    __tablename__ = 'scan_history'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    image_filename = db.Column(db.String(255))
    detected_ingredients = db.Column(db.JSON)
    allergen_warnings = db.Column(db.JSON)
    is_safe = db.Column(db.Boolean)
    confidence_score = db.Column(db.Float)
    scan_location = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Authentication endpoints (add these if missing)
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    required_fields = ['email', 'password', 'first_name', 'last_name']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if len(data['password']) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    if User.query.filter_by(email=data['email'].lower()).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    try:
        user = User(
            email=data['email'].lower(),
            password_hash=generate_password_hash(data['password']),
            first_name=data['first_name'].strip(),
            last_name=data['last_name'].strip()
        )
        
        db.session.add(user)
        db.session.commit()
        
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'message': 'Account created successfully',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not all(k in data for k in ['email', 'password']):
        return jsonify({'error': 'Missing email or password'}), 400
    
    user = User.query.filter_by(email=data['email'].lower(), is_active=True).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({
            'access_token': access_token,
            'user': {
                'id': user.id,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name
            }
        })
    
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    user = db.session.get(User, user_id)  # Updated for SQLAlchemy 2.0
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    allergies = [{
        'name': allergy.allergen_name, 
        'severity': allergy.severity, 
        'notes': allergy.notes
    } for allergy in user.allergies]
    
    return jsonify({
        'user': {
            'id': user.id,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'created_at': user.created_at.isoformat()
        },
        'allergies': allergies,
        'total_scans': len(user.scan_history)
    })

@app.route('/api/profile/allergies', methods=['POST'])
@jwt_required()
def update_allergies():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if 'allergies' not in data:
        return jsonify({'error': 'Missing allergies data'}), 400
    
    try:
        UserAllergy.query.filter_by(user_id=user_id).delete()
        
        for allergy_data in data['allergies']:
            if 'name' not in allergy_data:
                continue
                
            allergy = UserAllergy(
                user_id=user_id,
                allergen_name=allergy_data['name'].lower().strip(),
                severity=allergy_data.get('severity', 'moderate'),
                notes=allergy_data.get('notes', '')
            )
            db.session.add(allergy)
        
        db.session.commit()
        return jsonify({'message': 'Allergies updated successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to update allergies'}), 500

@app.route('/api/scan-history', methods=['GET'])
@jwt_required()
def get_scan_history():
    user_id = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 10, type=int), 50)
    
    scans = ScanHistory.query.filter_by(user_id=user_id)\
        .order_by(ScanHistory.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'scans': [{
            'id': scan.id,
            'ingredients': scan.detected_ingredients or [],
            'warnings': scan.allergen_warnings or [],
            'is_safe': scan.is_safe,
            'confidence': scan.confidence_score,
            'created_at': scan.created_at.isoformat()
        } for scan in scans.items],
        'total': scans.total,
        'pages': scans.pages,
        'current_page': page,
        'has_next': scans.has_next,
        'has_prev': scans.has_prev
    })

def initialize_app():
    """Initialize database tables and load YOLOv8 multi-model pipeline"""
    with app.app_context():
        db.create_all()
        pipeline_loaded = load_multi_model_pipeline()
        
        if pipeline_loaded:
            print("🚀 YOLOv8-Enhanced FoodGuard API Server initialized successfully!")
        else:
            print("⚠️  Server started but YOLOv8 model may not be loaded.")

if __name__ == '__main__':
    initialize_app()
    
    print("🍽️ YOLOv8-Enhanced FoodGuard API Server Starting...")
    print("📝 Required model files:")
    print("   - food_detectors/paneer_mint_yolov8/weights/best.pt (YOLOv8 model)")
    print("   - food_detector.pth (fallback general model - optional)")
    print("   - pytorch_ingredients.pkl (fallback ingredients - optional)")
    print()
    print("🎯 YOLOv8 Model Performance: mAP50 = 76.8%")
    print("🔍 Detection capabilities: Paneer + Mint with bounding boxes")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
