import os
import pandas as pd
import shutil
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

# Import untuk berbagai metode
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess

# ===== STEP 1: Setup Dataset =====
df = pd.read_csv("data.csv")
print(f"Total samples: {len(df)}")
print(f"Classes: {df['name'].unique()}")

# ===== STEP 2: Feature Extraction Methods =====

class FeatureExtractor:
    def __init__(self, method='vgg16'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        
        if method == 'vgg16':
            self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        elif method == 'resnet50':
            self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        elif method == 'mobilenet':
            self.model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    def extract_deep_features(self, img_path):
        """Extract features using pre-trained CNN"""
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            if self.method == 'vgg16':
                img_array = vgg_preprocess(img_array)
            elif self.method == 'resnet50':
                img_array = resnet_preprocess(img_array)
            elif self.method == 'mobilenet':
                img_array = mobile_preprocess(img_array)
            
            features = self.model.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None
    
    def extract_traditional_features(self, img_path):
        """Extract traditional computer vision features using PIL"""
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            features = []
            
            # 1. Color Histogram untuk setiap channel RGB
            for channel in range(3):
                hist, _ = np.histogram(img_array[:,:,channel], bins=32, range=(0, 256))
                features.extend(hist.flatten())
            
            # 2. Grayscale histogram
            gray_img = img.convert('L')
            gray_array = np.array(gray_img)
            gray_hist, _ = np.histogram(gray_array, bins=32, range=(0, 256))
            features.extend(gray_hist.flatten())
            
            # 3. Edge detection menggunakan PIL
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)
            edge_density = np.sum(edges_array > 50) / (edges_array.shape[0] * edges_array.shape[1])
            features.append(edge_density)
            
            # 4. Texture features - simplified LBP using PIL
            # Apply different filters untuk texture
            emboss = img.filter(ImageFilter.EMBOSS)
            emboss_gray = emboss.convert('L')
            emboss_array = np.array(emboss_gray)
            emboss_hist, _ = np.histogram(emboss_array, bins=16, range=(0, 256))
            features.extend(emboss_hist.flatten())
            
            # 5. Statistical features
            # Mean RGB values
            features.extend([np.mean(img_array[:,:,i]) for i in range(3)])
            
            # Standard deviation RGB
            features.extend([np.std(img_array[:,:,i]) for i in range(3)])
            
            # 6. Brightness and contrast
            brightness = np.mean(gray_array)
            contrast = np.std(gray_array)
            features.extend([brightness, contrast])
            
            # 7. Dominant colors (simplified)
            # Reshape dan ambil sample untuk dominant color
            pixels = img_array.reshape(-1, 3)
            # Simple dominant color - most common color in reduced palette
            from collections import Counter
            # Reduce to 8 colors per channel untuk simplicity
            reduced_pixels = (pixels // 32) * 32
            pixel_tuples = [tuple(pixel) for pixel in reduced_pixels[::100]]  # Sample setiap 100 pixel
            color_counts = Counter(pixel_tuples)
            most_common = color_counts.most_common(5)
            
            # Add dominant colors as features
            for color, count in most_common:
                features.extend(list(color))
                features.append(count)
            
            # Pad jika kurang dari 5 dominant colors
            while len(most_common) < 5:
                features.extend([0, 0, 0, 0])  # RGB + count
                most_common.append((0, 0))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error extracting traditional features from {img_path}: {e}")
            return None

# ===== STEP 3: Model Training dengan Multiple Algorithms =====

class HouseClassifier:
    def __init__(self):
        self.models = {}
        self.feature_extractor = None
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.classes = []
        
    def prepare_data(self, feature_method='vgg16'):
        """Prepare features and labels"""
        print(f"Extracting features using {feature_method}...")
        
        self.feature_extractor = FeatureExtractor(feature_method)
        
        X = []
        y = []
        
        for _, row in df.iterrows():
            img_path = os.path.join("images", row['image'])
            if not os.path.exists(img_path):
                continue
                
            if feature_method in ['vgg16', 'resnet50', 'mobilenet']:
                features = self.feature_extractor.extract_deep_features(img_path)
            else:
                features = self.feature_extractor.extract_traditional_features(img_path)
            
            if features is not None:
                X.append(features)
                y.append(row['name'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        self.classes = sorted(list(set(y)))
        self.label_encoder = {cls: idx for idx, cls in enumerate(self.classes)}
        y_encoded = [self.label_encoder[label] for label in y]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, np.array(y_encoded), y
    
    def train_multiple_models(self, X, y):
        """Train multiple classifiers"""
        
        # Split data
        if len(X) > 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if very small dataset
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=min(3, len(X_train)),
                weights='distance'
            )
        }
        
        results = {}
        
        for name, model in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross validation if enough data
                if len(X) > 5:
                    cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)), scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean, cv_std = accuracy, 0
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_mean:.4f} (+/- {cv_std:.4f})")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        return results
    
    def select_best_model(self, results):
        """Select best performing model"""
        if not results:
            return None, None
            
        # Sort by cross-validation score
        best_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        best_model = results[best_name]['model']
        
        print(f"\nBest model: {best_name}")
        print(f"CV Score: {results[best_name]['cv_mean']:.4f}")
        
        return best_model, best_name
    
    def predict_with_confidence(self, img_path, threshold=0.6):
        """Predict with confidence scoring"""
        if self.feature_extractor.method in ['vgg16', 'resnet50', 'mobilenet']:
            features = self.feature_extractor.extract_deep_features(img_path)
        else:
            features = self.feature_extractor.extract_traditional_features(img_path)
        
        if features is None:
            return None, 0.0, []
        
        features_scaled = self.scaler.transform([features])
        
        # Get prediction probabilities
        if hasattr(self.best_model, 'predict_proba'):
            proba = self.best_model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            
            # Get top 3 predictions
            top_3_idx = np.argsort(proba)[-3:][::-1]
            top_3 = [
                {
                    'label': self.classes[idx],
                    'confidence': float(proba[idx])
                }
                for idx in top_3_idx
            ]
        else:
            # For models without probability
            pred_idx = self.best_model.predict(features_scaled)[0]
            confidence = 0.8  # Default confidence
            top_3 = [{'label': self.classes[pred_idx], 'confidence': confidence}]
        
        predicted_class = self.classes[pred_idx]
        
        return predicted_class, confidence, top_3

# ===== STEP 4: Training Process =====
print("=== TRAINING MULTIPLE MODELS ===")

# Test different feature extraction methods
feature_methods = ['vgg16', 'traditional']  # Start with these two
best_overall_score = 0
best_method = None
best_classifier = None

for method in feature_methods:
    print(f"\n{'='*50}")
    print(f"TESTING METHOD: {method.upper()}")
    print(f"{'='*50}")
    
    try:
        classifier = HouseClassifier()
        X, y_encoded, y_original = classifier.prepare_data(method)
        
        if len(X) == 0:
            print(f"No features extracted for {method}")
            continue
        
        results = classifier.train_multiple_models(X, y_encoded)
        
        if results:
            best_model, best_name = classifier.select_best_model(results)
            if best_model is not None:
                classifier.best_model = best_model
                classifier.best_method_name = best_name
                
                score = results[best_name]['cv_mean']
                if score > best_overall_score:
                    best_overall_score = score
                    best_method = method
                    best_classifier = classifier
                    
                    # Save best model
                    with open(f'best_model_{method}_{best_name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                        pickle.dump(classifier, f)
        
    except Exception as e:
        print(f"Error with method {method}: {e}")

print(f"\n{'='*60}")
print(f"BEST OVERALL: {best_method} with score {best_overall_score:.4f}")
print(f"{'='*60}")

# ===== STEP 5: FastAPI with Best Model =====
app = FastAPI()

# Load best model
if best_classifier is not None:
    model_classifier = best_classifier
else:
    print("No model trained successfully, using fallback...")
    model_classifier = None

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model_classifier is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not trained properly"}
        )
    
    if not image.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "File harus berupa gambar"}
        )
    
    contents = await image.read()
    temp_path = f"temp_{image.filename}"

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        predicted_class, confidence, top_3 = model_classifier.predict_with_confidence(
            temp_path, threshold=0.6
        )
        
        if predicted_class is None:
            return JSONResponse(content={
                "error": "Gagal memproses gambar"
            })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error dalam prediksi: {str(e)}"}
        )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Response dengan confidence yang lebih ketat
    if confidence < 0.6:
        return JSONResponse(content={
            "prediction": "Rumah tidak dikenal",
            "confidence": confidence,
            "top_predictions": top_3,
            "method": f"{model_classifier.feature_extractor.method} + {model_classifier.best_method_name}",
            "message": f"Confidence terlalu rendah ({confidence:.2%})"
        })

    # Cari info rumah
    match = df[df['name'].str.lower() == predicted_class.lower()]
    if not match.empty:
        nama = match.iloc[0]['name']
        return JSONResponse(content={
            "prediction": f"Ini rumah {nama}",
            "confidence": confidence,
            "top_predictions": top_3,
            "method": f"{model_classifier.feature_extractor.method} + {model_classifier.best_method_name}",
            "url": f"http://127.0.0.1:8000/{nama.lower()}",
            "message": f"Prediksi dengan confidence {confidence:.2%}"
        })
    else:
        return JSONResponse(content={
            "prediction": "Rumah tidak dikenal",
            "confidence": confidence,
            "top_predictions": top_3,
            "method": f"{model_classifier.feature_extractor.method} + {model_classifier.best_method_name}",
            "message": "Label tidak ditemukan dalam database"
        })

@app.get("/model_info")
async def model_info():
    if model_classifier is None:
        return JSONResponse(content={"error": "No model available"})
    
    return JSONResponse(content={
        "feature_method": model_classifier.feature_extractor.method,
        "classifier": model_classifier.best_method_name,
        "classes": model_classifier.classes,
        "confidence_threshold": 0.6,
        "best_cv_score": best_overall_score
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)