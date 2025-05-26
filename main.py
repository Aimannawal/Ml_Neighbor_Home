import os
import pandas as pd
import shutil
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import uvicorn
import matplotlib.pyplot as plt

# ==== STEP 1: Siapkan Dataset dari data.csv ====
df = pd.read_csv("data.csv")

# Buat direktori dataset
dataset_dir = "dataset"
train_dir = "dataset/train"
val_dir = "dataset/validation"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Buat struktur folder dan split data
for label in df['name'].unique():
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(val_dir, label), exist_ok=True)

# Split data untuk setiap kelas (80% train, 20% validation)
for label in df['name'].unique():
    label_data = df[df['name'] == label]
    train_data, val_data = train_test_split(label_data, test_size=0.2, random_state=42)
    
    # Copy training images
    for _, row in train_data.iterrows():
        img_name = row['image']
        src = os.path.join("images", img_name)
        dst = os.path.join(train_dir, label, img_name)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Gambar tidak ditemukan: {src}")
    
    # Copy validation images
    for _, row in val_data.iterrows():
        img_name = row['image']
        src = os.path.join("images", img_name)
        dst = os.path.join(val_dir, label, img_name)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Gambar tidak ditemukan: {src}")

print(f"Dataset berhasil dibuat dengan {len(df['name'].unique())} kelas")

# ==== STEP 2: Improved CNN Model ====
image_size = (224, 224)  # Ukuran gambar lebih besar untuk detail lebih baik
batch_size = 16  # Batch size lebih kecil untuk memory efficiency

# Data augmentation yang lebih comprehensive
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Improved CNN Architecture
def create_improved_cnn_model(num_classes):
    model = models.Sequential([
        # First Convolutional Block
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Global Average Pooling (alternative to Flatten)
        layers.GlobalAveragePooling2D(),
        
        # Fully Connected Layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile the model
model = create_improved_cnn_model(train_gen.num_classes)

# Use a more sophisticated optimizer
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, decay=1e-6)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_3_accuracy']
)

print("Model Architecture:")
model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train the model
print("Memulai training...")
history = model.fit(
    train_gen,
    epochs=50,  # Lebih banyak epochs dengan early stopping
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# Save the model
model.save("improved_model.h5")
print("Model berhasil disimpan sebagai 'improved_model.h5'")

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)

# Evaluate model
print("\nEvaluasi Model:")
train_loss, train_acc, train_top3 = model.evaluate(train_gen, verbose=0)
val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=0)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Training Top-3 Accuracy: {train_top3:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation Top-3 Accuracy: {val_top3:.4f}")

# ==== STEP 3: Enhanced FastAPI Backend ====
app = FastAPI(title="House Classification API", version="2.0")

# Load model & data
try:
    model = load_model("improved_model.h5")
    print("Model berhasil dimuat")
except:
    print("Menggunakan model lama...")
    model = load_model("model.h5")

df = pd.read_csv("data.csv")

# Get class mapping
class_names = sorted(os.listdir(train_dir))
class_indices = {name: i for i, name in enumerate(class_names)}
index_to_class = {v: k for k, v in class_indices.items()}

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for prediction"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        temp_path = f"temp_{image.filename}"

        with open(temp_path, "wb") as f:
            f.write(contents)

        # Preprocess image
        img_array = preprocess_image(temp_path, target_size=(224, 224))

        # Make prediction
        preds = model.predict(img_array)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(preds[0])[-3:][::-1]
        top_3_probs = preds[0][top_3_indices]
        
        predictions = []
        for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_probs)):
            predicted_label = index_to_class[idx]
            confidence = float(prob)
            
            # Find matching data
            match = df[df['name'].str.lower() == predicted_label.lower()]
            if not match.empty:
                nama = match.iloc[0]['name']
                predictions.append({
                    "rank": i + 1,
                    "name": nama,
                    "confidence": confidence,
                    "percentage": f"{confidence * 100:.2f}%"
                })

        # Best prediction
        if predictions:
            best_prediction = predictions[0]
            result = f"Ini rumah {best_prediction['name']} (Confidence: {best_prediction['percentage']})"
            url = f"http://127.0.0.1:8000/{best_prediction['name'].lower()}"
        else:
            result = "Tidak ditemukan di data.csv"
            url = None

        # Clean up
        os.remove(temp_path)
        
        return JSONResponse(content={
            "prediction": result,
            "url": url,
            "top_predictions": predictions,
            "model_info": {
                "architecture": "Improved CNN",
                "input_size": "224x224",
                "classes": len(class_names)
            }
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"}, 
            status_code=500
        )

@app.get("/")
async def root():
    return {
        "message": "House Classification API v2.0",
        "model": "Improved CNN",
        "classes": len(class_names),
        "accuracy": f"Validation: {val_acc:.2%}"
    }

@app.get("/classes")
async def get_classes():
    return {"classes": class_names, "total": len(class_names)}

# ==== STEP 4: Jalankan Server ====
if __name__ == "__main__":
    print(f"\nServer siap dengan akurasi validasi: {val_acc:.2%}")
    print("Endpoint tersedia di:")
    print("- POST /predict - Upload gambar untuk prediksi")
    print("- GET / - Info API")
    print("- GET /classes - Daftar kelas")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)