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
import uvicorn

# ==== STEP 1: Siapkan Dataset dari data.csv ====
df = pd.read_csv("data.csv")

# Buat direktori dataset
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

for _, row in df.iterrows():
    label = row['name']
    img_name = row['image']
    label_dir = os.path.join(dataset_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    src = os.path.join("images", img_name)
    dst = os.path.join(label_dir, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"Gambar tidak ditemukan: {src}")

# ==== STEP 2: Melatih Model CNN ====
image_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10)
model.save("model.h5")

# ==== STEP 3: FastAPI Backend ====
app = FastAPI()

# Load model & data
model = load_model("model.h5")
df = pd.read_csv("data.csv")

# Ambil mapping kelas
class_names = sorted(os.listdir(dataset_dir))
class_indices = {name: i for i, name in enumerate(class_names)}
index_to_class = {v: k for k, v in class_indices.items()}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    contents = await image.read()
    temp_path = f"temp_{image.filename}"

    with open(temp_path, "wb") as f:
        f.write(contents)

    img = Image.open(temp_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    predicted_label = index_to_class[class_idx]

    match = df[df['name'].str.lower() == predicted_label.lower()]
    if not match.empty:
        nama = match.iloc[0]['name']
        result = f"Ini rumah {nama}"
        url = f"http://127.0.0.1:8000/{nama.lower()}"
    else:
        result = "Tidak ditemukan di data.csv"
        url = None

    os.remove(temp_path)
    return JSONResponse(content={"prediction": result, "url": url})

# ==== STEP 4: Jalankan Server ====
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
