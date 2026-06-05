import os
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# Configuration
NATURAL_DIR = "data/natural"
AI_DIR = "data/ai_generated"
IMAGE_COUNT = 500
IMAGE_SIZE = (512, 512)

def download_natural_images():
    """Downloads 500 random natural photographs from the Picsum API with retries."""
    os.makedirs(NATURAL_DIR, exist_ok=True)
    print(f"Downloading {IMAGE_COUNT} natural images from Picsum...")
    
    for i in range(IMAGE_COUNT):
        filepath = os.path.join(NATURAL_DIR, f"natural_{i:03d}.jpg")
        
        # 1. Skip if already downloaded successfully
        if os.path.exists(filepath):
            continue
            
        url = f"https://picsum.photos/seed/eipr_{i}/{IMAGE_SIZE[0]}/{IMAGE_SIZE[1]}"
        max_retries = 3
        
        # 2. Retry loop for network hiccups
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    img.save(filepath)
                    
                    if (i + 1) % 50 == 0:
                        print(f"  Downloaded {i + 1}/{IMAGE_COUNT} natural images")
                    break # Success, break out of retry loop
                else:
                    print(f"  [Attempt {attempt+1}] Failed image {i} (Status: {response.status_code})")
                    
            except Exception as e:
                print(f"  [Attempt {attempt+1}] Error on image {i}: {type(e).__name__}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait 2 seconds before retrying
                else:
                    print(f"  --> Giving up on image {i} after {max_retries} attempts.")

def download_ai_images():
    """Downloads AI-generated images and filters near-duplicates."""
    os.makedirs(AI_DIR, exist_ok=True)

    existing_ai = len([f for f in os.listdir(AI_DIR) if f.endswith(".jpg")])

    if existing_ai >= IMAGE_COUNT:
        print(f"Already have {existing_ai} AI images. Skipping download.")
        return

    print(f"\nDownloading {IMAGE_COUNT} AI-generated images from HuggingFace (Similarity Filtering Enabled)...")

    try:
        import warnings
        warnings.filterwarnings("ignore")
        from datasets import load_dataset

        print("  Connecting to Midjourney stream...")

        dataset = load_dataset(
            "ehristoforu/midjourney-images",
            split="train",
            streaming=True
        )

        iterator = iter(dataset)

        downloaded = 0
        skipped_similar = 0

        previous_embedding = None

        while downloaded < IMAGE_COUNT:
            try:
                item = next(iterator)
            except StopIteration:
                break

            try:
                img = item["image"].convert("RGB")

                similarity_img = img.resize(
                    (128, 128),
                    Image.Resampling.LANCZOS
                )

                current_embedding = np.asarray(
                    similarity_img,
                    dtype=np.float32
                )

                if previous_embedding is not None:
                    mse = np.mean(
                        (current_embedding - previous_embedding) ** 2
                    )

                    if mse < 100:
                        skipped_similar += 1
                        continue

                previous_embedding = current_embedding

                final_img = img.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                filepath = os.path.join(
                    AI_DIR,
                    f"ai_gen_{downloaded:03d}.jpg"
                )

                final_img.save(filepath)

                downloaded += 1

                if downloaded % 50 == 0:
                    print(
                        f"  Saved {downloaded}/{IMAGE_COUNT} images "
                        f"(Skipped {skipped_similar} near-duplicates)"
                    )

            except Exception:
                pass

        print(f"\nFinished.")
        print(f"Saved images: {downloaded}")
        print(f"Skipped near-duplicates: {skipped_similar}")

    except Exception as e:
        print(f"Error downloading AI images: {e}")

if __name__ == "__main__":
    print("Starting dataset generation (resuming if interrupted)...")
    download_natural_images()
    download_ai_images()
    print("\nDataset generation complete! You are ready to run the experiments.")