import os, shutil
from sklearn.model_selection import train_test_split
from glob import glob

NUM_CLIENTS = 3
DATASET_DIR = "chest_xray"
CLIENT_DIR = "federatedlearning/chest_xray"

def create_clients():
    pneumonia_images = glob(os.path.join(DATASET_DIR, "train", "PNEUMONIA", "*.jpeg"))
    normal_images = glob(os.path.join(DATASET_DIR, "train", "NORMAL", "*.jpeg"))

    pneumonia_labels = ['PNEUMONIA'] * len(pneumonia_images)
    normal_labels = ['NORMAL'] * len(normal_images)

    images = pneumonia_images + normal_images
    labels = pneumonia_labels + normal_labels

    train_data, _ = train_test_split(
        list(zip(images, labels)), test_size=0.1, stratify=labels, random_state=42
    )

    os.makedirs(CLIENT_DIR, exist_ok=True)
    for i in range(NUM_CLIENTS):
        for cls in ['PNEUMONIA', 'NORMAL']:
            os.makedirs(f"{CLIENT_DIR}/client_{i}/{cls}", exist_ok=True)

    for i, (img, label) in enumerate(train_data):
        client_id = i % NUM_CLIENTS
        dest = f"{CLIENT_DIR}/client_{client_id}/{label}/{os.path.basename(img)}"
        shutil.copy(img, dest)

if __name__ == "__main__":
    create_clients()
