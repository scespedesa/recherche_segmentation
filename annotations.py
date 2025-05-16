import os
import cv2
import numpy as np
import json
from skimage.filters import median, threshold_multiotsu
from skimage.morphology import disk
from skimage.measure import find_contours, approximate_polygon
import shutil
import random
from pathlib import Path

def split_dataset_aleatoire(base_folder, image_folder, label_folder, class_names,n_train, n_val, n_test, seed=42):
    # Collect images
    images = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])
    total = len(images)
    
    # Validation
    if n_train + n_val + n_test > total:
        raise ValueError("La suma de n_train, n_val y n_test excede el número total de imágenes.")

    # randomiser
    random.seed(seed)
    random.shuffle(images)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:n_train + n_val + n_test]

    print(f"✅ División aleatoria: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    # Create folder and copy
    for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(base_folder, split_name, sub), exist_ok=True)

        for img_file in split_imgs:
            base_name = Path(img_file).stem
            label_file = base_name + ".txt"

            shutil.copy(os.path.join(image_folder, img_file),
                        os.path.join(base_folder, split_name, "images", img_file))

            label_path = os.path.join(label_folder, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path,
                            os.path.join(base_folder, split_name, "labels", label_file))


def split_dataset(base_folder, image_folder, label_folder, class_names, n_train ,n_test=20, n_val=20):

    images = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])
    
    total = len(images)
    n_train = int(n_train)
    n_val = int(n_val)
    test_imgs= images[:n_test]
    val_imgs = images[n_test:n_test + n_val]
    train_imgs = images[n_test + n_val:n_val+n_train+n_test]

    # Structure for YOLO
    for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(base_folder, split_name, sub), exist_ok=True)

        for img_file in split_imgs:
            base_name = Path(img_file).stem
            label_file = base_name + ".txt"

            # Copy imagen
            shutil.copy(os.path.join(image_folder, img_file),
                        os.path.join(base_folder, split_name, "images", img_file))

            # Copy label
            label_path = os.path.join(label_folder, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path,
                            os.path.join(base_folder, split_name, "labels", label_file))

def split_dataset_porcentage(base_folder, image_folder, label_folder, class_names, seed=42):
    # set up
    train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    assert train_ratio + val_ratio + test_ratio == 1.0

    images = sorted([f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])
    random.seed(seed)
    random.shuffle(images)

    total = len(images)
    n_train = int(train_ratio * total)
    n_val = int(val_ratio * total)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(base_folder, split_name, sub), exist_ok=True)

        for img_file in split_imgs:
            base_name = Path(img_file).stem
            label_file = base_name + ".txt"

            #copy image
            shutil.copy(os.path.join(image_folder, img_file),
                        os.path.join(base_folder, split_name, "images", img_file))

            # copy label
            label_path = os.path.join(label_folder, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path,
                            os.path.join(base_folder, split_name, "labels", label_file))

def create_data_yaml(base_path, class_names):
    yaml_path = os.path.join(base_path, "data.yaml")
    content = f"""\
train: ../train/images
val: ../val/images
test: ../test/images
nc: {len(class_names)}
names: {class_names}
task: segment
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"✅ data.yaml generated in {yaml_path}")


def filter_image(tiff_image):
    # Convert to grayscale if it's not already
    tiff_image_gray = cv2.cvtColor(tiff_image, cv2.COLOR_BGR2GRAY) if len(tiff_image.shape) == 3 else tiff_image

    # Apply median filter
    filtered_image = median(tiff_image_gray, disk(7))  # Using a disk-shaped structuring element of radius 3

    # Apply multi-level thresholding
    thresholds = threshold_multiotsu(filtered_image, classes=3)
    segmented_image = np.digitize(filtered_image, bins=thresholds)

    return filtered_image, segmented_image


def save_yolo_segmentation(filename, width, height, dataset):
    label_path = filename  # Ya debe tener el .txt
    with open(label_path, "w") as f:
        for class_id, name in enumerate(["cracks", "grains"]):
            for polygon in dataset[name]:
                # be sure that each point has two coordinates (x, y)
                if not all(len(point) == 2 for point in polygon):
                    continue
                
                # Normalize
                norm_polygon = [(y / width, x / height) for x, y in polygon]

                if len(norm_polygon) < 3:
                    continue

                flat_coords = [f"{coord:.6f}" for point in norm_polygon for coord in point]
                f.write(f"{class_id} " + " ".join(flat_coords) + "\n")


def process_images(input_folder, output_folder, images_folder, labels_folder, polygons_folder, n1, n2, w):
    for folder in [output_folder, images_folder, labels_folder, polygons_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    all_files = sorted(os.listdir(input_folder))
    selected_files = all_files[n1:n2]
    via_dataset = {}

    for filename in selected_files:
        input_path = os.path.join(input_folder, filename)
        image_path = os.path.join(images_folder, os.path.splitext(filename)[0] + ".png")
        label_path = os.path.join(labels_folder, os.path.splitext(filename)[0] + ".txt")
        polygon_path = os.path.join(polygons_folder, os.path.splitext(filename)[0] + "_overlay.png")

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Image {filename} not found or unable to read.")
            continue
        
        img = np.double(img)
        
           
        ftd_img, segm_img = filter_image(img)
        dataset = {"cracks": [], "grains": []}
        
        overlay_img = img.copy()
        height, width = img.shape[:2]


        
        image_metadata = {"filename": os.path.basename(image_path), "size": os.path.getsize(input_path), "regions": []}
        
        for label, name, color in zip([0, 2], ["cracks", "grains"], [(0, 0, 255), (255, 0, 0)]):
            contours = find_contours(np.pad(segm_img==label, 1), 0.5)
            for contour in contours:
                polygon = approximate_polygon(contour, tolerance=1.0).astype(np.int32)
                if len(polygon) >= 3:  # Al menos 3 puntos para un polígono válido
                    dataset[name].append(polygon.tolist())
                    cv2.polylines(overlay_img, [polygon[:, ::-1].astype(np.int32)], isClosed=True, color=color, thickness=2)


        
        cv2.imwrite(polygon_path, overlay_img)
        print(f"Overlay saved: {polygon_path}")
        via_dataset[os.path.basename(image_path)] = image_metadata 
        cv2.imwrite(image_path, img)
        print(f"Processed and saved {filename}.")
        
        save_yolo_segmentation(label_path, width, height, dataset)



input_folder = './debug-test'
output_folder = './dataset-test'
images_folder = os.path.join(output_folder, 'images')
labels_folder = os.path.join(output_folder, 'labels')
polygons_folder = os.path.join(output_folder, 'polygons')

n1 = 0
n2 = 101
w = 320

process_images(input_folder, output_folder, images_folder, labels_folder, polygons_folder, n1, n2, w)
#nums_ale = random.sample(range(1, 61), 10)  # 10 números únicos del 1 al 60
nums_ale = [2,8,14, 20, 26, 32, 38, 46, 54, 60]
#nums_ale.sort()
i=0
for num_ale in nums_ale:
    i+=1
    seed = random.randint(0, 1000)
    print(num_ale)
    if not os.path.exists(f"./datasets_3/dataset_v{i}"):
        os.makedirs(f"./datasets_3/dataset_v{i}")
    split_dataset_aleatoire(
        base_folder=f"./datasets_3/dataset_v{i}",
        image_folder=f"./dataset-test/images",
        label_folder=f"./dataset-test/labels",
        class_names=["cracks", "grains"],
        n_train=num_ale,
        n_val=20,
        n_test=20,
        seed=seed  
    )
    create_data_yaml(f"./datasets_3/dataset_v{i}", ["cracks", "grains"])
