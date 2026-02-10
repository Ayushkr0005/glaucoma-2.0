import os
import shutil
import random
import numpy as np
import cv2
from PIL import Image

MODEL_INPUT_SIZES = {
    'GlaucoNet': (224, 224),
    'GlaucoNet_V2': (224, 224),
    'GlaucoNet_V3': (224, 224),
    'ResNet50': (224, 224),
    'ResNet50V2': (224, 224),
    'VGG16': (224, 224),
    'VGG19': (224, 224),
    'DenseNet121': (224, 224),
    'DenseNet169': (224, 224),
    'DenseNet201': (224, 224),
    'InceptionV3': (299, 299),
    'Xception': (299, 299),
    'MobileNetV2': (224, 224),
    'EfficientNetB0': (224, 224),
    'EfficientNetV2S': (384, 384),
    'EfficientNetV2M': (480, 480),
    'NASNetMobile': (224, 224),
}

CLASS_NAMES = ['glaucoma', 'normal']

def get_input_size(model_name):
    return MODEL_INPUT_SIZES.get(model_name, (224, 224))


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        result = clahe.apply(image)
    return result


def ben_graham_preprocessing(image, sigma=10):
    img = image.copy()
    img = img.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    result = cv2.addWeighted(img, 4.0, blur, -4.0, 128.0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def extract_green_channel(image):
    if len(image.shape) == 3 and image.shape[2] >= 3:
        green = image[:, :, 1]
        return cv2.merge([green, green, green])
    return image


def enhance_vessels(image, kernel_size=17):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
    vessels = cv2.subtract(background, enhanced)
    vessels = cv2.normalize(vessels, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([vessels, vessels, vessels])


def crop_optic_disc_region(image, crop_ratio=0.4):
    h, w = image.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    _, max_val, _, max_loc = cv2.minMaxLoc(blurred)

    cx, cy = max_loc
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)

    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)

    cropped = image[y1:y2, x1:x2]
    return cropped


def normalize_illumination(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)
    mean_l = np.mean(l_channel)
    target_mean = 128.0
    scale = target_mean / (mean_l + 1e-7)
    l_channel = np.clip(l_channel * scale, 0, 255).astype(np.uint8)
    lab[:, :, 0] = l_channel
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def remove_black_border(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image[y0:y1, x0:x1]
    return cropped


def adaptive_histogram_equalization(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def sharpen_image(image, strength=1.0):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    blended_kernel = np.eye(3, dtype=np.float32) * (1 - strength) + kernel * strength / 9
    padded = np.zeros((3, 3), dtype=np.float32)
    padded[1, 1] = 1 - strength
    padded += kernel * strength / 9
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 1 - strength, sharpened, strength, 0)


def denoise_image(image, strength=10):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


PREPROCESSING_PIPELINES = {
    'standard': ['remove_border', 'normalize_illumination', 'clahe'],
    'ben_graham': ['remove_border', 'ben_graham', 'clahe'],
    'green_channel': ['remove_border', 'green_channel', 'clahe'],
    'vessel_enhanced': ['remove_border', 'normalize_illumination', 'vessel_enhance'],
    'optic_disc_focus': ['remove_border', 'normalize_illumination', 'optic_disc_crop', 'clahe'],
    'full_pipeline': ['remove_border', 'denoise', 'normalize_illumination', 'clahe', 'sharpen'],
}

MODEL_PREPROCESSING = {
    'GlaucoNet': 'standard',
    'GlaucoNet_V2': 'full_pipeline',
    'GlaucoNet_V3': 'full_pipeline',
    'ResNet50': 'standard',
    'ResNet50V2': 'standard',
    'VGG16': 'ben_graham',
    'VGG19': 'ben_graham',
    'DenseNet121': 'standard',
    'DenseNet169': 'standard',
    'DenseNet201': 'standard',
    'InceptionV3': 'standard',
    'Xception': 'standard',
    'MobileNetV2': 'standard',
    'EfficientNetB0': 'standard',
    'EfficientNetV2S': 'full_pipeline',
    'EfficientNetV2M': 'full_pipeline',
    'NASNetMobile': 'standard',
}


def apply_preprocessing_pipeline(image, pipeline_name='standard'):
    steps = PREPROCESSING_PIPELINES.get(pipeline_name, PREPROCESSING_PIPELINES['standard'])
    img = image.copy()

    for step in steps:
        if step == 'remove_border':
            img = remove_black_border(img)
        elif step == 'normalize_illumination':
            img = normalize_illumination(img)
        elif step == 'clahe':
            img = apply_clahe(img)
        elif step == 'ben_graham':
            img = ben_graham_preprocessing(img)
        elif step == 'green_channel':
            img = extract_green_channel(img)
        elif step == 'vessel_enhance':
            img = enhance_vessels(img)
        elif step == 'optic_disc_crop':
            img = crop_optic_disc_region(img)
        elif step == 'denoise':
            img = denoise_image(img)
        elif step == 'sharpen':
            img = sharpen_image(img, strength=0.3)

    return img


def preprocess_for_model(image, model_name, target_size=None):
    if target_size is None:
        target_size = get_input_size(model_name)

    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    pipeline_name = MODEL_PREPROCESSING.get(model_name, 'standard')
    img = apply_preprocessing_pipeline(img, pipeline_name)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def preprocess_single_image(image, target_size=(224, 224)):
    if isinstance(image, str):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    img = remove_black_border(img)
    img = normalize_illumination(img)
    img = apply_clahe(img)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def get_dataset_stats(data_dir):
    stats = {}
    total_images = 0

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[class_name] = count
            total_images += count

    stats['total'] = total_images
    return stats
