"""Quick evaluation - verify pipeline and generate baseline metrics for all models"""
import os, gc, json, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

print("Loading TensorFlow...")
sys.stdout.flush()
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.models_enhanced import MODEL_CONFIGS, create_pretrained_model
from src.data_pipeline import prepare_rfmid_dataset, get_class_weights
from src.data_preprocessing import MODEL_PREPROCESSING, apply_preprocessing_pipeline
from PIL import Image
import cv2

os.makedirs('saved_models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

print("Loading dataset...")
sys.stdout.flush()
data = prepare_rfmid_dataset('data/rfmid', label_column='ODC')
test_paths, test_labels = data['test']

SAMPLE_SIZE = min(100, len(test_paths))
np.random.seed(42)
indices = np.random.choice(len(test_paths), SAMPLE_SIZE, replace=False)
sample_paths = [test_paths[i] for i in indices]
sample_labels = [test_labels[i] for i in indices]

print(f"Using {SAMPLE_SIZE} test images for evaluation")
sys.stdout.flush()

def load_and_preprocess(path, image_size, pipeline='standard'):
    try:
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        img = apply_preprocessing_pipeline(img, pipeline)
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LANCZOS4)
        img = img.astype(np.float32) / 255.0
        return img
    except Exception:
        return None

models_to_evaluate = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_CONFIGS.keys())

all_results = {}
for model_name in models_to_evaluate:
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    sys.stdout.flush()

    try:
        config = MODEL_CONFIGS[model_name]
        image_size = config.get('input_size', (224, 224))
        pipeline = MODEL_PREPROCESSING.get(model_name, 'standard')

        images = []
        labels = []
        for path, label in zip(sample_paths, sample_labels):
            img = load_and_preprocess(path, image_size, pipeline)
            if img is not None:
                images.append(img)
                labels.append(label)

        if len(images) == 0:
            print(f"  No images loaded, skipping")
            continue

        X = np.array(images)
        y = np.array(labels)

        print(f"  Loaded {len(X)} images ({sum(y)} glaucoma, {len(y)-sum(y)} normal)")
        sys.stdout.flush()

        model_path = f'saved_models/{model_name}_best.keras'
        if os.path.exists(model_path):
            print(f"  Loading trained model...")
            model = tf.keras.models.load_model(model_path)
        else:
            print(f"  Creating new model (untrained baseline)...")
            if 'create_fn' in config:
                model = config['create_fn'](input_shape=(*image_size, 3), num_classes=1)
            else:
                model, _ = create_pretrained_model(model_name, input_shape=(*image_size, 3))

        preds = model.predict(X, batch_size=8, verbose=0).flatten()
        y_pred = (preds > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, preds) if len(np.unique(y)) > 1 else 0.5),
            'sensitivity': float(recall_score(y, y_pred, zero_division=0)),
            'specificity': float(recall_score(y, y_pred, pos_label=0, zero_division=0)),
        }

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        with open(f'results/metrics/{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        all_results[model_name] = metrics
        print(f"  DONE: {model_name}")

    except Exception as e:
        print(f"  ERROR: {e}")
        all_results[model_name] = {'error': str(e)}

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    sys.stdout.flush()

with open('results/metrics/all_models_metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*50}")
print(f"Evaluation complete: {len([r for r in all_results.values() if 'error' not in r])}/{len(all_results)} models succeeded")
