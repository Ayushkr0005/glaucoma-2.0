#!/bin/bash
cd /home/runner/workspace

MODELS=("GlaucoNet" "GlaucoNet_V2" "GlaucoNet_V3" "ResNet50" "ResNet50V2" "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "InceptionV3" "Xception" "MobileNetV2" "EfficientNetB0" "EfficientNetV2S" "EfficientNetV2M" "NASNetMobile")

echo "Starting training of ${#MODELS[@]} models..."
echo "Start time: $(date)"

for model in "${MODELS[@]}"; do
    echo ""
    echo "============================================"
    echo "Training: $model"
    echo "Time: $(date)"
    echo "============================================"
    
    python3 train_single.py "$model" 10 5
    
    if [ $? -eq 0 ]; then
        echo "$model: SUCCESS"
    else
        echo "$model: FAILED"
    fi
done

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE"
echo "End time: $(date)"
echo "============================================"

python3 -c "
import os, json
metrics_dir = 'results/metrics'
all_metrics = {}
for f in os.listdir(metrics_dir):
    if f.endswith('_metrics.json'):
        name = f.replace('_metrics.json', '')
        with open(os.path.join(metrics_dir, f)) as fh:
            all_metrics[name] = json.load(fh)

with open(os.path.join(metrics_dir, 'all_models_metrics.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f'Merged metrics for {len(all_metrics)} models')
for name, m in sorted(all_metrics.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True):
    print(f'  {name}: acc={m[\"accuracy\"]:.4f} auc={m[\"roc_auc\"]:.4f}')
"
