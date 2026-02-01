import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# PyTorch for GPU acceleration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# Configuration
CONFIG = {
    'data_dir': 'data',
    'train_dir': 'data/Train/Train',
    'test_dir': 'data/Test/Test',
    'val_dir': 'data/Val/Val',
    'window_size': (128, 64),  # Height x Width (standard for pedestrian detection)
    'hog_params': {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys',
        'visualize': False,
        'transform_sqrt': True
    },
    'negative_samples_per_image': 10,  # Number of negative samples to extract per image
    'max_samples': None,  # Set to None for all samples, or a number to limit
    'model_path': 'pedestrian_detector_model.pkl',
    'scaler_path': 'feature_scaler.pkl'
}


def parse_annotation(annotation_path):
    """Parse Pascal VOC format XML annotation file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'person':
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            difficult = obj.find('difficult')
            is_difficult = int(difficult.text) if difficult is not None else 0

            objects.append({
                'name': name,
                'bbox': (xmin, ymin, xmax, ymax),
                'difficult': is_difficult
            })

    return {
        'width': img_width,
        'height': img_height,
        'objects': objects
    }


def compute_hog_features(image):
    """Compute HOG features for an image patch."""
    # Resize to standard window size
    resized = cv2.resize(image, (CONFIG['window_size'][1], CONFIG['window_size'][0]))

    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # Compute HOG features
    features = hog(
        gray,
        orientations=CONFIG['hog_params']['orientations'],
        pixels_per_cell=CONFIG['hog_params']['pixels_per_cell'],
        cells_per_block=CONFIG['hog_params']['cells_per_block'],
        block_norm=CONFIG['hog_params']['block_norm'],
        visualize=CONFIG['hog_params']['visualize'],
        transform_sqrt=CONFIG['hog_params']['transform_sqrt']
    )

    return features


def compute_hog_visualization(image):
    """Compute HOG features with visualization."""
    resized = cv2.resize(image, (CONFIG['window_size'][1], CONFIG['window_size'][0]))

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    features, hog_image = hog(
        gray,
        orientations=CONFIG['hog_params']['orientations'],
        pixels_per_cell=CONFIG['hog_params']['pixels_per_cell'],
        cells_per_block=CONFIG['hog_params']['cells_per_block'],
        block_norm=CONFIG['hog_params']['block_norm'],
        visualize=True,
        transform_sqrt=CONFIG['hog_params']['transform_sqrt']
    )

    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return features, hog_image


def compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def extract_positive_samples(data_dir):
    """Extract positive samples (pedestrians) from the dataset."""
    images_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    positive_samples = []

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    if CONFIG['max_samples']:
        annotation_files = annotation_files[:CONFIG['max_samples']]

    print(f"Extracting positive samples from {len(annotation_files)} images...")

    for ann_file in tqdm(annotation_files, desc="Extracting positives"):
        annotation_path = os.path.join(annotations_dir, ann_file)
        image_name = ann_file.replace('.xml', '.png')
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_name = ann_file.replace('.xml', '.jpg')
            image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        try:
            annotation = parse_annotation(annotation_path)
            image = cv2.imread(image_path)

            if image is None:
                continue

            for obj in annotation['objects']:
                if obj['difficult']:
                    continue

                xmin, ymin, xmax, ymax = obj['bbox']

                # Ensure valid bounding box
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(image.shape[1], xmax)
                ymax = min(image.shape[0], ymax)

                if xmax <= xmin or ymax <= ymin:
                    continue

                # Extract pedestrian region
                pedestrian_region = image[ymin:ymax, xmin:xmax]

                if pedestrian_region.size == 0:
                    continue

                # Compute HOG features
                features = compute_hog_features(pedestrian_region)
                positive_samples.append(features)

        except Exception as e:
            print(f"Error processing {ann_file}: {e}")
            continue

    return np.array(positive_samples)


def extract_negative_samples(data_dir, positive_boxes_per_image=None):
    """Extract negative samples (non-pedestrian regions) from the dataset."""
    images_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    negative_samples = []

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]

    if CONFIG['max_samples']:
        annotation_files = annotation_files[:CONFIG['max_samples']]

    print(f"Extracting negative samples from {len(annotation_files)} images...")

    for ann_file in tqdm(annotation_files, desc="Extracting negatives"):
        annotation_path = os.path.join(annotations_dir, ann_file)
        image_name = ann_file.replace('.xml', '.png')
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_name = ann_file.replace('.xml', '.jpg')
            image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        try:
            annotation = parse_annotation(annotation_path)
            image = cv2.imread(image_path)

            if image is None:
                continue

            img_height, img_width = image.shape[:2]
            person_boxes = [obj['bbox'] for obj in annotation['objects']]

            # Extract random negative samples
            samples_extracted = 0
            max_attempts = CONFIG['negative_samples_per_image'] * 10
            attempts = 0

            while samples_extracted < CONFIG['negative_samples_per_image'] and attempts < max_attempts:
                attempts += 1

                # Random window size with aspect ratio similar to pedestrian
                width = np.random.randint(32, min(128, img_width))
                height = int(width * 2)  # Pedestrian aspect ratio ~2:1

                if height >= img_height or width >= img_width:
                    continue

                # Random position
                x = np.random.randint(0, img_width - width)
                y = np.random.randint(0, img_height - height)

                candidate_box = (x, y, x + width, y + height)

                # Check IoU with all pedestrian boxes
                is_negative = True
                for person_box in person_boxes:
                    if compute_iou(candidate_box, person_box) > 0.3:
                        is_negative = False
                        break

                if is_negative:
                    region = image[y:y+height, x:x+width]
                    if region.size > 0:
                        features = compute_hog_features(region)
                        negative_samples.append(features)
                        samples_extracted += 1

        except Exception as e:
            continue

    return np.array(negative_samples)


def prepare_training_data():
    """Prepare training data with positive and negative samples."""
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)

    # Extract positive samples (pedestrians)
    positive_samples = extract_positive_samples(CONFIG['train_dir'])
    print(f"Extracted {len(positive_samples)} positive samples")

    # Extract negative samples (non-pedestrians)
    negative_samples = extract_negative_samples(CONFIG['train_dir'])
    print(f"Extracted {len(negative_samples)} negative samples")

    # Balance the dataset
    min_samples = min(len(positive_samples), len(negative_samples))
    if min_samples > 0:
        # Undersample the majority class
        if len(positive_samples) > min_samples:
            indices = np.random.choice(len(positive_samples), min_samples, replace=False)
            positive_samples = positive_samples[indices]
        if len(negative_samples) > min_samples:
            indices = np.random.choice(len(negative_samples), min_samples, replace=False)
            negative_samples = negative_samples[indices]

    print(f"Balanced dataset: {len(positive_samples)} positive, {len(negative_samples)} negative")

    # Combine samples
    X = np.vstack([positive_samples, negative_samples])
    y = np.hstack([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def prepare_test_data():
    """Prepare test data."""
    print("\n" + "="*60)
    print("PREPARING TEST DATA")
    print("="*60)

    # Extract positive samples
    positive_samples = extract_positive_samples(CONFIG['test_dir'])
    print(f"Extracted {len(positive_samples)} positive test samples")

    # Extract negative samples
    negative_samples = extract_negative_samples(CONFIG['test_dir'])
    print(f"Extracted {len(negative_samples)} negative test samples")

    # Combine samples
    if len(positive_samples) == 0 or len(negative_samples) == 0:
        return None, None

    X = np.vstack([positive_samples, negative_samples])
    y = np.hstack([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


class GPUSVMClassifier(nn.Module):
    """GPU-accelerated SVM-like classifier using hinge loss."""

    def __init__(self, input_dim, C=1.0):
        super(GPUSVMClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.C = C

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(DEVICE)
            outputs = self.forward(x)
            return (outputs.squeeze() > 0).cpu().numpy().astype(int)

    def predict_proba(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(DEVICE)
            outputs = torch.sigmoid(self.forward(x))
            probs = outputs.squeeze().cpu().numpy()
            return np.column_stack([1 - probs, probs])


def hinge_loss(outputs, labels, model, C):
    """SVM hinge loss with L2 regularization."""
    labels = labels.float() * 2 - 1  # Convert 0/1 to -1/1
    losses = torch.clamp(1 - labels * outputs.squeeze(), min=0)

    # L2 regularization
    l2_reg = 0.5 * torch.sum(model.linear.weight ** 2)

    return l2_reg + C * torch.mean(losses)


def train_svm_gpu(X_train, y_train, C=1.0, epochs=100, batch_size=256, lr=0.01):
    """Train SVM classifier on GPU using PyTorch."""
    print("\n" + "="*60)
    print("TRAINING SVM CLASSIFIER (GPU ACCELERATED)")
    print("="*60)
    print(f"Device: {DEVICE}")

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Feature vector dimension: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_tensor = torch.FloatTensor(y_train).to(DEVICE)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = GPUSVMClassifier(X_train.shape[1], C=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\nTraining with C={C}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print("Training progress:")

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = hinge_loss(outputs, batch_y, model, C)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Calculate training accuracy
            model.eval()
            with torch.no_grad():
                train_pred = model.predict(X_tensor)
                train_acc = accuracy_score(y_train, train_pred)
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()

    # Final training accuracy
    with torch.no_grad():
        final_pred = model.predict(X_tensor)
        final_acc = accuracy_score(y_train, final_pred)
    print(f"\nFinal Training Accuracy: {final_acc:.4f}")

    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': X_train.shape[1],
        'C': C
    }, CONFIG['model_path'].replace('.pkl', '_gpu.pth'))

    with open(CONFIG['scaler_path'], 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nGPU Model saved to {CONFIG['model_path'].replace('.pkl', '_gpu.pth')}")
    print(f"Scaler saved to {CONFIG['scaler_path']}")

    return model, scaler


def train_svm(X_train, y_train):
    """Train SVM classifier - uses GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        # Use GPU-accelerated training
        return train_svm_gpu(X_train, y_train, C=1.0, epochs=100, batch_size=256, lr=0.01)
    else:
        # Fallback to sklearn SVM on CPU
        print("\n" + "="*60)
        print("TRAINING SVM CLASSIFIER (CPU)")
        print("="*60)
        print("Note: GPU not available, using sklearn SVM on CPU")

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        print(f"Feature vector dimension: {X_train.shape[1]}")
        print(f"Training samples: {X_train.shape[0]}")

        # Train SVM with RBF kernel
        print("\nTraining SVM with grid search for optimal parameters...")

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.001],
            'kernel': ['rbf']
        }

        svm = SVC(probability=True, random_state=42)

        grid_search = GridSearchCV(
            svm, param_grid, cv=3, scoring='accuracy',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_

        # Save model and scaler
        with open(CONFIG['model_path'], 'wb') as f:
            pickle.dump(best_model, f)
        with open(CONFIG['scaler_path'], 'wb') as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved to {CONFIG['model_path']}")
        print(f"Scaler saved to {CONFIG['scaler_path']}")

        return best_model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate the trained model on test data."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    # Scale test features
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy:':<20} {accuracy:.4f}")
    print(f"{'Precision:':<20} {precision:.4f}")
    print(f"{'Recall:':<20} {recall:.4f}")
    print(f"{'F1-Score:':<20} {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual Neg      {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual Pos      {cm[1,0]:5d}  {cm[1,1]:5d}")

    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Pedestrian', 'Pedestrian']))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print(f"AUC-ROC Score: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_results(results):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = results['confusion_matrix']
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=['Non-Pedestrian', 'Pedestrian'],
            yticklabels=['Non-Pedestrian', 'Pedestrian'],
            ylabel='True Label',
            xlabel='Predicted Label',
            title='Confusion Matrix')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    # 2. ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(results['fpr'], results['tpr'], color='darkorange', lw=2,
             label=f'ROC curve (AUC = {results["roc_auc"]:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # 3. Performance Metrics Bar Chart
    ax3 = axes[1, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = ax3.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylim([0, 1])
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax3.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Prediction Distribution
    ax4 = axes[1, 1]
    ax4.hist(results['y_pred_proba'][results['y_test'] == 0], bins=30, alpha=0.7,
             label='Non-Pedestrian', color='blue', density=True)
    ax4.hist(results['y_pred_proba'][results['y_test'] == 1], bins=30, alpha=0.7,
             label='Pedestrian', color='red', density=True)
    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold')
    ax4.set_xlabel('Prediction Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('Prediction Probability Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nResults saved to 'evaluation_results.png'")


def visualize_hog_features():
    """Visualize HOG features for sample images."""
    print("\n" + "="*60)
    print("VISUALIZING HOG FEATURES")
    print("="*60)

    images_dir = os.path.join(CONFIG['train_dir'], 'JPEGImages')
    annotations_dir = os.path.join(CONFIG['train_dir'], 'Annotations')

    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')][:3]

    fig, axes = plt.subplots(len(annotation_files), 3, figsize=(12, 4 * len(annotation_files)))

    for idx, ann_file in enumerate(annotation_files):
        annotation_path = os.path.join(annotations_dir, ann_file)
        image_name = ann_file.replace('.xml', '.png')
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            image_name = ann_file.replace('.xml', '.jpg')
            image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            continue

        annotation = parse_annotation(annotation_path)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get first pedestrian bounding box
        if annotation['objects']:
            bbox = annotation['objects'][0]['bbox']
            xmin, ymin, xmax, ymax = bbox
            pedestrian = image[ymin:ymax, xmin:xmax]
            pedestrian_rgb = cv2.cvtColor(pedestrian, cv2.COLOR_BGR2RGB)

            # Compute HOG with visualization
            _, hog_image = compute_hog_visualization(pedestrian)

            # Draw bounding box on original image
            image_with_box = image_rgb.copy()
            cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Plot
            if len(annotation_files) == 1:
                ax_row = axes
            else:
                ax_row = axes[idx]

            ax_row[0].imshow(image_with_box)
            ax_row[0].set_title('Original Image with Detection')
            ax_row[0].axis('off')

            ax_row[1].imshow(cv2.resize(pedestrian_rgb, (CONFIG['window_size'][1], CONFIG['window_size'][0])))
            ax_row[1].set_title('Extracted Pedestrian')
            ax_row[1].axis('off')

            ax_row[2].imshow(hog_image, cmap='gray')
            ax_row[2].set_title('HOG Features')
            ax_row[2].axis('off')

    plt.tight_layout()
    plt.savefig('hog_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("HOG visualization saved to 'hog_visualization.png'")


def detect_pedestrians_sliding_window(image, model, scaler, step_size=16, scales=[1.0, 0.75, 0.5]):
    """Detect pedestrians using sliding window approach."""
    detections = []
    window_h, window_w = CONFIG['window_size']

    for scale in scales:
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        if resized.shape[0] < window_h or resized.shape[1] < window_w:
            continue

        for y in range(0, resized.shape[0] - window_h, step_size):
            for x in range(0, resized.shape[1] - window_w, step_size):
                window = resized[y:y + window_h, x:x + window_w]

                features = compute_hog_features(window)
                features_scaled = scaler.transform([features])

                prob = model.predict_proba(features_scaled)[0][1]

                if prob > 0.8:  # High confidence threshold
                    # Scale back coordinates
                    x1 = int(x / scale)
                    y1 = int(y / scale)
                    x2 = int((x + window_w) / scale)
                    y2 = int((y + window_h) / scale)

                    detections.append((x1, y1, x2, y2, prob))

    return detections


def non_max_suppression(detections, overlap_thresh=0.3):
    """Apply Non-Maximum Suppression to reduce overlapping detections."""
    if len(detections) == 0:
        return []

    boxes = np.array([[d[0], d[1], d[2], d[3]] for d in detections])
    scores = np.array([d[4] for d in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:
        i = indices[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[indices[1:]]

        indices = indices[np.where(overlap <= overlap_thresh)[0] + 1]

    return [detections[i] for i in keep]


def demo_detection(model, scaler, num_images=5):
    """Demo pedestrian detection on test images."""
    print("\n" + "="*60)
    print("DEMO: PEDESTRIAN DETECTION")
    print("="*60)

    images_dir = os.path.join(CONFIG['test_dir'], 'JPEGImages')
    annotations_dir = os.path.join(CONFIG['test_dir'], 'Annotations')

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))][:num_images]

    fig, axes = plt.subplots(len(image_files), 2, figsize=(12, 5 * len(image_files)))

    for idx, img_file in enumerate(image_files):
        image_path = os.path.join(images_dir, img_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Ground truth
        ann_file = img_file.replace('.png', '.xml').replace('.jpg', '.xml')
        annotation_path = os.path.join(annotations_dir, ann_file)

        gt_boxes = []
        if os.path.exists(annotation_path):
            annotation = parse_annotation(annotation_path)
            gt_boxes = [obj['bbox'] for obj in annotation['objects']]

        # Detect pedestrians
        detections = detect_pedestrians_sliding_window(image, model, scaler)
        detections = non_max_suppression(detections)

        # Plot ground truth
        img_gt = image_rgb.copy()
        for box in gt_boxes:
            cv2.rectangle(img_gt, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Plot detections
        img_det = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2, score = det
            cv2.rectangle(img_det, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_det, f'{score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if len(image_files) == 1:
            ax_row = axes
        else:
            ax_row = axes[idx]

        ax_row[0].imshow(img_gt)
        ax_row[0].set_title(f'Ground Truth ({len(gt_boxes)} pedestrians)')
        ax_row[0].axis('off')

        ax_row[1].imshow(img_det)
        ax_row[1].set_title(f'Detections ({len(detections)} detected)')
        ax_row[1].axis('off')

    plt.tight_layout()
    plt.savefig('detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Detection demo saved to 'detection_demo.png'")


def main():
    """Main function to run the complete pipeline."""
    print("="*60)
    print("PEDESTRIAN DETECTION USING HOG + SVM")
    print("="*60)

    # Check if data exists
    if not os.path.exists(CONFIG['train_dir']):
        print(f"Error: Training data not found at {CONFIG['train_dir']}")
        return

    # Step 1: Visualize HOG features
    visualize_hog_features()

    # Step 2: Prepare training data
    X_train, y_train = prepare_training_data()

    if len(X_train) == 0:
        print("Error: No training data extracted!")
        return

    # Step 3: Train SVM classifier
    model, scaler = train_svm(X_train, y_train)

    # Step 4: Prepare test data
    X_test, y_test = prepare_test_data()

    if X_test is None or len(X_test) == 0:
        print("Warning: No test data available. Using validation data...")
        # Try validation data
        val_positive = extract_positive_samples(CONFIG['val_dir'])
        val_negative = extract_negative_samples(CONFIG['val_dir'])

        if len(val_positive) > 0 and len(val_negative) > 0:
            X_test = np.vstack([val_positive, val_negative])
            y_test = np.hstack([np.ones(len(val_positive)), np.zeros(len(val_negative))])
        else:
            print("Error: No test/validation data available!")
            return

    # Step 5: Evaluate model
    results = evaluate_model(model, scaler, X_test, y_test)

    # Step 6: Plot results
    plot_results(results)

    # Step 7: Demo detection
    demo_detection(model, scaler)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final Precision: {results['precision']:.4f}")
    print(f"Final Recall: {results['recall']:.4f}")
    print(f"Final F1-Score: {results['f1']:.4f}")
    print(f"AUC-ROC: {results['roc_auc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
