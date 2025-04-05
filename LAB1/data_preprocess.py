import numpy as np
import pickle
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

def load_cifar_batch(file_name):
    """Load CIFAR-10 single batch file."""
    with open(file_name, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']  # 3072 dimensional images
        labels = batch[b'labels']

        # Rearrange data into (num_samples, 32, 32, 3), normalize to [0, 1].
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        labels = np.array(labels)
        return images, labels

def load_cifar10_train(data_dir):
    """Load CIFAR-10 training data."""
    images_list = []
    labels_list = []
    for i in range(1, 6):
        file_name = os.path.join(data_dir, f"data_batch_{i}")
        images, labels = load_cifar_batch(file_name)
        images_list.append(images)
        labels_list.append(labels)
    
    # Merge all batches into one.
    X_train = np.vstack(images_list)
    y_train = np.hstack(labels_list)
    return X_train, y_train

def load_cifar10_test(data_dir):
    """Load CIFAR-10 test data."""
    file_name = os.path.join(data_dir, "test_batch")
    X_test, y_test = load_cifar_batch(file_name)
    return X_test, y_test

def encode_one_hot(labels, num_classes):
    """One-hot encode labels."""
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot = encoder.fit_transform(labels)
    return one_hot

def regmixup_data(images, labels, alpha=0.2, reg_factor=0.1):
    """
    Apply RegMixup data augmentation: combines Mixup with regularization.
    :param images: Numpy array of images.
    :param labels: Numpy array of one-hot encoded labels.
    :param alpha: Beta distribution parameter for Mixup.
    :param reg_factor: Regularization factor scaling the perturbation.
    :return: Augmented images and labels.
    """
    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)
    
    # Shuffle index for mixup
    indices = np.random.permutation(batch_size)
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    # Apply small adversarial perturbation as regularization
    perturbation = np.random.uniform(-reg_factor, reg_factor, size=mixed_images.shape)
    mixed_images = np.clip(mixed_images + perturbation, 0, 1)  # Ensure values remain normalized
    
    return mixed_images, mixed_labels

def augment_data(images, labels=None, use_regmixup=False, alpha=0.2, reg_factor=0.1):
    """Data augmentation pipeline with RegMixup."""
    augmented_images = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to PIL image
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation ±15 degrees
        angle = np.random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle)

        # Turn back to numpy array, normalize to [0, 1]
        img = np.array(pil_img) / 255.0
        augmented_images.append(img)

    augmented_images = np.array(augmented_images)

    # Apply RegMixup if enabled
    if use_regmixup and labels is not None:
        augmented_images, labels = regmixup_data(augmented_images, labels, alpha=alpha, reg_factor=reg_factor)

    return augmented_images, labels

def preprocess_and_save(data_dir, save_dir, augment=False, use_regmixup=False, alpha=0.2, reg_factor=0.1):
    """Load, augment, preprocess CIFAR-10 data and save locally."""
    # Load CIFAR-10 data
    X_train, y_train = load_cifar10_train(data_dir)
    X_test, y_test = load_cifar10_test(data_dir)

    # One-hot encode labels BEFORE data augmentation
    print("Processing one-hot encoding...")
    y_train_one_hot = encode_one_hot(y_train, 10)
    y_test_one_hot = encode_one_hot(y_test, 10)

    # Data augmentation
    if augment:
        print("Processing data augmentation...")
        X_train, y_train_one_hot = augment_data(X_train, labels=y_train_one_hot, use_regmixup=use_regmixup, alpha=alpha, reg_factor=reg_factor)

    # Flatten images
    print("Processing flattening images...")
    X_train = X_train.reshape(X_train.shape[0], -1)  # (50000, 3072)
    X_test = X_test.reshape(X_test.shape[0], -1)    # (10000, 3072)

    # Save processed data to local directory
    print("Loading processed data to local directory...")
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train_one_hot)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test_one_hot)

if __name__ == "__main__":
    # Raw data directory
    data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data/cifar-10-batches-py"
    # Processed data directory
    save_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"

    # Enable augmentation with RegMixup
    augment = True
    use_regmixup = True
    alpha = 0.2  # Beta distribution parameter for RegMixup
    reg_factor = 0.1  # Perturbation scale for regularization

    preprocess_and_save(data_dir, save_dir, augment=augment, use_regmixup=use_regmixup, alpha=alpha, reg_factor=reg_factor)
    print("Data preprocessing done!")
