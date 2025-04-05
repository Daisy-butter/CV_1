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

def apply_cutout(image, mask_size):
    """
    Apply Cutout data augmentation: random masking.
    :param image: Numpy array of shape (32, 32, 3).
    :param mask_size: Size of the square mask.
    :return: Image with a random masked region.
    """
    h, w = image.shape[:2]
    mask_size_half = mask_size // 2
    center_x = np.random.randint(0, w)
    center_y = np.random.randint(0, h)

    x1 = max(0, center_x - mask_size_half)
    x2 = min(w, center_x + mask_size_half)
    y1 = max(0, center_y - mask_size_half)
    y2 = min(h, center_y + mask_size_half)

    image[y1:y2, x1:x2, :] = 0  # Set the region to black.
    return image

def mixup_data(images, labels, alpha=0.2):
    """
    Apply Mixup data augmentation: linear interpolation of two image-label pairs.
    :param images: Numpy array of images.
    :param labels: Numpy array of one-hot encoded labels.
    :param alpha: Beta distribution parameter for Mixup.
    :return: Mixed images and labels.
    """
    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)
    
    indices = np.random.permutation(batch_size)
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    return mixed_images, mixed_labels

def augment_data(images, labels=None, use_cutout=False, cutout_size=8, use_mixup=False):
    """Data augmentation: horizontal flip, rotation, Cutout, and Mixup."""
    augmented_images = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # Turn into PIL image.
        
        # Random horizontal flip.
        if np.random.rand() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random rotation ±15 degrees.
        angle = np.random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle)

        # Random crop: Resize to 32x32 with random crop size between 0.9 and 1.0.
        crop_size = np.random.uniform(0.9, 1.0)  # Decide the crop size.
        crop_width, crop_height = int(32 * crop_size), int(32 * crop_size)
        left = np.random.randint(0, 32 - crop_width)
        top = np.random.randint(0, 32 - crop_height)
        pil_img = pil_img.crop((left, top, left + crop_width, top + crop_height))
        pil_img = pil_img.resize((32, 32))  # Adjust to 32x32.

        # Turn back to numpy array, normalize to [0, 1].
        img = np.array(pil_img) / 255.0

        # Apply Cutout if enabled.
        if use_cutout:
            img = apply_cutout(img, mask_size=cutout_size)

        augmented_images.append(img)

    augmented_images = np.array(augmented_images)

    # Apply Mixup if enabled.
    if use_mixup and labels is not None:
        augmented_images, labels = mixup_data(augmented_images, labels)

    return augmented_images, labels

def preprocess_and_save(data_dir, save_dir, augment=False, use_cutout=False, cutout_size=8, use_mixup=False):
    """Load, augment, preprocess CIFAR-10 data and save locally."""
    # Load CIFAR-10 data.
    X_train, y_train = load_cifar10_train(data_dir)
    X_test, y_test = load_cifar10_test(data_dir)

    # One-hot encode labels BEFORE data augmentation.
    print("Processing one-hot encoding...")
    y_train_one_hot = encode_one_hot(y_train, 10)
    y_test_one_hot = encode_one_hot(y_test, 10)

    # Data augmentation.
    if augment:
        print("Processing data augmentation...")
        X_train, y_train_one_hot = augment_data(X_train, labels=y_train_one_hot, use_cutout=use_cutout, cutout_size=cutout_size, use_mixup=use_mixup)

    # Flatten images.
    print("Processing flattening images...")
    X_train = X_train.reshape(X_train.shape[0], -1)  # (50000, 3072)
    X_test = X_test.reshape(X_test.shape[0], -1)    # (10000, 3072)

    # Save processed data to local directory.
    print("Loading processed data to local directory...")
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train_one_hot)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test_one_hot)

if __name__ == "__main__":
    # Raw data directory.
    data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data/cifar-10-batches-py"
    # Processed data directory.
    save_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"

    augment = True
    use_cutout = True
    cutout_size = 8
    use_mixup = True

    preprocess_and_save(data_dir, save_dir, augment=augment, use_cutout=use_cutout, cutout_size=cutout_size, use_mixup=use_mixup)
    print("Data preprocessing done!")
