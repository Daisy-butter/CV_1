import numpy as np
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def load_cifar_batch(file_name):
    with open(file_name, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']

        # Rearrange data into (num_samples, 32, 32, 3), normalize to [0, 1].
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        labels = np.array(labels)
        return images, labels

def load_cifar10_train(data_dir):
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
    file_name = os.path.join(data_dir, "test_batch")
    X_test, y_test = load_cifar_batch(file_name)
    return X_test, y_test

def encode_one_hot(labels, num_classes):
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot = encoder.fit_transform(labels)
    return one_hot

def regmixup_data(images, labels, alpha=0.2, reg_factor=0.1):
    batch_size = images.shape[0]
    lam = np.random.beta(alpha, alpha)
    
    # Shuffle index for mixup
    indices = np.random.permutation(batch_size)
    mixed_images = lam * images + (1 - lam) * images[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    # Apply small adversarial perturbation as regularization
    perturbation = np.random.uniform(-reg_factor, reg_factor, size=mixed_images.shape)
    mixed_images = np.clip(mixed_images + perturbation, 0, 1)
    
    return mixed_images, mixed_labels

def augment_data(images, labels=None, use_regmixup=False, alpha=0.2, reg_factor=0.1):
    augmented_images = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        
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

    if use_regmixup and labels is not None:
        augmented_images, labels = regmixup_data(augmented_images, labels, alpha=alpha, reg_factor=reg_factor)

    return augmented_images, labels

def visualize_samples(images, labels=None, title="Sample Images", save_dir="data_preprocess_visualization"):
    os.makedirs(save_dir, exist_ok=True)

    # Select fixed samples for visualization (first 20 samples)
    sample_indices = np.arange(20)
    sampled_images = images[sample_indices]
    sampled_labels = labels[sample_indices] if labels is not None else None

    # Plot images in a grid (4 rows x 5 columns)
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow((sampled_images[i] * 255).astype(np.uint8))  # Convert from normalized to pixel values [0, 255]
        ax.axis('off')
        if sampled_labels is not None:
            class_label = np.argmax(sampled_labels[i])
            ax.set_title(f"Label: {class_label}", fontsize=8)

    # Save the figure with the specified title
    plt.suptitle(title, fontsize=28)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def preprocess_and_save(data_dir, save_dir, augment=False, use_regmixup=False, alpha=0.2, reg_factor=0.1):
    X_train, y_train = load_cifar10_train(data_dir)
    X_test, y_test = load_cifar10_test(data_dir)

    y_train_one_hot = encode_one_hot(y_train, 10)
    y_test_one_hot = encode_one_hot(y_test, 10)

    if augment:
        augmented_images, augmented_labels = augment_data(X_train, labels=y_train_one_hot, use_regmixup=use_regmixup, alpha=alpha, reg_factor=reg_factor)
        
        print("Saving original training samples...")
        visualize_samples(X_train, labels=y_train_one_hot, title="Original Training Samples", save_dir="data_preprocess_visualization")
        
        print("Saving augmented training samples...")
        visualize_samples(augmented_images, labels=augmented_labels, title="Augmented Training Samples", save_dir="data_preprocess_visualization")

        X_train = np.vstack((X_train, augmented_images))
        y_train_one_hot = np.vstack((y_train_one_hot, augmented_labels))

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train_one_hot)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test_one_hot)

if __name__ == "__main__":
    data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data/cifar-10-batches-py"
    save_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"
    augment = True
    use_regmixup = True
    preprocess_and_save(data_dir, save_dir, augment=augment, use_regmixup=use_regmixup)
    print("Data preprocessing including augmentation done!")
