import numpy as np
import pickle
import os
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

def load_cifar_batch(file_name):
    # load CIFAR-10 batch data
    with open(file_name, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']  # 3072 dimensional images
        labels = batch[b'labels']

        # rearrange data into (num_samples, 32, 32, 3)，normalize to [0, 1]
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
        labels = np.array(labels)
        return images, labels

def load_cifar10_train(data_dir):
    # load CIFAR-10 training data
    images_list = []
    labels_list = []
    for i in range(1, 6):
        file_name = os.path.join(data_dir, f"data_batch_{i}")
        images, labels = load_cifar_batch(file_name)
        images_list.append(images)
        labels_list.append(labels)
    
    # merge all batches into one
    X_train = np.vstack(images_list)
    y_train = np.hstack(labels_list)
    return X_train, y_train

def load_cifar10_test(data_dir):
    # load CIFAR-10 test data
    file_name = os.path.join(data_dir, "test_batch")
    X_test, y_test = load_cifar_batch(file_name)
    return X_test, y_test

def encode_one_hot(labels, num_classes):
    # one-hot encode labels
    encoder = OneHotEncoder(categories='auto', sparse_output=False)
    labels = labels.reshape(-1, 1)
    one_hot = encoder.fit_transform(labels)
    return one_hot

def augment_data(images):
    # data augmentation: random horizontal flip and random rotation
    augmented_images = []
    for img in images:
        pil_img = Image.fromarray((img * 255).astype(np.uint8))  # turn into PIL image
        # random horizontal flip
        if np.random.rand() > 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # randomly rotate +-15 degrees
        angle = np.random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle)
        
        # random crop: resize to 32x32 with random crop size between 0.9 and 1.0
        crop_size = np.random.uniform(0.9, 1.0)  # decide the crop size
        crop_width, crop_height = int(32 * crop_size), int(32 * crop_size)
        left = np.random.randint(0, 32 - crop_width)
        top = np.random.randint(0, 32 - crop_height)
        pil_img = pil_img.crop((left, top, left + crop_width, top + crop_height))
        pil_img = pil_img.resize((32, 32))  # adjust to 32x32

        # turn back to numpy array, normalize to [0, 1]
        augmented_images.append(np.array(pil_img) / 255.0)
    
    return np.array(augmented_images)

def preprocess_and_save(data_dir, save_dir, augment=False):
    # load CIFAR-10 data
    X_train, y_train = load_cifar10_train(data_dir)
    X_test, y_test = load_cifar10_test(data_dir)

    #  data augmentation
    if augment:
        print("Processing data augmentation...")
        X_train = augment_data(X_train)

    # one-hot encode labels
    print("Processing one-hot encoding...")
    y_train = encode_one_hot(y_train, 10)
    y_test = encode_one_hot(y_test, 10)

    # flatten images
    print("Processing flattening images...")
    X_train = X_train.reshape(X_train.shape[0], -1)  # (50000, 3072)
    X_test = X_test.reshape(X_test.shape[0], -1)    # (10000, 3072)

    # load data to local directory
    print("Loading processed data to local directory...")
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    
if __name__ == "__main__":
    # raw data directory
    data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data/cifar-10-batches-py"
    # processed data directory
    save_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"
    
    augment = True

    preprocess_and_save(data_dir, save_dir, augment=augment)
    print("Data prepocessing done!")