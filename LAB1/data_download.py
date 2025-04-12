import os
import tarfile
import urllib.request

def download_cifar10(data_dir):
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "cifar-10-python.tar.gz"
    file_path = os.path.join(data_dir, file_name)

    # Create the target directory if it does not exist
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset
    if not os.path.exists(file_path):
        print(f"Downloading CIFAR-10 dataset to {file_path}...")
        urllib.request.urlretrieve(cifar10_url, file_path)
        print("Download complete.")
    else:
        print(f"CIFAR-10 dataset already exists at {file_path}.")

    # Extract the dataset
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        print(f"Extracting CIFAR-10 dataset to {extracted_dir}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    else:
        print(f"CIFAR-10 dataset is already extracted at {extracted_dir}.")

    print("Dataset is ready.")


if __name__ == "__main__":
    # Replace with your desired directory path
    data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data" 
    
    download_cifar10(data_dir)
