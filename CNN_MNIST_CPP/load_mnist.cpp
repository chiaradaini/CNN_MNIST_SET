#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<unsigned char>> read_mnist_images(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open MNIST image file: " << filename << std::endl;
        return {};
    }

    int magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    std::vector<std::vector<unsigned char>> images(num_images, std::vector<unsigned char>(num_rows * num_cols));

    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
    }

    return images;
}

std::vector<unsigned char> read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open MNIST label file: " << filename << std::endl;
        return {};
    }

    int magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    std::vector<unsigned char> labels(num_labels);

    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

void print_image(const std::vector<unsigned char>& image, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            unsigned char pixel = image[row * num_cols + col];
            std::cout << (pixel > 127 ? "*" : " ");
        }
        std::cout << std::endl;
    }
}

int main() {
    // Specify the file paths of the MNIST dataset
    std::string train_images_file = "train-images-idx3-ubyte.gz";
    std::string train_labels_file = "train-labels-idx1-ubyte.gz";
    std::string test_images_file = "t10k-images-idx3-ubyte.gz";
    std::string test_labels_file = "t10k-labels-idx1-ubyte.gz";

    // Read the MNIST images
    std::vector<std::vector<unsigned char>> train_images = read_mnist_images(train_images_file);
    std::vector<std::vector<unsigned char>> test_images = read_mnist_images(test_images_file);

    // Read the MNIST labels
    std::vector<unsigned char> train_labels = read_mnist_labels(train_labels_file);
    std::vector<unsigned char> test_labels = read_mnist_labels(test_labels_file);

    // Print the number of images and labels read
    std::cout << "Number of training images: " << train_images.size() << std::endl;
    std::cout << "Number of training labels: " << train_labels.size() << std::endl;
    std::cout << "Number of test images: " << test_images.size() << std::endl;
    std::cout << "Number of test labels: " << test_labels.size() << std::endl;

    // Print the first image and its label
    int image_index = 0;
    int num_rows = train_images[image_index].size() / 28;  // Assuming MNIST images are 28x28
    int num_cols = 28;

    std::cout << "Label: " << static_cast<int>(train_labels[image_index]) << std::endl;
    print_image(train_images[image_index], num_rows, num_cols);

    return 0;
}
