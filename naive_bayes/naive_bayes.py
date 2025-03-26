import numpy as np
import cv2
import os

# Load dataset from images
def load_data(image_folder, mask_folder):
    X = []
    y = []

    image_files = os.listdir(image_folder)
    print(image_files)

    # for image_file in image_files:
    #     image_path = os.path.join(image_folder, image_file)
    #     image_path = image_path.replace("\\","/")
    #     mask_path = os.path.join(mask_folder, image_file)  # Assuming same filename
    #     mask_path = mask_path.replace("\\","/")

    #     image = cv2.imread(image_path)
    #     mask = cv2.imread(mask_path)  # Load mask in grayscale

    #     if image is None or mask is None:
    #         print(f"Skipping {image_file} due to loading error.")
    #         continue

    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    #     # Extract skin and non-skin pixels
    #     skin_pixels = image[mask == 255]  # Skin is white (255 in mask)
    #     non_skin_pixels = image[mask == 0]  # Non-skin is black (0 in mask)

    #     X.extend(skin_pixels)
    #     y.extend([1] * len(skin_pixels))  # Label skin as 1
    #     X.extend(non_skin_pixels)
    #     y.extend([0] * len(non_skin_pixels))  # Label non-skin as 0

    # return np.array(X), np.array(y)

# Compute Gaussian parameters
def find_parameters(X, y):
    skin_pixels = X[y == 1]
    non_skin_pixels = X[y == 0]

    skin_mean = np.mean(skin_pixels, axis=0)
    skin_var = np.var(skin_pixels, axis=0)
    
    non_skin_mean = np.mean(non_skin_pixels, axis=0)
    non_skin_var = np.var(non_skin_pixels, axis=0)

    prior_skin = len(skin_pixels) / len(X)
    prior_non_skin = len(non_skin_pixels) / len(X)
    
    return (skin_mean, skin_var), (non_skin_mean, non_skin_var), prior_skin, prior_non_skin

# Gaussian probability density function
def gaussian_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

# Predict skin or non-skin for each pixel
def predict_skin_or_non_skin(image, skin_mean, skin_var, non_skin_mean, non_skin_var, prior_skin, prior_non_skin):
    height, width, _ = image.shape
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            prob_skin = np.prod([gaussian_pdf(pixel[k], skin_mean[k], skin_var[k]) for k in range(3)]) * prior_skin
            prob_non_skin = np.prod([gaussian_pdf(pixel[k], non_skin_mean[k], non_skin_var[k]) for k in range(3)]) * prior_non_skin

            result[i, j] = 255 if prob_skin > prob_non_skin else 0  # 255 for skin, 0 for non-skin

    return result

# Main function to test with an image
def test_skin_detection(image_path, skin_params, non_skin_params, prior_skin, prior_non_skin):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    result = predict_skin_or_non_skin(image, *skin_params, *non_skin_params, prior_skin, prior_non_skin)

    cv2.imshow("Original Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Skin Detection Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Execute training and testing in one script
if __name__ == '__main__':
    image_folder = './imageData/Pratheepan_Dataset/FacePhoto'  # Replace with the actual path
    mask_folder = './imageData/Ground_Truth/GroundT_FacePhoto'  # Replace with the actual path
    load_data(image_folder, mask_folder)
    # print(X)
    # skin_params, non_skin_params, prior_skin, prior_non_skin = find_parameters(X, y)

    # test_image_path = './Images/ProfilePic.jpg'  # Replace with the actual image path
    # test_skin_detection(test_image_path, skin_params, non_skin_params, prior_skin, prior_non_skin)
