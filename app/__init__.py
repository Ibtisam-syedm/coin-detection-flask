from flask import Flask, request
from PIL import Image
import cv2
from flask_cors import CORS
import numpy as np
from sklearn import svm
from skimage.feature import hog
import os
import joblib


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

clf = svm.SVC(decision_function_shape='ovo')

@app.route('/process-image', methods=['POST'])
def process_image():
    # Retrieve the uploaded image
    total = 0
    image = request.files['image']
    image = Image.open(image)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    preprocessedImage = preprocessing(image)
    if preprocessedImage is not None:
        # detecting the circles
        circles = detecting_circles(preprocessedImage)
        if circles == "Error":
            print("error occurred")
        elif circles is None:
            print("No circles detected")
        else:
            # coins matching
            # image = load_image_in_gray(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            image = adaptive_histogram_processing(image)
            total = coins_matching(circles, image)
            print("total :", total)
    else:
        print("failed to preprocess image")
    # Process the image (your code goes here)
    # ...

    # Return the integer result
    # result = 42
    return str(total)


def predict_coin(grayImage):
    # Extract HOG features from the preprocessed image
    input_features = hog(grayImage, orientations=11,
                         pixels_per_cell=(12, 12), cells_per_block=(1, 1))
    # input_features = scaler.transform(input_features)
    input_features = input_features.reshape(1, -1)
    # Later, when you want to make predictions, load the model from disk
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'svm_model.pkl')
    loaded_model = joblib.load(model_path)
    # loaded_model = joblib.load('svm_model.pkl')

    # Use the loaded model to make predictions
    predicted_class = loaded_model.predict(input_features)
    # predicted_class = clf.predict(input_features)
    return predicted_class


def adaptive_thresholding(grayImage):
    # Adaptive thresholding in order to present better results
    maximum_value_to_assign = 255
    local_neighbourhood_window = 9
    adjust_contrast = 3  # to subtract from mean value
    return cv2.adaptiveThreshold(grayImage,
                                 maximum_value_to_assign,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 local_neighbourhood_window,
                                 adjust_contrast)


def adaptive_histogram_processing(grayImage):
    # Local Histogram processing to enhance the image locally by dividing into small segments
    contrast_limiting_threshold = 3.0
    ahp = cv2.createCLAHE(
        clipLimit=contrast_limiting_threshold, tileGridSize=(8, 8))
    # Applying to grayscale image
    return ahp.apply(grayImage)


def load_image_in_gray(path):
    try:
        image = cv2.imread(path)
        if image is not None:
            # Convert the image to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print('Error:', e)
    return None


def gaussian_blur(grayImage):
    # gaussian blur, to reduce noise in the image
    controlBlurness = 0
    kernel_size = 15
    blur = cv2.GaussianBlur(
        grayImage, (kernel_size, kernel_size), controlBlurness)
    return blur


def hough_circle(blurred_image):
    return cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,  # minimum distance between center of detected circles
        param1=50,  # to detect edge, canny edge
        param2=35,  # smaller value, results in false circles
        minRadius=10,  # min radius of circle to detect
        maxRadius=120,  # max radius of circle to detect
    )

def detecting_circles(preprocessedImage):
    # Used Hough transform function to find out all the circles in the image
    try:
        circles = hough_circle(preprocessedImage)
        # image = cv2.imread(path)
        totalCirclesDetected = 0
        if circles is not None:
            # rounding to nearest integer and converting to int
            circles = np.round(circles[0, :]).astype("int")
            circles = filtered_circles(circles)

            # for (x, y, r) in circles:
            #     totalCirclesDetected += 1
            #     # Draw the circle and its center
            #     cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            #     cv2.circle(image, (x, y), 1, (255, 0, 0), 2)

            # Display the output image
            # cv2.imshow("Detected Circles", image)
            # cv2.waitKey(0)
            print("total Circles Detected:", totalCirclesDetected)
        else:
            print("No circles detected.")
        # print(circles)

        return circles
    except Exception as e:
        print('Error:', e)
        return "Error"


def circles_overlap(circle1, circle2, overlap_threshold):
    (x1, y1, r1) = circle1
    (x2, y2, r2) = circle2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance < (r1 + r2) * overlap_threshold


def filtered_circles(circles):
    overlap_threshold = 0.8  # less than 1 means that circles can overlap
    filtered_circles = []
    for (x, y, r) in circles:
        overlap = False
        for (x2, y2, r2) in filtered_circles:
            if circles_overlap((x, y, r), (x2, y2, r2), overlap_threshold):
                overlap = True
                break
        if not overlap:
            filtered_circles.append((x, y, r))
    return filtered_circles


# input blurred image from preprocessing
def extract_coin_shape(gray_image, center, radius):
    x, y = center
    r = int(radius * 1.1)
    return gray_image[y - r: y + r, x - r: x + r]


def coins_matching(circles, preprocessedImage):
    total = 0
    for (x, y, r) in circles:
        coin_shape = extract_coin_shape(preprocessedImage, (x, y), r)
        # display_image("coin",coin_shape)
        input_image_gray = cv2.resize(coin_shape, (128, 128))
        predicted_class = predict_coin(input_image_gray)
        print(f"Predicted class: {predicted_class}")
        if predicted_class[0].isdigit():
            total += int(predicted_class[0])

    return total


def compute_distance_btw_moments(m1, m2):
    # Euclidean distance between the Hu moments
    return np.sqrt(np.sum((m1 - m2) ** 2))


def preprocessing(image):
    try:
        # image = load_image_in_gray(image)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            image = adaptive_histogram_processing(image)
            # Apply Gaussian blur to reduce noise
            image = gaussian_blur(image)
            # Apply Canny edge detection
            # image = cannyEdge(image)
            return image
        else:
            print("failed to load image for preprocesing")
            return None
    except Exception as e:
        print('Error:', e)

    return None