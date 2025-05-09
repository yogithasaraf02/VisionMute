# VisionMute: Wink & Shush Detection

**VisionMute** is a cutting-edge computer vision project built using Python and OpenCV, designed to detect two specific facial gestures — **wink** and **shush**. The project leverages machine learning techniques and Haar Cascade classifiers to identify these gestures in images and videos.

## Project Overview

The project is divided into two primary detection functionalities:

1. **Wink Detection**:
   - This feature detects when one eye is closed (a common gesture for a wink) in images and videos.
   
2. **Shush Detection**:
   - This feature detects the "shush" gesture, commonly represented by a closed mouth and a finger or hand near the lips.

The system processes both images and videos and can be used for real-time or batch processing of image files.

---

## Key Features

- **Wink Detection**: Uses OpenCV to detect a wink by identifying the closed eye in the image.
- **Shush Detection**: Identifies when the mouth is closed and the user is signaling the "shush" gesture.
- Supports **image files** (JPEG, PNG) and **video files** for processing.
- Bulk image processing from **ZIP files** for efficient detection of gestures across multiple images.
- **Real-time video processing**: You can use a webcam for live wink and shush detection.
- **Detection Summary**: After processing, the system provides a total count of detected gestures across the images or video frames.

---

## Requirements

To run this project, you'll need the following Python libraries:

- **Python 3.x**: Ensure you have Python 3.x installed.
- **OpenCV**: For computer vision tasks.
- **NumPy**: For numerical operations.
- **Matplotlib**: For image display.
- **Other dependencies**: All other dependencies are listed in the `requirements.txt` file.

### Installing Dependencies

1. Install Python 3.x on your system if you don’t have it already. You can download it from [python.org](https://www.python.org/).

2. Install the necessary dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### 1. Upload a ZIP file with images
You can upload a ZIP file containing multiple images for batch processing. The system will:
- Detect the wink and/or shush gestures.
- Highlight the face and gesture with bounding boxes on the images.
- Display the result image and provide a count of the detected gestures.

### 2. Webcam / Live Video Processing
You can also run the system on live webcam footage for real-time wink and shush detection. The system will display the webcam feed and process it frame by frame to detect any gestures.

### How to Run the Code

**For Wink and Shush Detection in Images:**
1. Place your image files inside a ZIP file.
2. Upload the ZIP file using the provided interface or manually upload files into the directory.
3. The system will automatically detect and highlight the gestures.

**For Live Webcam Processing:**
Run the following command to start the webcam feed and get live wink and shush detection:

```bash
python your_script.py
