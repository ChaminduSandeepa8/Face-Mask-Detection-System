# Face Mask Detection System 😷

Hey everyone! 👋 Welcome to my real-time Face Mask Detection project. 

What started as a straightforward computer vision task quickly turned into an awesome learning curve about memory management and hardware limits. This system captures a live webcam feed, detects faces, and classifies them as wearing a mask or not in real-time.

## 🚀 What It Does
* **Real-time Detection:** Uses OpenCV's Haar Cascades to instantly lock onto faces via the webcam.
* **Live Classification:** Draws a Green bounding box for "Mask" and a Red one for "No Mask" along with the confidence score.
* **Under the Hood:** Powered by a custom-trained model built on top of the MobileNetV2 architecture using TensorFlow/Keras.

## 🧠 The Real Challenge: Beating Hardware Limits
Anyone can copy-paste a CNN model, but the real challenge in this project was dealing with the data. I trained this on a dataset of over 7,500 images, which led to some serious bottlenecks. Here is how I solved them:

**1. The "12GB RAM Crash" in Google Colab**
Trying to directly convert 7,500+ preprocessed images into a single NumPy array (`np.array`) kept crashing my Colab session. Instead of giving up, I bypassed this by pre-allocating an empty array (`np.empty`), transferring images iteratively, actively setting old list elements to `None`, and forcing Python's garbage collector (`gc.collect()`) to free up RAM on the fly.

**2. The `train_test_split` Memory Trap**
Passing a massive 4.5GB array into Scikit-Learn's split function duplicates the data, instantly maxing out memory again. I fixed this by splitting the *indices* first, creating empty `X_train` and `X_test` arrays, and mapping the data manually while actively deleting the source data in the process.

**3. Ubuntu Storage vs. Windows Inference**
I initially started on my Ubuntu partition but ran out of disk space when trying to install the heavy TensorFlow GPU libraries. My workaround? I decoupled the workflow: I used Colab's Cloud T4 GPU for the heavy lifting (training), exported the `.h5` model, and then switched to my Windows environment to run the live webcam inference with just the lightweight packages.

## 💻 How to Run It Locally

```markdown
## 💻 How to Run It Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/ChaminduSandeepa8/Face-Mask-Detection-System.git
   cd Face-Mask-Detection-System

```

2. **Install the dependencies:**
*(Note: You only need the standard packages for inference, no heavy GPU setup required!)*
```bash
pip install tensorflow opencv-python numpy

```


3. **Start the webcam:**
```bash
python detect_mask_video.py

```


*(Simply press the `q` key on your keyboard to exit the live camera feed).*

## 📁 Project Structure

* `detect_mask_video.py`: The main script to fire up the webcam and run predictions.
* `train_mask_detector.ipynb`: The Colab notebook containing all the memory-optimized training code.
* `mask_detector.h5`: The exported, ready-to-use Keras model.
