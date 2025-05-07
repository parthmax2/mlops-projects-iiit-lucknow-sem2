
# Real-Time Sign Language Recognition using TensorFlow Object Detection API
    
## Project Overview


This project focuses on real-time recognition of Indian Sign Language (ISL) using a webcam feed. It leverages the TensorFlow Object Detection API and a custom-trained SSD MobileNet V2 model to detect and classify hand gestures corresponding to different ISL signs.



## Model Architecture

We use the **SSD MobileNet V2** architecture:
- **Backbone**: MobileNet V2 for lightweight and fast feature extraction.
- **Detector**: Single Shot MultiBox Detector (SSD) to predict object bounding boxes and classes in one forward pass.
- **Pipeline**: Pre-trained model fine-tuned on a custom ISL dataset.

**Why SSD MobileNet V2?**
- **Real-time Performance**: Optimized for speed and low computational load.
- **Good Accuracy-Speed Tradeoff**: Performs better on CPUs, suitable for real-time detection.
- **Pretrained on COCO Dataset**: Easy transfer learning.

### Alternative Models Considered
- **YOLO**: Faster, but requires higher GPU processing, and tuning is more complex.
- **Faster R-CNN**: Very accurate but too slow for real-time on CPU.
- **SSD with Inception**: Heavier than MobileNet, which affects FPS during inference.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/RealTimeSignRecognition.git
cd RealTimeSignRecognition
```

### 2. Create and Activate a Virtual Environment
```bash
conda create -n tflod python=3.10
conda activate tflod
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install TensorFlow Object Detection API
Refer to [TFOD API installation guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) or use the included script.

## Dataset Preparation


- Hand sign images are organized into `images/train` and `images/test`.
- Annotations are generated using LabelImg and converted to TFRecord format.
- Label map is defined in: `Tensorflow/workspace/annotations/label_map.pbtxt`.

Example:
```text
item {
  id: 1
  name: 'A'
}
item {
  id: 2
  name: 'B'
}
```

---

##  Model Training

1. Modify the `pipeline.config` file:
   - Set paths for `train_record`, `test_record`, `label_map`, etc.

2. Train the model:
```bash
python model_main_tf2.py --model_dir=models/my_ssd_mobnet --pipeline_config_path=models/my_ssd_mobnet/pipeline.config
```

3. Export the trained model:
```bash
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_ssd_mobnet/pipeline.config --trained_checkpoint_dir models/my_ssd_mobnet --output_directory exported-model
```

---

##  Real-Time Inference with Streamlit

Run this command:
```bash
streamlit run streamlit_app.py
```

- A checkbox is provided to activate the camera.
- When you show a sign, it detects and labels the gesture in real time.

---

##  Deployment Plan

- Web-based deployment using Streamlit.
- Can later be deployed to the cloud using Streamlit sharing or platforms like:
  - **Heroku**
  - **Streamlit Community Cloud**
  - **Azure / GCP / AWS EC2**

---

##  FAQs / Cross-Question Preparation

**Q: Why SSD MobileNet over YOLO or Faster R-CNN?**  
A: SSD MobileNet offers a better speed-accuracy tradeoff for CPU-based systems, enabling real-time detection with lower latency.

**Q: How is feature extraction performed?**  
A: MobileNetV2 uses depthwise separable convolutions to extract features from the image while keeping the model lightweight.

**Q: Can ROI (Region of Interest) be applied?**  
A: Yes, ROI can be applied to focus detection within specific bounding boxes, though the current model detects from the entire frame.

**Q: What are the limitations?**  
- Limited dataset
- Background noise or lighting variations can affect accuracy

---

##  Acknowledgements

- TensorFlow Team
- Official TensorFlow Object Detection API
- Indian Sign Language Dataset



