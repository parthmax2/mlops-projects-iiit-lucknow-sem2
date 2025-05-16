Eye_Drowsiness_Detection/
├── data/
│   ├── train/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── valid/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── test/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   ├── data.yaml  # Data configuration for training
├── model/
│   ├── last.pt   # Trained model file
│   └── shape_predictor.dat  # Pre-trained dlib shape predictor
├── src/
│   ├── detection.py 
│   ├── prediction.py  # Real-time detection script for webcam
├── app.py  # Streamlit or Flask web app for deployment
├── requirements.txt  # List of dependencies for the project
└── README.md  # Project documentation
