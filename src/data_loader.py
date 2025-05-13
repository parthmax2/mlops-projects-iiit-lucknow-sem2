import os
import json
import math
import librosa

# Constants
DATASET_PATH = r"C:\Users\user\OneDrive\Desktop\ML\Mlops\music_genre_prediction_mlops_project_MSA24025\data\Data\genres_original"
OUTPUT_PATH = r"C:\Users\user\OneDrive\Desktop\ML\Mlops\music_genre_prediction_mlops_project_MSA24025\data\data.json"

SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """
    Extract MFCCs from audio files, skip unreadable files, and save to JSON.

    :param dataset_path: Path to dataset with genre subfolders
    :param json_path: Output path for MFCC data (JSON)
    :param num_mfcc: Number of MFCC coefficients
    :param n_fft: FFT window size
    :param hop_length: Hop length for FFT
    :param num_segments: Number of segments to split each track into
    """
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors = math.ceil(samples_per_segment / hop_length)

    # Traverse genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            continue

        genre_label = os.path.basename(dirpath)
        data["mapping"].append(genre_label)
        print(f"\nProcessing: {genre_label}")

        for f in filenames:
            file_path = os.path.join(dirpath, f)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue

            for s in range(num_segments):
                start = samples_per_segment * s
                end = start + samples_per_segment

                mfcc = librosa.feature.mfcc(
                    y=signal[start:end],
                    sr=sr,
                    n_mfcc=num_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length
                )
                mfcc = mfcc.T

                if len(mfcc) == expected_num_mfcc_vectors:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i - 1)

    # Save to JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        print(f"\nMFCCs saved to: {json_path}")


# Run it
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, OUTPUT_PATH)
