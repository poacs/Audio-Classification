def run_trained_model(X):
    import io
    import os
    import gdown
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import librosa
    import librosa.display
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # The class-to-label mapping must be exactly the same as in training
    CLASS_TO_LABEL = {
        'water': 0,
        'table': 1,
        'sofa': 2,
        'railing': 3,
        'glass': 4,
        'blackboard': 5,
        'ben': 6
    }

    # Model architecture must be the same as during training
    class CNNClassifier(nn.Module):
        def __init__(self, num_classes=7):
            super(CNNClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 32 * 32, num_classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x

    # Use the same transforms as in training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Function to convert a WAV file to a PNG image, following the logic used during training
    def wav_to_png_image(file_path, sr=22050, n_mels=128, hop_length=512):
        y, _ = librosa.load(file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Use the same plotting parameters as in training
        plt.figure(figsize=(10,4))
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram: {os.path.basename(file_path)}')
        plt.tight_layout()

        # Save the plot to memory instead of a file
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Load the PIL image from memory
        image = Image.open(buf).convert('RGB')
        return image

    def download_model_weights():
        # Update the URL or file ID as needed
        url = "https://drive.google.com/uc?id=1Z79uSqiK079hGhXPZKXYsXBqhhHUH18p"
        output = "model_weights.pth"
        if not os.path.exists(output):
            gdown.download(url, output, fuzzy=True)
        return output

    # Load model and weights
    weight_path = download_model_weights()
    model = CNNClassifier(num_classes=7)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()

    predictions = []
    with torch.no_grad():
        for file_path in X:
            # 1. Convert WAV to PNG image (in memory)
            image = wav_to_png_image(file_path)
            # 2. Apply the same transforms as during training
            input_tensor = transform(image).unsqueeze(0)  # [1, 3, 128, 128]
            # 3. Model inference
            outputs = model(input_tensor)
            pred = outputs.argmax(dim=1).item()
            predictions.append(pred)

    return np.array(predictions)
