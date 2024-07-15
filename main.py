import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

app = FastAPI()

class SimpleEmotionRecognitionModel(nn.Module):
    def __init__(self):
        super(SimpleEmotionRecognitionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(128 * 20 * 12, 256, batch_first=True)

        self.fc1 = nn.Linear(256, 7)
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        cnn_out = []
        for i in range(seq_len):
            out = self.cnn(x[:, i, :, :, :])
            out = out.view(batch_size, -1)
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)

        lstm_out, _ = self.lstm(cnn_out)
        lstm_out = lstm_out[:, -1, :]

        out1 = self.fc1(lstm_out)
        out2 = self.fc2(lstm_out)

        out1 = nn.Softmax(dim=1)(out1)
        out2 = nn.Softmax(dim=1)(out2)

        return out1, out2

# 클래스 이름 매핑
emotion_target_mapping = {
    0: "기쁨",
    1: "놀라움",
    2: "두려움",
    3: "사랑스러움",
    4: "슬픔",
    5: "화남",
    6: "없음"
}

def convert_predictions_to_strings(predictions):
    return [emotion_target_mapping[pred.item()] for pred in predictions]

model_path = 'model.pth'
model = SimpleEmotionRecognitionModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


def merge_audio_files(file_paths, output_path):
    combined = AudioSegment.empty()
    for file_path in file_paths:
        audio = AudioSegment.from_file(file_path)
        combined += audio
    combined.export(output_path, format="wav")


def create_spectrogram(file_path, output_dir):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    target_width = 1000
    target_height = 160

    S_width_interpolated = np.zeros((S_dB.shape[0], target_width))
    for i in range(S_dB.shape[0]):
        S_width_interpolated[i, :] = np.interp(
            np.linspace(0, S_dB.shape[1], target_width),
            np.arange(S_dB.shape[1]),
            S_dB[i, :]
        )

    S_interpolated = np.zeros((target_height, target_width))
    for i in range(target_width):
        S_interpolated[:, i] = np.interp(
            np.linspace(0, S_dB.shape[0], target_height),
            np.arange(S_dB.shape[0]),
            S_width_interpolated[:, i]
        )

    split_width = target_width // 10

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(10):
        start_idx = i * split_width
        end_idx = (i + 1) * split_width if i < 9 else target_width

        plt.figure(figsize=(1, 1.6))
        plt.imshow(S_interpolated[:, start_idx:end_idx], aspect='auto', origin='lower')
        plt.axis('off')
        output_file = os.path.join(output_dir, f"spectrogram_{i + 1}.png")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

def load_spectrograms(spectrogram_dir):
    spectrograms = []
    for i in range(10):
        image_path = os.path.join(spectrogram_dir, f"spectrogram_{i + 1}.png")
        image = Image.open(image_path).convert('L')
        image = image.resize((100, 160))
        image_array = np.array(image)
        spectrograms.append(image_array)
    spectrograms = np.array(spectrograms)
    spectrograms = spectrograms / 255.0
    spectrograms = spectrograms[:, np.newaxis, :, :]
    return spectrograms


@app.post("/ai/upload")
async def upload_audio(request: Request, audio_file: UploadFile = File(...)):
    session_id = request.headers.get("Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())

    user_dir = f"uploads/{session_id}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_count = len(os.listdir(user_dir))
    if file_count >= 10:
        raise HTTPException(status_code=400, detail="You have already uploaded 10 files.")

    file_path = os.path.join(user_dir, f"audio_{file_count + 1}.wav")
    with open(file_path, "wb") as f:
        f.write(await audio_file.read())

    response = {"message": f"File {file_count + 1} uploaded successfully.", "session_id": session_id}
    return JSONResponse(content=response)


@app.post("/ai/predict")
async def predict_audio(request: Request):
    session_id = request.headers.get("Session-ID")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session-ID header is missing.")

    user_dir = f"uploads/{session_id}"
    file_paths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.startswith("audio_")]
    if len(file_paths) != 10:
        raise HTTPException(status_code=400, detail="Please upload exactly 10 files.")

    combined_audio_path = os.path.join(user_dir, 'combined_audio.wav')
    output_dir = os.path.join(user_dir, 'output_spectrograms')

    merge_audio_files(file_paths, combined_audio_path)
    create_spectrogram(combined_audio_path, output_dir)
    spectrograms = load_spectrograms(output_dir)

    spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).to(
        'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, channels, height, width = spectrograms_tensor.size()
    spectrograms_tensor = spectrograms_tensor.view(batch_size // 10, 10, channels, height, width)

    out1, out2 = model(spectrograms_tensor)

    # 각 출력에서 확률이 가장 높은 클래스 선택
    _, preds1 = torch.max(out1, 1)
    _, preds2 = torch.max(out2, 1)

    # 클래스 인덱스를 문자열 라벨로 변환
    labels1 = convert_predictions_to_strings(preds1)
    labels2 = convert_predictions_to_strings(preds2)

    response = {
        "Predicted labels for Person 1": labels1,
        "Predicted labels for Person 2": labels2
    }
    return JSONResponse(content=response)


# FastAPI 실행
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)