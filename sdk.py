import os
import json
import zipfile
from pathlib import Path
import soundfile as sf
import numpy as np

# Cambia esto a False si prefieres TensorFlow
_USE_TORCH = True

if _USE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
else:
    import tensorflow as tf
    from tensorflow import keras

class OtoEntry:
    def __init__(self, wav_path, alias, offset, consonant, cutoff, preutterance, overlap):
        self.wav_path = wav_path
        self.alias = alias
        self.offset = float(offset)
        self.consonant = float(consonant)
        self.cutoff = float(cutoff)
        self.preutterance = float(preutterance)
        self.overlap = float(overlap)

    @staticmethod
    def from_line(line: str, base_path: str = "."):
        wav, rest = line.strip().split('=', 1)
        fields = rest.split(',')
        if len(fields) != 6:
            raise ValueError("Invalid oto.ini entry: " + line)
        wav_path = os.path.join(base_path, wav)
        return OtoEntry(wav_path, *fields)

class SimpleAutoencoderTorch(nn.Module):
    def __init__(self, input_dim=16000, bottleneck=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class VoicebankSDK:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.oto_ini = self.root_dir / "oto.ini"
        self.info_json = self.root_dir / "info.json"
        self.entries = []
        self.voice_samples = {}
        self.sample_rate = None
        self.name = None

    def setup_voicebank(self):
        print("=== Creación de Voicebank HLVB ===")
        self.name = input("Nombre del voicebank: ").strip()
        author = input("Autor: ").strip()
        description = input("Descripción: ").strip()
        self._write_info_json(self.name, author, description)

    def _write_info_json(self, name, author, description):
        meta = {
            "name": name,
            "author": author,
            "description": description,
            "hlvb_version": "1.0"
        }
        with open(self.info_json, "w", encoding="utf8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print("info.json creado.")

    def parse_oto_ini(self):
        if not self.oto_ini.exists():
            raise FileNotFoundError(f"oto.ini no encontrado en {self.root_dir}")
        self.entries = []
        with open(self.oto_ini, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    entry = OtoEntry.from_line(line, base_path=str(self.root_dir))
                    self.entries.append(entry)
        print(f"{len(self.entries)} entradas OTO cargadas.")

    def check_samples(self):
        missing = []
        for entry in self.entries:
            if not os.path.isfile(entry.wav_path):
                missing.append(entry.wav_path)
        if missing:
            print("Faltan los siguientes archivos WAV:")
            for f in missing:
                print(" -", f)
            return False
        print("Todos los samples del oto.ini están presentes.")
        return True

    def load_samples(self, sample_len=16000):
        for entry in self.entries:
            data, sr = sf.read(entry.wav_path)
            if self.sample_rate is None:
                self.sample_rate = sr
            elif sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: {entry.wav_path} ({sr} != {self.sample_rate})")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # mono
            # Normaliza longitud
            if len(data) < sample_len:
                data = np.pad(data, (0, sample_len - len(data)))
            else:
                data = data[:sample_len]
            self.voice_samples[entry.alias] = data.astype(np.float32)
        print(f"Se cargaron {len(self.voice_samples)} samples, cada uno de {sample_len} muestras.")

    def train_voice_model(self, epochs=15, out_model="voice.pth"):
        print("=== Entrenamiento real del modelo Autoencoder ===")
        sample_len = 16000
        X = np.stack([self.voice_samples[a] for a in self.voice_samples])
        if _USE_TORCH:
            self._train_torch(X, epochs, out_model)
        else:
            self._train_tensorflow(X, epochs, out_model)

    def _train_torch(self, X, epochs, out_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        model = SimpleAutoencoderTorch(input_dim=X.shape[1], bottleneck=256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        model.train()
        print("Entrenando modelo Autoencoder (PyTorch)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = loss_fn(output, X_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")
        torch.save(model.state_dict(), self.root_dir / out_model)
        print(f"Modelo PyTorch guardado en {out_model}.")

    def _train_tensorflow(self, X, epochs, out_model):
        input_dim = X.shape[1]
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(input_dim, activation="tanh")
        ])
        model.compile(optimizer='adam', loss='mse')
        print("Entrenando modelo Autoencoder (TensorFlow)...")
        model.fit(X, X, epochs=epochs, verbose=1)
        model.save(self.root_dir / "voice_tf_model")
        with open(self.root_dir / "voice_tf_model.flag", "w") as f:
            f.write("tensorflow")
        print("Modelo TensorFlow guardado en voice_tf_model/.")

    def pack_hlvb(self, out_path=None):
        if out_path is None:
            out_path = self.root_dir.parent / (self.root_dir.name + ".hlvb")
        else:
            out_path = Path(out_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.info_json, "info.json")
            zf.write(self.oto_ini, "oto.ini")
            # PyTorch
            model_path = self.root_dir / "voice.pth"
            if model_path.exists():
                zf.write(model_path, "voice.pth")
            # TensorFlow
            tf_folder = self.root_dir / "voice_tf_model"
            if tf_folder.exists():
                for foldername, subfolders, filenames in os.walk(tf_folder):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        arcname = os.path.relpath(file_path, self.root_dir)
                        zf.write(file_path, arcname)
                flag = self.root_dir / "voice_tf_model.flag"
                if flag.exists():
                    zf.write(flag, "voice_tf_model.flag")
            if not model_path.exists() and not tf_folder.exists():
                print("Advertencia: voice.pth/voice_tf_model no encontrado, sólo se empaquetarán info.json y oto.ini")
        print(f"HLVB creado en: {out_path}")

    def create_and_train(self):
        self.setup_voicebank()
        self.parse_oto_ini()
        if not self.check_samples():
            print("Por favor agrega los samples faltantes a la carpeta y vuelve a intentarlo.")
            return
        self.load_samples()
        self.train_voice_model()
        self.pack_hlvb()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HLVB SDK - Creador de voicebanks neural UTAU (.hlvb)")
    parser.add_argument("voicebank_dir", help="Carpeta del voicebank (con oto.ini y samples WAV)")
    parser.add_argument("--tensorflow", action="store_true", help="Entrenar usando TensorFlow en vez de PyTorch")
    args = parser.parse_args()
    if args.tensorflow:
        _USE_TORCH = False
    sdk = VoicebankSDK(args.voicebank_dir)
    sdk.create_and_train()
