import sys
import os
import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from PyQt5 import QtWidgets, QtCore, QtGui

# --- Aquí puedes cambiar a TensorFlow si lo prefieres ---
_USE_TORCH = True
if _USE_TORCH:
    import torch
    import torch.nn as nn
    import torch.optim as optim
else:
    import tensorflow as tf
    from tensorflow import keras

# ---- Modelo de ejemplo (Autoencoder) ----
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

# ---- OTO Entry ----
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

# ---- SDK lógica ----
class VoicebankSDK:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.oto_ini = self.root_dir / "oto.ini"
        self.info_json = self.root_dir / "info.json"
        self.entries = []
        self.voice_samples = {}
        self.sample_rate = None
        self.name = None
        self.author = None
        self.description = None

    def write_info_json(self):
        meta = {
            "name": self.name,
            "author": self.author,
            "description": self.description,
            "hlvb_version": "1.0"
        }
        with open(self.info_json, "w", encoding="utf8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def parse_oto_ini(self):
        if not self.oto_ini.exists():
            raise FileNotFoundError(f"oto.ini no encontrado en {self.root_dir}")
        self.entries = []
        with open(self.oto_ini, encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    entry = OtoEntry.from_line(line, base_path=str(self.root_dir))
                    self.entries.append(entry)

    def check_samples(self):
        missing = []
        for entry in self.entries:
            if not os.path.isfile(entry.wav_path):
                missing.append(entry.wav_path)
        return missing

    def load_samples(self, sample_len=16000):
        for entry in self.entries:
            data, sr = sf.read(entry.wav_path)
            if self.sample_rate is None:
                self.sample_rate = sr
            elif sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: {entry.wav_path} ({sr} != {self.sample_rate})")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # mono
            if len(data) < sample_len:
                data = np.pad(data, (0, sample_len - len(data)))
            else:
                data = data[:sample_len]
            self.voice_samples[entry.alias] = data.astype(np.float32)

    def train_voice_model(self, epochs=15, out_model="voice.pth"):
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
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = loss_fn(output, X_tensor)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), self.root_dir / out_model)

    def _train_tensorflow(self, X, epochs, out_model):
        input_dim = X.shape[1]
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(input_dim, activation="tanh")
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=epochs, verbose=1)
        model.save(self.root_dir / "voice_tf_model")
        with open(self.root_dir / "voice_tf_model.flag", "w") as f:
            f.write("tensorflow")

    def pack_hlvb(self, out_path=None):
        if out_path is None:
            out_path = self.root_dir.parent / (self.root_dir.name + ".hlvb")
        else:
            out_path = Path(out_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(self.info_json, "info.json")
            zf.write(self.oto_ini, "oto.ini")
            model_path = self.root_dir / "voice.pth"
            if model_path.exists():
                zf.write(model_path, "voice.pth")
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
        return out_path

# --- Interfaz Kawaii con PyQt5 ---
class KawaiiVoicebankGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HiroshiLOID Official SDK 🎵")
        self.setStyleSheet("background: #f5f7fa;")

        # Kawaii Title
        title = QtWidgets.QLabel("🎀 HLVB Voicebank Creator 🎀")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4;")

        # Inputs
        self.voicebank_dir_edit = QtWidgets.QLineEdit()
        self.voicebank_dir_edit.setPlaceholderText("Selecciona la carpeta del voicebank...")
        self.voicebank_dir_edit.setStyleSheet("font-size: 18px; border-radius: 10px; background: #fffde7;")
        dir_btn = QtWidgets.QPushButton("Elegir carpeta")
        dir_btn.setStyleSheet("background: #ffe4ec; border-radius: 16px; font-size: 16px;")
        dir_btn.clicked.connect(self.choose_dir)

        name_label = QtWidgets.QLabel("Nombre kawaii del voicebank:")
        name_label.setStyleSheet("font-size: 16px; color: #ff69b4;")
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setStyleSheet("font-size: 16px; border-radius: 10px;")

        author_label = QtWidgets.QLabel("Autor:")
        author_label.setStyleSheet("font-size: 16px; color: #ff69b4;")
        self.author_edit = QtWidgets.QLineEdit()
        self.author_edit.setStyleSheet("font-size: 16px; border-radius: 10px;")

        desc_label = QtWidgets.QLabel("Descripción:")
        desc_label.setStyleSheet("font-size: 16px; color: #ff69b4;")
        self.desc_edit = QtWidgets.QLineEdit()
        self.desc_edit.setStyleSheet("font-size: 16px; border-radius: 10px;")

        self.status = QtWidgets.QLabel("¡Bienvenida! Elige tu carpeta kawaii y completa los datos. (◕‿◕✿)")
        self.status.setStyleSheet("font-size: 16px; color: #a7a7a7;")

        self.oto_table = QtWidgets.QTableWidget()
        self.oto_table.setColumnCount(2)
        self.oto_table.setHorizontalHeaderLabels(["Alias", "Archivo WAV"])
        self.oto_table.setStyleSheet("background: #fffde7; font-size: 14px; border-radius: 8px;")
        self.oto_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.oto_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        train_btn = QtWidgets.QPushButton("Entrenar y crear HLVB (✿◠‿◠)")
        train_btn.setStyleSheet("background: #baffc9; border-radius: 16px; font-size: 18px;")
        train_btn.clicked.connect(self.train_and_pack)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(name_label, self.name_edit)
        form.addRow(author_label, self.author_edit)
        form.addRow(desc_label, self.desc_edit)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(title)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.voicebank_dir_edit)
        hbox.addWidget(dir_btn)
        vbox.addLayout(hbox)
        vbox.addLayout(form)
        vbox.addWidget(self.oto_table)
        vbox.addWidget(self.status)
        vbox.addWidget(train_btn)

        self.sdk = None

    def choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del voicebank")
        if d:
            self.voicebank_dir_edit.setText(d)
            self.load_oto(d)

    def load_oto(self, d):
        try:
            self.sdk = VoicebankSDK(d)
            self.sdk.parse_oto_ini()
            self.status.setText(f"OTO.ini cargado con {len(self.sdk.entries)} fonemas kawaii.")
            self.oto_table.setRowCount(len(self.sdk.entries))
            for i, entry in enumerate(self.sdk.entries):
                self.oto_table.setItem(i, 0, QtWidgets.QTableWidgetItem(entry.alias))
                self.oto_table.setItem(i, 1, QtWidgets.QTableWidgetItem(os.path.basename(entry.wav_path)))
            missing = self.sdk.check_samples()
            if missing:
                self.status.setText("⚠️ Faltan archivos WAV:\n" + "\n".join([os.path.basename(m) for m in missing]))
        except Exception as e:
            self.status.setText("Error al cargar oto.ini: " + str(e))

    def train_and_pack(self):
        if not self.sdk:
            self.status.setText("Cargar primero el voicebank kawaii.")
            return
        self.sdk.name = self.name_edit.text().strip()
        self.sdk.author = self.author_edit.text().strip()
        self.sdk.description = self.desc_edit.text().strip()
        if not (self.sdk.name and self.sdk.author):
            self.status.setText("Por favor completa nombre y autor.")
            return
        self.sdk.write_info_json()
        missing = self.sdk.check_samples()
        if missing:
            self.status.setText("No se puede entrenar: faltan WAV.")
            return
        try:
            self.sdk.load_samples()
            self.status.setText("Entrenando modelo (esto puede tardar un poquito, ten paciencia kawaii)...")
            QtWidgets.QApplication.processEvents()
            self.sdk.train_voice_model()
            out_file = self.sdk.pack_hlvb()
            self.status.setText(f"¡Voicebank HLVB creado con éxito! Archivo: {str(out_file)}")
        except Exception as e:
            self.status.setText("Error durante el entrenamiento/empaquetado: " + str(e))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = KawaiiVoicebankGUI()
    w.resize(720, 600)
    w.show()
    sys.exit(app.exec_())
