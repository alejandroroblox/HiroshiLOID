#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>
#include <QTableWidget>
#include <QHeaderView>
#include <QSplitter>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QMouseEvent>
#include <QProcess>
#include <QTemporaryFile>
#include <cmath>
#include <vector>
#include <algorithm>
#include <optional>
#include <QDir>
#include "Voicebank.h"
#include "WavUtils.h"
#include "PhonemeEntry.h"

// --- ONNX Runtime Integration ---
#include <onnxruntime_cxx_api.h>

// --- Piano Roll Implementation (VOCALOID6 style) ---
// ... (tu implementación igual que antes, omitida por brevedad)
// --- End Piano Roll Implementation ---

struct PianoNote {
    int pitch;           // MIDI note number
    int start;           // In "ticks" (e.g. 480 = quarter note)
    int length;          // In "ticks"
    QString lyric;       // Phoneme or syllable
};

// Helper: Convert .pth (PyTorch) to .onnx (for this demo, requires prior conversion)
QString pthToOnnx(const QString& pthPath) {
    // Busca un .onnx con el mismo nombre en el mismo folder, o lanza advertencia
    QString onnxPath = pthPath;
    if (onnxPath.endsWith(".pth"))
        onnxPath.replace(".pth", ".onnx");
    if (QFile::exists(onnxPath))
        return onnxPath;
    QMessageBox::warning(nullptr, "ONNX Runtime", "No se encontró el archivo .onnx correspondiente al modelo .pth.\nConvierte primero tu modelo .pth a .onnx (por ejemplo, usando torch.onnx.export en Python).");
    return "";
}

// --- IA Voice Synthesis: ONNX Runtime (requiere .onnx del modelo IA) ---
bool exportVoiceAIWav_ONNX(const QString& filename, const std::vector<PianoNote>& notes, const QString& onnxPath, int bpm = 120, int sampleRate = 44100) {
    if (!QFile::exists(onnxPath)) return false;
    try {
        // Inicializa ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "kawaii-onnx");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        Ort::Session session(env, onnxPath.toStdString().c_str(), session_options);

        // Prepara entrada: convierte las notas a un arreglo de floats (esto depende de tu modelo)
        // Ejemplo: [start, length, pitch, lyric_id, ...] para cada nota
        // Aquí usamos solo pitch y duración; para un modelo real, adapta a lo que tu modelo espera
        std::vector<float> note_feats;
        for (const auto& n : notes) {
            note_feats.push_back(float(n.start));
            note_feats.push_back(float(n.length));
            note_feats.push_back(float(n.pitch));
            // Si tu modelo usa IDs de fonema, aquí deberías mapear n.lyric a un ID
        }

        // Prepara tensores de entrada
        std::vector<int64_t> note_shape = { int64_t(notes.size()), 3 }; // [num_notes, 3]
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names = { session.GetInputName(0, allocator) };
        std::vector<const char*> output_names = { session.GetOutputName(0, allocator) };

        Ort::MemoryInfo mem_info("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, note_feats.data(), note_feats.size(), note_shape.data(), note_shape.size());

        // Ejecuta inferencia
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        // Recupera salida: asume que el modelo devuelve un array [num_samples] de float32 (PCM -1..1)
        float* audio_data = output_tensors[0].GetTensorMutableData<float>();
        size_t num_samples = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // Normalizar y guardar como WAV PCM
        QFile file(filename);
        if (!file.open(QIODevice::WriteOnly)) return false;
        int dataChunkSize = int(num_samples) * 2;
        int fileSize = 44 + dataChunkSize;
        file.write("RIFF", 4);
        file.write(reinterpret_cast<const char*>(&fileSize), 4);
        file.write("WAVEfmt ", 8);
        int fmtSize = 16;
        file.write(reinterpret_cast<const char*>(&fmtSize), 4);
        int16_t audioFormat = 1, numChannels = 1, bitsPerSample = 16;
        file.write(reinterpret_cast<const char*>(&audioFormat), 2);
        file.write(reinterpret_cast<const char*>(&numChannels), 2);
        file.write(reinterpret_cast<const char*>(&sampleRate), 4);
        int byteRate = sampleRate * numChannels * bitsPerSample / 8;
        file.write(reinterpret_cast<const char*>(&byteRate), 4);
        int16_t blockAlign = numChannels * bitsPerSample / 8;
        file.write(reinterpret_cast<const char*>(&blockAlign), 2);
        file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);
        file.write("data", 4);
        file.write(reinterpret_cast<const char*>(&dataChunkSize), 4);
        for (size_t i = 0; i < num_samples; ++i) {
            float v = std::max(-1.0f, std::min(1.0f, audio_data[i]));
            int16_t s = int16_t(v * 32767.0f);
            file.putChar(s & 0xFF);
            file.putChar((s >> 8) & 0xFF);
        }
        file.close();
        return true;
    } catch (const Ort::Exception& e) {
        QMessageBox::critical(nullptr, "ONNX Runtime", QString("Error ONNX: ") + e.what());
        return false;
    }
}
// --- End IA Voice Synthesis ONNX ---

class KawaiiMainWindow : public QWidget {
    Q_OBJECT
public:
    KawaiiMainWindow(QWidget* parent = nullptr)
        : QWidget(parent), voicebank(new Voicebank(this)), wavUtils(new WavUtils(this)) {
        this->setWindowTitle(QString::fromUtf8("UTAU Kawaii Synth 🎀"));

        QVBoxLayout* mainLayout = new QVBoxLayout(this);

        QMenuBar* menubar = new QMenuBar(this);
        QMenu* archivoMenu = menubar->addMenu("Archivo");
        QAction* abrirAction = new QAction("Abrir", this);
        QAction* guardarAction = new QAction("Guardar", this);
        QAction* exportarWavAction = new QAction("Exportar WAV (voz IA ONNX)", this);
        archivoMenu->addAction(abrirAction);
        archivoMenu->addAction(guardarAction);
        archivoMenu->addAction(exportarWavAction);
        mainLayout->setMenuBar(menubar);

        QLabel* kawaiiTitle = new QLabel("🎀 UTAU Kawaii Synth 🎵", this);
        kawaiiTitle->setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4; text-align: center;");
        kawaiiTitle->setAlignment(Qt::AlignCenter);

        QPushButton* loadVoicepthButton = new QPushButton("Selecciona voice.pth (modelo IA)", this);
        loadVoicepthButton->setStyleSheet("background: #ffe4ec; border-radius: 16px; font-size: 16px;");

        statusLabel = new QLabel("¡Bienvenida! Por favor selecciona tu modelo voice.pth (y ten el .onnx en la misma carpeta). (◕‿◕✿)", this);
        statusLabel->setStyleSheet("font-size: 16px; color: #a7a7a7;");

        phonemeTable = new QTableWidget(this);
        phonemeTable->setColumnCount(3);
        phonemeTable->setHorizontalHeaderLabels({"Alias", "Archivo WAV", "Offset (ms)"});
        phonemeTable->horizontalHeader()->setStyleSheet("font-size: 14px; color: #ff69b4;");
        phonemeTable->setStyleSheet("background: #fffde7; font-size: 14px; border-radius: 8px;");
        phonemeTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        phonemeTable->setEditTriggers(QAbstractItemView::NoEditTriggers);

        pianoRoll = new PianoRollWidget(this);
        pianoRoll->setMinimumHeight(350);

        connect(pianoRoll, &PianoRollWidget::notesChanged, this, [this](const std::vector<PianoNote>& notes){
            statusLabel->setText("¡Notas actualizadas! Haz doble clic en una nota para editar letra/fonema.");
        });

        connect(loadVoicepthButton, &QPushButton::clicked, this, &KawaiiMainWindow::onLoadVoicepth);

        connect(guardarAction, &QAction::triggered, this, &KawaiiMainWindow::onGuardarHL);
        connect(abrirAction, &QAction::triggered, this, &KawaiiMainWindow::onAbrirHL);

        connect(exportarWavAction, &QAction::triggered, this, &KawaiiMainWindow::onExportarWavAI);

        QSplitter* splitter = new QSplitter(Qt::Vertical, this);
        QWidget* topWidget = new QWidget;
        QVBoxLayout* topLayout = new QVBoxLayout(topWidget);
        topLayout->addWidget(kawaiiTitle);
        topLayout->addWidget(loadVoicepthButton);
        topLayout->addWidget(statusLabel);
        topLayout->addWidget(phonemeTable);
        splitter->addWidget(topWidget);
        splitter->addWidget(pianoRoll);
        splitter->setStretchFactor(1, 2);

        mainLayout->addWidget(splitter);

        setAcceptDrops(true);
    }

protected:
    void dragEnterEvent(QDragEnterEvent* event) override {
        if (event->mimeData()->hasUrls()) {
            QList<QUrl> urls = event->mimeData()->urls();
            if (!urls.isEmpty() && urls[0].toLocalFile().endsWith(".hl")) {
                event->acceptProposedAction();
            }
        }
    }
    void dropEvent(QDropEvent* event) override {
        QList<QUrl> urls = event->mimeData()->urls();
        if (!urls.isEmpty()) {
            QString filePath = urls[0].toLocalFile();
            if (filePath.endsWith(".hl")) {
                loadHLFile(filePath);
            }
        }
    }

public slots:
    void onLoadVoicepth() {
        QString file = QFileDialog::getOpenFileName(this, "Selecciona voice.pth", QString(), "Modelo IA (*.pth)");
        if (file.isEmpty()) return;
        loadedVoicepthPath = file;
        loadedOnnxPath = pthToOnnx(loadedVoicepthPath);
        statusLabel->setText("Modelo voice.pth cargado: " + file + "\nModelo .onnx: " + loadedOnnxPath);
    }

    void onGuardarHL() {
        QString fileName = QFileDialog::getSaveFileName(this, "Guardar archivo HL", QString(), "Archivo HL (*.hl)");
        if (fileName.isEmpty())
            return;
        if (!fileName.endsWith(".hl"))
            fileName += ".hl";
        saveHLFile(fileName);
    }

    void onAbrirHL() {
        QString fileName = QFileDialog::getOpenFileName(this, "Abrir archivo HL", QString(), "Archivo HL (*.hl)");
        if (!fileName.isEmpty()) {
            loadHLFile(fileName);
        }
    }

    void onExportarWavAI() {
        if (loadedOnnxPath.isEmpty()) {
            QMessageBox::warning(this, "Exportar WAV", "Por favor selecciona tu modelo voice.pth y asegúrate de tener el .onnx correspondiente.");
            return;
        }
        QString fileName = QFileDialog::getSaveFileName(this, "Exportar WAV (voz IA ONNX)", "", "Audio WAV (*.wav)");
        if (fileName.isEmpty()) return;
        if (exportVoiceAIWav_ONNX(fileName, pianoRoll->notes(), loadedOnnxPath)) {
            QMessageBox::information(this, "Exportar WAV", "¡Archivo WAV exportado con voz IA (ONNX)!");
        } else {
            QMessageBox::warning(this, "Exportar WAV", "Error al exportar WAV. Asegúrate de tener un modelo IA .onnx válido.");
        }
    }

    void loadHLFile(const QString& filePath) {
        QFile file(filePath);
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::warning(this, "Error", "No se puede abrir el archivo HL.");
            return;
        }
        QByteArray data = file.readAll();
        file.close();

        QJsonDocument doc = QJsonDocument::fromJson(data);
        if (!doc.isObject()) {
            QMessageBox::warning(this, "Error", "El archivo HL no es JSON válido.");
            return;
        }
        QJsonObject root = doc.object();
        QJsonArray notesArr = root.value("notes").toArray();

        std::vector<PianoNote> notes;
        for (const QJsonValue& v : notesArr) {
            QJsonObject obj = v.toObject();
            PianoNote n;
            n.pitch = obj.value("pitch").toInt();
            n.start = obj.value("start").toInt();
            n.length = obj.value("length").toInt();
            n.lyric = obj.value("lyric").toString();
            notes.push_back(n);
        }
        pianoRoll->setNotes(notes);
        statusLabel->setText("Archivo HL cargado exitosamente.");
    }

    void saveHLFile(const QString& filePath) {
        std::vector<PianoNote> notes = pianoRoll->notes();
        QJsonArray notesArr;
        for (const PianoNote& n : notes) {
            QJsonObject obj;
            obj["pitch"] = n.pitch;
            obj["start"] = n.start;
            obj["length"] = n.length;
            obj["lyric"] = n.lyric;
            notesArr.append(obj);
        }
        QJsonObject root;
        root["notes"] = notesArr;
        QJsonDocument doc(root);

        QFile file(filePath);
        if (!file.open(QIODevice::WriteOnly)) {
            QMessageBox::warning(this, "Error", "No se puede guardar el archivo HL.");
            return;
        }
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
        statusLabel->setText("Archivo HL guardado exitosamente.");
    }

private:
    Voicebank* voicebank;
    WavUtils* wavUtils;
    QTableWidget* phonemeTable;
    QLabel* statusLabel;
    PianoRollWidget* pianoRoll;
    QString loadedVoicepthPath;
    QString loadedOnnxPath;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    KawaiiMainWindow w;
    w.resize(1024, 720);
    w.show();
    return app.exec();
}
