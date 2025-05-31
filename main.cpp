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
#include <QTemporaryDir>
#include <cmath>
#include <vector>
#include <algorithm>
#include <optional>
#include <QDir>
#include <QComboBox>
#include <QSet>
#include "Voicebank.h"
#include "WavUtils.h"
#include "PhonemeEntry.h"

// --- ONNX Runtime Integration ---
#include <onnxruntime_cxx_api.h>

struct PianoNote {
    int pitch;
    int start;
    int length;
    QString lyric;
};

// Dummy PianoRollWidget for demo
class PianoRollWidget : public QWidget {
    Q_OBJECT
public:
    PianoRollWidget(QWidget* parent = nullptr) : QWidget(parent) {}
    std::vector<PianoNote> notes() const { return {}; }
    void setNotes(const std::vector<PianoNote>&) {}
signals:
    void notesChanged(const std::vector<PianoNote>&);
};

// Helper: Extrae .hlvb (zip) a temporal y retorna ruta de voice.pth/.onnx/info.json/oto.ini
bool extractHlVoicebank(const QString& hlvbPath, QString& outTempDir, QString& outVoicepth, QString& outOnnx, QString& outInfoJson, QString& outOtoIni) {
    QTemporaryDir* tempDir = new QTemporaryDir();
    if (!tempDir->isValid()) return false;
    outTempDir = tempDir->path();

    // Usa unzip para extraer el .hlvb
    QProcess unzipProc;
    unzipProc.setWorkingDirectory(outTempDir);
    unzipProc.start("unzip", QStringList() << hlvbPath);
    unzipProc.waitForFinished(-1);

    if (unzipProc.exitStatus() != QProcess::NormalExit || unzipProc.exitCode() != 0) {
        qWarning() << "Unzip failed:" << unzipProc.readAllStandardError();
        return false;
    }

    // Busca voice.pth, voice.onnx, info.json, oto.ini en el directorio extraído
    QDir dir(outTempDir);
    QStringList files = dir.entryList(QDir::Files | QDir::NoDotAndDotDot | QDir::AllDirs, QDir::Name);
    outVoicepth = outOnnx = outInfoJson = outOtoIni = "";

    for (const QString& file : files) {
        if (file.endsWith("voice.pth")) outVoicepth = dir.absoluteFilePath(file);
        if (file.endsWith("voice.onnx")) outOnnx = dir.absoluteFilePath(file);
        if (file == "info.json") outInfoJson = dir.absoluteFilePath(file);
        if (file == "oto.ini") outOtoIni = dir.absoluteFilePath(file);
    }
    // Si hay subcarpetas, busca recursivamente
    QStringList subdirs = dir.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
    for (const QString& subdir : subdirs) {
        QDir sub(dir.absoluteFilePath(subdir));
        for (const QString& file : sub.entryList(QDir::Files)) {
            if (file.endsWith("voice.pth")) outVoicepth = sub.absoluteFilePath(file);
            if (file.endsWith("voice.onnx")) outOnnx = sub.absoluteFilePath(file);
            if (file == "info.json") outInfoJson = sub.absoluteFilePath(file);
            if (file == "oto.ini") outOtoIni = sub.absoluteFilePath(file);
        }
    }

    return QFile::exists(outVoicepth) && QFile::exists(outOnnx);
}

// --- IA Voice Synthesis: ONNX Runtime (requiere .onnx del modelo IA) ---
bool exportVoiceAIWav_ONNX(const QString& filename, const std::vector<PianoNote>& notes, const QString& onnxPath, int bpm = 120, int sampleRate = 44100) {
    if (!QFile::exists(onnxPath)) return false;
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "kawaii-onnx");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        Ort::Session session(env, onnxPath.toStdString().c_str(), session_options);

        // Prepara entrada: convierte las notas a features reales para tu modelo
        std::vector<float> note_feats;
        for (const auto& n : notes) {
            note_feats.push_back(float(n.start));
            note_feats.push_back(float(n.length));
            note_feats.push_back(float(n.pitch));
            // Si tu modelo ONNX espera mels, fonemas, etc, aquí conviértelos
        }

        std::vector<int64_t> note_shape = { int64_t(notes.size()), 3 };
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names = { session.GetInputName(0, allocator) };
        std::vector<const char*> output_names = { session.GetOutputName(0, allocator) };

        Ort::MemoryInfo mem_info("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, note_feats.data(), note_feats.size(), note_shape.data(), note_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        float* audio_data = output_tensors[0].GetTensorMutableData<float>();
        size_t num_samples = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

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
        QAction* cargarHlVbAction = new QAction("Cargar voicebank (.hlvb)", this);
        QAction* exportarWavAction = new QAction("Exportar WAV (voz IA ONNX)", this);
        archivoMenu->addAction(abrirAction);
        archivoMenu->addAction(guardarAction);
        archivoMenu->addAction(cargarHlVbAction);
        archivoMenu->addAction(exportarWavAction);
        mainLayout->setMenuBar(menubar);

        QLabel* kawaiiTitle = new QLabel("🎀 UTAU Kawaii Synth 🎵", this);
        kawaiiTitle->setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4; text-align: center;");
        kawaiiTitle->setAlignment(Qt::AlignCenter);

        statusLabel = new QLabel("¡Bienvenida! Por favor carga un voicebank .hlvb con modelo IA (pth+onnx). (◕‿◕✿)", this);
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

        // --- NUEVO: Listado de HLVBs ---
        hlvbCombo = new QComboBox(this);
        hlvbCombo->setStyleSheet("font-size: 18px; background: #ffe4ec; border-radius: 12px;");
        connect(hlvbCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &KawaiiMainWindow::onSelectHlvbCombo);

        // Layout
        QSplitter* splitter = new QSplitter(Qt::Vertical, this);
        QWidget* topWidget = new QWidget;
        QVBoxLayout* topLayout = new QVBoxLayout(topWidget);
        topLayout->addWidget(kawaiiTitle);
        topLayout->addWidget(hlvbCombo);
        topLayout->addWidget(statusLabel);
        topLayout->addWidget(phonemeTable);
        splitter->addWidget(topWidget);
        splitter->addWidget(pianoRoll);
        splitter->setStretchFactor(1, 2);
        mainLayout->addWidget(splitter);

        connect(cargarHlVbAction, &QAction::triggered, this, &KawaiiMainWindow::onLoadHlVoicebank);
        connect(guardarAction, &QAction::triggered, this, &KawaiiMainWindow::onGuardarHL);
        connect(abrirAction, &QAction::triggered, this, &KawaiiMainWindow::onAbrirHL);
        connect(exportarWavAction, &QAction::triggered, this, &KawaiiMainWindow::onExportarWavAI);

        setAcceptDrops(true);

        // --- Mostrar todos los HLVB al iniciar ---
        listAndShowHlvbs();
    }

    void listAndShowHlvbs() {
        hlvbCombo->clear();
        hlvbPaths.clear();

        // Busca .hlvb en ~/.kawaii_hlvb y en el directorio actual
        QStringList searchDirs = { QDir::homePath() + "/.kawaii_hlvb", QDir::currentPath() };
        QSet<QString> seen;
        for (const QString& d : searchDirs) {
            QDir dir(d);
            for (const QString& file : dir.entryList(QStringList() << "*.hlvb", QDir::Files)) {
                QString abs = dir.absoluteFilePath(file);
                if (!seen.contains(abs)) {
                    hlvbPaths << abs;
                    seen.insert(abs);
                }
            }
        }
        if (!hlvbPaths.isEmpty()) {
            for (const QString& path : hlvbPaths) {
                QString name = QFileInfo(path).baseName();
                hlvbCombo->addItem(name, path);
            }
            hlvbCombo->setCurrentIndex(0);
            onSelectHlvbCombo(0);
        } else {
            hlvbCombo->addItem("No hay voicebanks .hlvb encontrados");
            statusLabel->setText("No se encontró voicebank HLVB instalado. ¡Crea o importa uno kawaii!");
        }
    }

    void onSelectHlvbCombo(int idx) {
        if (idx < 0 || idx >= hlvbPaths.size())
            return;
        QString found = hlvbPaths[idx];
        QString tempDir, tempPth, tempOnnx, tempInfo, tempOto;
        if (extractHlVoicebank(found, tempDir, tempPth, tempOnnx, tempInfo, tempOto)) {
            QFile infoFile(tempInfo);
            if (infoFile.open(QIODevice::ReadOnly)) {
                QJsonDocument doc = QJsonDocument::fromJson(infoFile.readAll());
                if (doc.isObject()) {
                    QJsonObject obj = doc.object();
                    QString name = obj.value("name").toString();
                    QString author = obj.value("author").toString();
                    QString desc = obj.value("description").toString();
                    statusLabel->setText("HLVB instalado: " + name + " por " + author + "\n" + desc);
                }
                infoFile.close();
            }
            voicebank->loadFromOtoIni(tempOto);
            phonemeTable->setRowCount(0);
            int row = 0;
            for (auto it = voicebank->phonemeMap.begin(); it != voicebank->phonemeMap.end(); ++it) {
                PhonemeEntry* entry = it.value();
                phonemeTable->insertRow(row);
                phonemeTable->setItem(row, 0, new QTableWidgetItem(entry->alias));
                phonemeTable->setItem(row, 1, new QTableWidgetItem(entry->waveFilePath));
                phonemeTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry->offset)));
                ++row;
            }
            extractedDir = tempDir; extractedVoicepth = tempPth; extractedOnnx = tempOnnx; extractedInfoJson = tempInfo; extractedOtoIni = tempOto;
        }
    }

    void onLoadHlVoicebank() {
        QString file = QFileDialog::getOpenFileName(this, "Selecciona voicebank (.hlvb)", QString(), "Voicebank HLVB (*.hlvb)");
        if (file.isEmpty()) return;
        if (!extractHlVoicebank(file, extractedDir, extractedVoicepth, extractedOnnx, extractedInfoJson, extractedOtoIni)) {
            QMessageBox::critical(this, "Error", "No se pudo importar el .hlvb. ¿Contiene voice.pth y voice.onnx?");
            return;
        }
        QFile infoFile(extractedInfoJson);
        if (infoFile.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(infoFile.readAll());
            if (doc.isObject()) {
                QJsonObject obj = doc.object();
                QString name = obj.value("name").toString();
                QString author = obj.value("author").toString();
                QString desc = obj.value("description").toString();
                statusLabel->setText("Voicebank: " + name + " por " + author + "\n" + desc);
            }
            infoFile.close();
        }
        voicebank->loadFromOtoIni(extractedOtoIni);
        phonemeTable->setRowCount(0);
        int row = 0;
        for (auto it = voicebank->phonemeMap.begin(); it != voicebank->phonemeMap.end(); ++it) {
            PhonemeEntry* entry = it.value();
            phonemeTable->insertRow(row);
            phonemeTable->setItem(row, 0, new QTableWidgetItem(entry->alias));
            phonemeTable->setItem(row, 1, new QTableWidgetItem(entry->waveFilePath));
            phonemeTable->setItem(row, 2, new QTableWidgetItem(QString::number(entry->offset)));
            ++row;
        }
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
        if (extractedOnnx.isEmpty()) {
            QMessageBox::warning(this, "Exportar WAV", "Debes cargar un voicebank .hlvb con voice.onnx exportado.");
            return;
        }
        QString fileName = QFileDialog::getSaveFileName(this, "Exportar WAV (voz IA ONNX)", "", "Audio WAV (*.wav)");
        if (fileName.isEmpty()) return;
        if (exportVoiceAIWav_ONNX(fileName, pianoRoll->notes(), extractedOnnx)) {
            QMessageBox::information(this, "Exportar WAV", "¡Archivo WAV exportado con voz IA (ONNX)!");
        } else {
            QMessageBox::warning(this, "Exportar WAV", "Error al exportar WAV. ¿voice.onnx es válido?");
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
    QComboBox* hlvbCombo;
    QStringList hlvbPaths;
    QString extractedDir, extractedVoicepth, extractedOnnx, extractedInfoJson, extractedOtoIni;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    KawaiiMainWindow w;
    w.resize(1024, 720);
    w.show();
    return app.exec();
}
