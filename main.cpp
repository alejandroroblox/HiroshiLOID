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
#include "Voicebank.h"
#include "WavUtils.h"
#include "PhonemeEntry.h"

// --- Piano Roll Implementation (VOCALOID6 style) ---
#include <QPainter>
#include <QInputDialog>
#include <vector>
#include <algorithm>
#include <optional>

struct PianoNote {
    int pitch;           // MIDI note number
    int start;           // In "ticks" (e.g. 480 = quarter note)
    int length;          // In "ticks"
    QString lyric;       // Phoneme or syllable
};

class PianoRollWidget : public QWidget {
    Q_OBJECT
public:
    explicit PianoRollWidget(QWidget* parent = nullptr)
        : QWidget(parent) {
        setMinimumHeight((m_maxPitch - m_minPitch + 1) * m_noteHeight);
        setMinimumWidth(xFromTick(m_totalTicks));
        setMouseTracking(true);
    }

    std::vector<PianoNote> notes() const { return m_notes; }
    void setNotes(const std::vector<PianoNote>& notes) {
        m_notes = notes;
        update();
    }

    void setTicksPerBeat(int ticks) { m_ticksPerBeat = ticks; }
    void setTotalTicks(int ticks) {
        m_totalTicks = ticks;
        setMinimumWidth(xFromTick(m_totalTicks));
        update();
    }

    void clearNotes() {
        m_notes.clear();
        update();
    }

signals:
    void notesChanged(const std::vector<PianoNote>& notes);

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);

        // Draw background grid
        for (int i = 0; i <= (m_maxPitch - m_minPitch); ++i) {
            int y = i * m_noteHeight;
            QColor bg = ((m_maxPitch - i) % 12 == 0) ? QColor("#ffe4ec") : QColor("#fffde7");
            p.fillRect(0, y, width(), m_noteHeight, bg);
            p.setPen(QColor("#ffe4ec"));
            p.drawLine(0, y, width(), y);
        }
        for (int t = 0; t <= m_totalTicks; t += 120) { // grid every 16th note
            int x = xFromTick(t);
            p.setPen(QColor("#baffc9"));
            p.drawLine(x, 0, x, height());
        }

        // Draw notes
        for (size_t idx = 0; idx < m_notes.size(); ++idx) {
            const PianoNote& n = m_notes[idx];
            int x = xFromTick(n.start);
            int w = xFromTick(n.start + n.length) - x;
            int y = yFromPitch(n.pitch);
            QRect rect(x, y, std::max(18, w), m_noteHeight - 2);
            p.setBrush(QColor("#ffb3de"));
            p.setPen(Qt::NoPen);
            p.drawRect(rect);
            p.setPen(QColor("#ff69b4"));
            p.drawText(rect, Qt::AlignCenter, n.lyric.isEmpty() ? "♪" : n.lyric);
        }

        // Draw new note (while drawing)
        if (m_drawingNote) {
            int x = xFromTick(m_newNote.start);
            int w = xFromTick(m_newNote.start + m_newNote.length) - x;
            int y = yFromPitch(m_newNote.pitch);
            QRect rect(x, y, std::max(18, w), m_noteHeight - 2);
            p.setBrush(QColor(255, 179, 222, 128));
            p.setPen(QColor("#ff69b4"));
            p.drawRect(rect);
        }
    }

    void mousePressEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton) {
            m_pressPos = e->pos();
            m_newNote.start = tickFromX(e->x());
            m_newNote.pitch = pitchFromY(e->y());
            m_newNote.length = 120; // por defecto 16th note
            m_newNote.lyric = "";
            m_drawingNote = true;
            update();
        }
    }
    void mouseMoveEvent(QMouseEvent* e) override {
        if (m_drawingNote) {
            int tickLength = tickFromX(e->x()) - m_newNote.start;
            m_newNote.length = std::max(120, tickLength);
            update();
        }
    }
    void mouseReleaseEvent(QMouseEvent* e) override {
        if (m_drawingNote && e->button() == Qt::LeftButton) {
            m_drawingNote = false;
            // Extender si es necesario
            extendRollIfNeeded(m_newNote.start + m_newNote.length);
            m_notes.push_back(m_newNote);
            emit notesChanged(m_notes);
            update();
        }
    }
    void mouseDoubleClickEvent(QMouseEvent* e) override {
        auto idxOpt = noteAt(e->pos());
        if (idxOpt.has_value()) {
            int idx = idxOpt.value();
            PianoNote& note = m_notes[idx];
            QRect rect(xFromTick(note.start), yFromPitch(note.pitch),
                       xFromTick(note.start + note.length) - xFromTick(note.start),
                       m_noteHeight - 2);
            bool ok;
            QString text = QInputDialog::getText(this, "Editar letra kawaii",
                                                 "Letra/Fonema:", QLineEdit::Normal,
                                                 note.lyric, &ok, Qt::WindowFlags(), Qt::ImhNone);
            if (ok) {
                note.lyric = text.trimmed();
                emit notesChanged(m_notes);
                update();
            }
        }
    }
    void resizeEvent(QResizeEvent*) override {
        update();
    }

private:
    std::vector<PianoNote> m_notes;
    int m_ticksPerBeat = 480;
    int m_totalTicks = 1920; // 4 bars default
    int m_minPitch = 48;     // C3
    int m_maxPitch = 84;     // C6

    bool m_drawingNote = false;
    QPoint m_pressPos;
    PianoNote m_newNote;
    int m_gridWidth = 24;    // pixels per 120 ticks (e.g. 16th note)
    int m_noteHeight = 18;   // pixels per semitone

    int tickFromX(int x) const { return (x * 120) / m_gridWidth; }
    int xFromTick(int tick) const { return (tick * m_gridWidth) / 120; }
    int pitchFromY(int y) const {
        int pitch = m_maxPitch - (y / m_noteHeight);
        return std::clamp(pitch, m_minPitch, m_maxPitch);
    }
    int yFromPitch(int pitch) const { return (m_maxPitch - pitch) * m_noteHeight; }

    void extendRollIfNeeded(int noteEndTick) {
        if (noteEndTick > m_totalTicks - 120) {
            setTotalTicks(m_totalTicks + 1920); // Extiende 4 compases más
        }
    }

    std::optional<int> noteAt(const QPoint& pos) const {
        for (size_t idx = 0; idx < m_notes.size(); ++idx) {
            const PianoNote& n = m_notes[idx];
            int x = xFromTick(n.start);
            int w = xFromTick(n.start + n.length) - x;
            int y = yFromPitch(n.pitch);
            QRect rect(x, y, std::max(18, w), m_noteHeight - 2);
            if (rect.contains(pos)) {
                return static_cast<int>(idx);
            }
        }
        return std::nullopt;
    }
};
// --- End Piano Roll Implementation ---

// --- Simple UTAU-like Voice Synthesis for Exporting WAV ---
#include <QMap>
#include <QFile>
#include <QDir>
#include <QAudioBuffer>
#include <QtMath>

// Estructura básica para OTO entries (de tu voicebank)
struct OtoEntry {
    QString alias;
    QString wavPath;
    int offsetMs = 0;
    int lengthMs = 0;
};

bool exportVoiceWav(const QString& filename, const std::vector<PianoNote>& notes, const QMap<QString, OtoEntry>& otoMap, int bpm = 120, int sampleRate = 44100) {
    double tickLen = 60.0 / (bpm * 480.0);
    int lastTick = 0;
    for (const auto& n : notes) {
        int end = n.start + n.length;
        if (end > lastTick) lastTick = end;
    }
    int totalSamples = int((lastTick * tickLen) * sampleRate) + sampleRate;
    std::vector<float> audio(totalSamples, 0.0f);

    for (const auto& note : notes) {
        if (!otoMap.contains(note.lyric)) continue;
        const OtoEntry& oto = otoMap[note.lyric];

        // Leer el WAV del fonema
        QFile wavFile(oto.wavPath);
        if (!wavFile.open(QIODevice::ReadOnly)) continue;
        QByteArray wavData = wavFile.readAll();
        wavFile.close();
        if (wavData.size() < 44) continue;

        // Extraer PCM de 16 bits mono (esto es MUY SIMPLIFICADO; usa una lib real para robustez)
        int dataOffset = 44;
        int sampleLen = (wavData.size() - dataOffset) / 2;
        std::vector<float> sampleData(sampleLen, 0.0f);
        for (int i = 0; i < sampleLen; ++i) {
            int16_t s = *(const int16_t*)(wavData.constData() + dataOffset + i * 2);
            sampleData[i] = float(s) / 32768.0f;
        }

        // Recorta según offset/length del OTO.ini
        int startSample = sampleRate * oto.offsetMs / 1000;
        int lengthSample = oto.lengthMs > 0 ? sampleRate * oto.lengthMs / 1000 : sampleData.size() - startSample;
        if (startSample + lengthSample > sampleData.size())
            lengthSample = sampleData.size() - startSample;
        if (lengthSample < 1) continue;

        std::vector<float> region(sampleData.begin() + startSample, sampleData.begin() + startSample + lengthSample);

        // Ajusta duración (time-stretch simple: repite/recorta)
        int noteSamples = int(note.length * tickLen * sampleRate);
        std::vector<float> noteAudio(noteSamples, 0.0f);
        for (int i = 0; i < noteSamples; ++i) {
            noteAudio[i] = region[(i * region.size()) / noteSamples];
        }

        // Ajusta pitch (NO real, pero cambia velocidad de reproducción para demostrar)
        double basePitch = 60; // puedes leer desde otoMap o asumir C4
        double pitchRatio = std::pow(2.0, (note.pitch - basePitch) / 12.0);
        for (int i = 0; i < noteSamples; ++i) {
            int srcIdx = int(i / pitchRatio);
            if (srcIdx < noteAudio.size())
                noteAudio[i] = noteAudio[srcIdx];
        }

        // Mezcla al buffer final
        int outStart = int(note.start * tickLen * sampleRate);
        for (int i = 0; i < noteSamples && outStart + i < audio.size(); ++i)
            audio[outStart + i] += noteAudio[i];
    }

    // Normalizar y guardar como WAV PCM
    float max = 0;
    for (auto v : audio) if (qAbs(v) > max) max = qAbs(v);
    if (max > 0.99f) for (auto& v : audio) v /= max;

    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) return false;
    int dataChunkSize = int(audio.size()) * 2;
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
    for (auto v : audio) {
        int16_t s = int16_t(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        file.putChar(s & 0xFF);
        file.putChar((s >> 8) & 0xFF);
    }
    file.close();
    return true;
}
// --- End Simple UTAU-like Voice Synthesis ---

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
        QAction* exportarWavAction = new QAction("Exportar WAV (voz real)", this);
        archivoMenu->addAction(abrirAction);
        archivoMenu->addAction(guardarAction);
        archivoMenu->addAction(exportarWavAction);
        mainLayout->setMenuBar(menubar);

        QLabel* kawaiiTitle = new QLabel("🎀 UTAU Kawaii Synth 🎵", this);
        kawaiiTitle->setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4; text-align: center;");
        kawaiiTitle->setAlignment(Qt::AlignCenter);

        QPushButton* loadHlVoicebankButton = new QPushButton("Cargar Voicebank (.hlvb) (｡♥‿♥｡)", this);
        loadHlVoicebankButton->setStyleSheet("background: #ffe4ec; border-radius: 16px; font-size: 16px;");

        statusLabel = new QLabel("¡Bienvenida! Por favor carga un voicebank. (◕‿◕✿)", this);
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

        connect(loadHlVoicebankButton, &QPushButton::clicked, this, &KawaiiMainWindow::onLoadHlVoicebank);
        connect(voicebank, &Voicebank::voicebankLoaded, this, &KawaiiMainWindow::onVoicebankLoaded);
        connect(voicebank, &Voicebank::errorOccurred, this, &KawaiiMainWindow::onError);

        connect(guardarAction, &QAction::triggered, this, &KawaiiMainWindow::onGuardarHL);
        connect(abrirAction, &QAction::triggered, this, &KawaiiMainWindow::onAbrirHL);

        connect(exportarWavAction, &QAction::triggered, this, &KawaiiMainWindow::onExportarWav);

        QSplitter* splitter = new QSplitter(Qt::Vertical, this);
        QWidget* topWidget = new QWidget;
        QVBoxLayout* topLayout = new QVBoxLayout(topWidget);
        topLayout->addWidget(kawaiiTitle);
        topLayout->addWidget(loadHlVoicebankButton);
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
    void onLoadHlVoicebank() {
        QString file = QFileDialog::getOpenFileName(this, "Selecciona voicebank (.hlvb)", QString(), "Voicebank HLVB (*.hlvb)");
        if (file.isEmpty()) return;

        QString extractedDir, infoJson, otoIni, voiceModel;
        if (!unzipHlVoicebank(file, extractedDir, infoJson, otoIni, voiceModel)) {
            QMessageBox::critical(this, "Error", "No se pudo importar el .hlvb. ¿Es un archivo válido?");
            return;
        }

        // Leer info.json y mostrarlo
        QFile infoFile(infoJson);
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

        // Cargar oto.ini usando Voicebank adaptado
        voicebank->loadFromOtoIni(otoIni);

        // Guardar info sobre el modelo entrenado
        voicebank->deepModelPath = voiceModel;
    }

    void onVoicebankLoaded() {
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
        statusLabel->setText(statusLabel->text() + "\n¡Voicebank cargado! Fonemas abajo.");
    }

    void onError(const QString& msg) {
        QMessageBox::critical(this, "Error", msg);
        statusLabel->setText("Ocurrió un error (╥﹏╥)");
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

    void onExportarWav() {
        QString fileName = QFileDialog::getSaveFileName(this, "Exportar WAV (voz real)", "", "Audio WAV (*.wav)");
        if (fileName.isEmpty()) return;
        // Voicebank debe tener un map alias->OtoEntry
        QMap<QString, OtoEntry> otoMap;
        // Construye el otoMap desde tu Voicebank
        for (auto it = voicebank->phonemeMap.begin(); it != voicebank->phonemeMap.end(); ++it) {
            PhonemeEntry* entry = it.value();
            OtoEntry o;
            o.alias = entry->alias;
            o.wavPath = entry->waveFilePath;
            o.offsetMs = entry->offset;
            o.lengthMs = entry->consonant; // o usa entry->cutoff, etc.
            otoMap.insert(o.alias, o);
        }
        if (exportVoiceWav(fileName, pianoRoll->notes(), otoMap)) {
            QMessageBox::information(this, "Exportar WAV", "¡Archivo WAV exportado con voz real!");
        } else {
            QMessageBox::warning(this, "Exportar WAV", "Error al exportar WAV.");
        }
    }

    bool unzipHlVoicebank(const QString& zipPath, QString& outExtractedDir, QString& outInfoJson, QString& outOtoIni, QString& outVoiceModel) {
        QTemporaryDir* tempDir = new QTemporaryDir();
        if (!tempDir->isValid())
            return false;

        outExtractedDir = tempDir->path();

        QProcess unzipProc;
        unzipProc.setWorkingDirectory(outExtractedDir);
        unzipProc.start("unzip", QStringList() << zipPath);
        unzipProc.waitForFinished(-1);

        if (unzipProc.exitStatus() != QProcess::NormalExit || unzipProc.exitCode() != 0) {
            qWarning() << "Unzip failed:" << unzipProc.readAllStandardError();
            return false;
        }

        outInfoJson = outExtractedDir + "/info.json";
        outOtoIni = outExtractedDir + "/oto.ini";
        outVoiceModel = outExtractedDir + "/voice.pth";

        QFileInfo info(outInfoJson);
        QFileInfo oto(outOtoIni);
        QFileInfo pth(outVoiceModel);

        if (!info.exists() || !oto.exists() || !pth.exists())
            return false;

        return true;
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
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    KawaiiMainWindow w;
    w.resize(1024, 720);
    w.show();
    return app.exec();
}
