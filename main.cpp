#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QTableWidget>
#include <QHeaderView>
#include <QMessageBox>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QDir>
#include <QProcess>
#include <QDebug>

#include "Voicebank.h"
#include "WavUtils.h"
#include "PhonemeEntry.h"

// Función kawaii para desempaquetar un .hlvb (zip renombrado)
bool unzipHlVoicebank(const QString& zipPath, QString& outExtractedDir, QString& outInfoJson, QString& outOtoIni, QString& outVoiceModel) {
    QTemporaryDir* tempDir = new QTemporaryDir();
    if (!tempDir->isValid())
        return false;

    outExtractedDir = tempDir->path();

    // Usar el comando 'unzip' disponible en el sistema, o puedes integrar QuaZip/quazip5 para multiplataforma
    QProcess unzipProc;
    unzipProc.setWorkingDirectory(outExtractedDir);
    unzipProc.start("unzip", QStringList() << zipPath);
    unzipProc.waitForFinished(-1);

    if (unzipProc.exitStatus() != QProcess::NormalExit || unzipProc.exitCode() != 0) {
        qWarning() << "Unzip failed:" << unzipProc.readAllStandardError();
        return false;
    }

    // Buscar info.json, oto.ini, voice.pth
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

class KawaiiMainWindow : public QWidget {
    Q_OBJECT
public:
    KawaiiMainWindow(QWidget* parent = nullptr)
        : QWidget(parent), voicebank(new Voicebank(this)), wavUtils(new WavUtils(this)) {
        this->setWindowTitle(QString::fromUtf8("HiroshiLOID"));
        QVBoxLayout* layout = new QVBoxLayout(this);

        QLabel* kawaiiTitle = new QLabel("HiroshiLOID", this);
        kawaiiTitle->setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4; text-align: center;");
        kawaiiTitle->setAlignment(Qt::AlignCenter);

        QPushButton* loadHlVoicebankButton = new QPushButton("Cargar Voicebank (.hlvb) (｡♥‿♥｡)", this);
        loadHlVoicebankButton->setStyleSheet("background: #ffe4ec; border-radius: 16px; font-size: 16px;");

        phonemeTable = new QTableWidget(this);
        phonemeTable->setColumnCount(3);
        phonemeTable->setHorizontalHeaderLabels({"Alias", "Archivo WAV", "Offset (ms)"});
        phonemeTable->horizontalHeader()->setStyleSheet("font-size: 14px; color: #ff69b4;");
        phonemeTable->setStyleSheet("background: #fffde7; font-size: 14px; border-radius: 8px;");
        phonemeTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        phonemeTable->setEditTriggers(QAbstractItemView::NoEditTriggers);

        statusLabel = new QLabel("¡Bienvenida! Por favor carga un voicebank. (◕‿◕✿)", this);
        statusLabel->setStyleSheet("font-size: 16px; color: #a7a7a7;");

        layout->addWidget(kawaiiTitle);
        layout->addWidget(loadHlVoicebankButton);
        layout->addWidget(phonemeTable);
        layout->addWidget(statusLabel);

        connect(loadHlVoicebankButton, &QPushButton::clicked, this, &KawaiiMainWindow::onLoadHlVoicebank);
        connect(voicebank, &Voicebank::voicebankLoaded, this, &KawaiiMainWindow::onVoicebankLoaded);
        connect(voicebank, &Voicebank::errorOccurred, this, &KawaiiMainWindow::onError);
    }

private slots:
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

private:
    Voicebank* voicebank;
    WavUtils* wavUtils;
    QTableWidget* phonemeTable;
    QLabel* statusLabel;
};

#include "main.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    KawaiiMainWindow w;
    w.show();
    return app.exec();
}
