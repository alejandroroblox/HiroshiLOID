#include <QApplication>
#include <QWidget>
#include "Voicebank.h"
#include "WavUtils.h"
#include "PhonemeEntry.h"
#include <QVBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QFileDialog>
#include <QTableWidget>
#include <QHeaderView>
#include <QMessageBox>

// Una ventanita kawaii para mostrar el voicebank
class KawaiiMainWindow : public QWidget {
    Q_OBJECT
public:
    KawaiiMainWindow(QWidget* parent = nullptr)
        : QWidget(parent), voicebank(new Voicebank(this)), wavUtils(new WavUtils(this)) {
        this->setWindowTitle(QString::fromUtf8("UTAU Kawaii Synth 🎀"));
        QVBoxLayout* layout = new QVBoxLayout(this);

        QLabel* kawaiiTitle = new QLabel("🎀 UTAU Kawaii Synth 🎵", this);
        kawaiiTitle->setStyleSheet("font-size: 32px; font-family: Comic Sans MS, cursive; color: #ff69b4; text-align: center;");
        kawaiiTitle->setAlignment(Qt::AlignCenter);

        QPushButton* loadOtoButton = new QPushButton("Cargar oto.ini (｡♥‿♥｡)", this);
        loadOtoButton->setStyleSheet("background: #ffe4ec; border-radius: 16px; font-size: 16px;");

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
        layout->addWidget(loadOtoButton);
        layout->addWidget(phonemeTable);
        layout->addWidget(statusLabel);

        connect(loadOtoButton, &QPushButton::clicked, this, &KawaiiMainWindow::onLoadOtoIni);
        connect(voicebank, &Voicebank::voicebankLoaded, this, &KawaiiMainWindow::onVoicebankLoaded);
        connect(voicebank, &Voicebank::errorOccurred, this, &KawaiiMainWindow::onError);
    }

private slots:
    void onLoadOtoIni() {
        QString file = QFileDialog::getOpenFileName(this, "Selecciona oto.ini", QString(), "oto.ini (*.ini)");
        if (!file.isEmpty()) {
            voicebank->load(file);
            statusLabel->setText("Cargando voicebank kawaii... (｡♥‿♥｡)");
        }
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
        statusLabel->setText("¡Voicebank cargado! Puedes ver los fonemas kawaii abajo.");
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
