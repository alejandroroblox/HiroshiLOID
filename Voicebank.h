#pragma once
#include <QObject>
#include <QMap>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QJsonObject>
#include <QJsonDocument>
#include "PhonemeEntry.h"

// Voicebank que soporta cargar .hlvb (zip con info.json, oto.ini y voice.pth)
class Voicebank : public QObject {
    Q_OBJECT
public:
    explicit Voicebank(QObject* parent = nullptr) : QObject(parent) {}

    QMap<QString, PhonemeEntry*> phonemeMap;

    // Ruta al modelo deep learning (voice.pth), para pasar a Python/TensorFlow/PyTorch
    QString deepModelPath;
    // Info opcional de info.json
    QJsonObject metaInfo;

    // Carga oto.ini (usado tras extraer un .hlvb)
    Q_INVOKABLE void loadFromOtoIni(const QString& otoIniPath) {
        QFile file(otoIniPath);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            emit errorOccurred("No se pudo abrir oto.ini: " + otoIniPath);
            return;
        }
        QTextStream in(&file);
        int lineNum = 0;
        while (!in.atEnd()) {
            QString line = in.readLine();
            ++lineNum;
            if (line.trimmed().isEmpty() || line.trimmed().startsWith("#")) continue;
            try {
                PhonemeEntry* entry = PhonemeEntry::parseOtoLine(line, this);
                phonemeMap[entry->alias] = entry;
            } catch (const std::exception& e) {
                emit errorOccurred(QString("Error parsing oto.ini line %1: %2").arg(lineNum).arg(e.what()));
            }
        }
        emit voicebankLoaded();
    }

    // Carga un info.json (opcional, para mostrar metadatos)
    Q_INVOKABLE void loadInfoJson(const QString& infoJsonPath) {
        QFile file(infoJsonPath);
        if (file.open(QIODevice::ReadOnly)) {
            QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
            if (doc.isObject()) {
                metaInfo = doc.object();
            }
            file.close();
        }
    }

    Q_INVOKABLE PhonemeEntry* getPhonemeEntry(const QString& alias) const {
        return phonemeMap.value(alias, nullptr);
    }

signals:
    void errorOccurred(const QString& message);
    void voicebankLoaded();
};
