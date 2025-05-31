#pragma once
#include <QObject>
#include <QVector>
#include <QString>

extern "C" {
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
}

// Kawaii utility for loading WAV
class WavUtils : public QObject {
    Q_OBJECT
public:
    explicit WavUtils(QObject* parent = nullptr) : QObject(parent) {}

    // Carga WAV a buffer de float normalizado, usando dr_wav
    Q_INVOKABLE bool loadWavToBuffer(
        const QString& filePath,
        QVector<float>& outBuffer,
        unsigned int& outputSampleRate,
        unsigned int& outputChannels,
        unsigned int& outputBitsPerSample
    ) {
        drwav wav;
        if (!drwav_init_file(&wav, filePath.toUtf8().constData(), NULL)) {
            emit errorOccurred("No se pudo abrir el archivo WAV: " + filePath);
            return false;
        }
        outputSampleRate = wav.sampleRate;
        outputChannels = wav.channels;
        outputBitsPerSample = wav.bitsPerSample;

        size_t totalFrames = static_cast<size_t>(wav.totalPCMFrameCount) * wav.channels;
        outBuffer.resize(static_cast<int>(totalFrames));

        size_t framesRead = drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, outBuffer.data());
        if (framesRead * wav.channels != totalFrames) {
            drwav_uninit(&wav);
            emit errorOccurred("Error al leer el archivo WAV: " + filePath);
            return false;
        }
        drwav_uninit(&wav);
        return true;
    }

signals:
    void errorOccurred(const QString& message);
};
