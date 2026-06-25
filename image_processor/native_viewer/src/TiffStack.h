#pragma once

#include <QString>

#include <cstdint>
#include <functional>
#include <vector>

struct TiffStackInfo {
    QString path;
    int width = 0;
    int height = 0;
    int frameCount = 0;
    int bitsPerSample = 0;
    int sampleFormat = 1;
    int samplesPerPixel = 0;
    bool tiled = false;
    bool bigTiff = false;
    bool indexComplete = true;
    bool fromCache = false;
    std::vector<uint64_t> directoryOffsets;
    qint64 elapsedMs = 0;

    QString pixelType() const;
    bool hasDirectoryOffsets() const;
};

struct TiffFrameResult {
    bool ok = false;
    QString error;
    int width = 0;
    int height = 0;
    int bitsPerSample = 0;
    int sampleFormat = 1;
    double observedMin = 0.0;
    double observedMax = 0.0;
    std::vector<uint8_t> samples8;
    std::vector<uint16_t> samples16;
    std::vector<float> samplesFloat;
    qint64 elapsedMs = 0;
    bool usedDirectoryOffset = false;
    bool cancelled = false;

    bool hasSamples() const;
};

class TiffStack {
public:
    static bool readPreviewInfo(const QString& path, TiffStackInfo* info, QString* error);
    static bool readPreviewInfo(
        const QString& path,
        TiffStackInfo* info,
        QString* error,
        const std::function<bool()>& shouldCancel);
    static bool readInfo(const QString& path, TiffStackInfo* info, QString* error);
    static bool readInfo(
        const QString& path,
        TiffStackInfo* info,
        QString* error,
        const std::function<bool()>& shouldCancel);
    static TiffFrameResult readFrame(const QString& path, int frameIndex);
    static TiffFrameResult readFrame(
        const QString& path,
        int frameIndex,
        const std::function<bool()>& shouldCancel);
    static TiffFrameResult readFrame(const TiffStackInfo& info, int frameIndex);
    static TiffFrameResult readFrame(
        const TiffStackInfo& info,
        int frameIndex,
        const std::function<bool()>& shouldCancel);
};
