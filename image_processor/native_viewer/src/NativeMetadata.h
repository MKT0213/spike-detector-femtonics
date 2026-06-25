#pragma once

#include <QList>
#include <QString>

struct NativeRoiBox {
    int ordinal = 0;
    int roiIndex = 0;
    int row = 0;
    int column = 0;
    int left = 0;
    int upper = 0;
    int right = 0;
    int lower = 0;
};

struct NativeRoiOverlay {
    QString metadataPath;
    int width = 0;
    int height = 0;
    int roiCount = 0;
    int columns = 0;
    int rows = 0;
    int roiWidth = 0;
    int roiHeight = 0;
    QString channel;
    QString dataType;
    QString comment;
    double samplingRateHz = 0.0;
    bool hasSamplingRate = false;
    QList<NativeRoiBox> boxes;

    bool isValid() const;
    QString summary() const;
};

struct NativeMetadataResult {
    bool ok = false;
    QString error;
    NativeRoiOverlay overlay;
};

class NativeMetadataReader {
public:
    static QString metadataPathForTiff(const QString& tiffPath);
    static NativeMetadataResult readForTiff(const QString& tiffPath);
    static NativeMetadataResult read(const QString& metadataPath);
};
