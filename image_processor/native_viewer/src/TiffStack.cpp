#include "TiffStack.h"

#include <QDateTime>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QList>
#include <QMutex>
#include <QMutexLocker>

#include <tiffio.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {

struct TiffCloser {
    void operator()(TIFF* tif) const
    {
        if (tif != nullptr) {
            TIFFClose(tif);
        }
    }
};

using TiffHandle = std::unique_ptr<TIFF, TiffCloser>;

struct InfoCacheEntry {
    TiffStackInfo info;
    qint64 size = -1;
    qint64 modifiedMs = -1;
};

QMutex gInfoCacheMutex;
QHash<QString, InfoCacheEntry> gInfoCache;
QList<QString> gInfoCacheLru;
constexpr int MaxInfoCacheEntries = 32;

TIFF* openTiff(const QString& path)
{
#ifdef _WIN32
    return TIFFOpenW(reinterpret_cast<const wchar_t*>(path.utf16()), "r");
#else
    const QByteArray encodedPath = QFile::encodeName(path);
    return TIFFOpen(encodedPath.constData(), "r");
#endif
}

bool readDirectoryFields(TIFF* tif, TiffStackInfo* info, QString* error)
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint16_t bitsPerSample = 0;
    uint16_t sampleFormat = SAMPLEFORMAT_UINT;
    uint16_t samplesPerPixel = 1;

    if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width) != 1
        || TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height) != 1) {
        if (error != nullptr) {
            *error = QStringLiteral("TIFF is missing image width or height.");
        }
        return false;
    }

    if (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample) != 1) {
        if (error != nullptr) {
            *error = QStringLiteral("TIFF is missing bits-per-sample metadata.");
        }
        return false;
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

    if (width > static_cast<uint32_t>(std::numeric_limits<int>::max())
        || height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        if (error != nullptr) {
            *error = QStringLiteral("TIFF dimensions are too large for Qt image display.");
        }
        return false;
    }

    info->width = static_cast<int>(width);
    info->height = static_cast<int>(height);
    info->bitsPerSample = static_cast<int>(bitsPerSample);
    info->sampleFormat = static_cast<int>(sampleFormat);
    info->samplesPerPixel = static_cast<int>(samplesPerPixel);
    info->tiled = TIFFIsTiled(tif) != 0;
    info->bigTiff = TIFFIsBigTIFF(tif) != 0;
    return true;
}

QString cacheKeyForPath(const QString& path)
{
    return QFileInfo(path).absoluteFilePath();
}

void touchInfoCacheKey(const QString& key)
{
    gInfoCacheLru.removeAll(key);
    gInfoCacheLru.prepend(key);
}

bool fileFingerprint(const QString& path, QString* key, qint64* size, qint64* modifiedMs)
{
    const QFileInfo fileInfo(path);
    if (!fileInfo.isFile()) {
        return false;
    }
    if (key != nullptr) {
        *key = fileInfo.absoluteFilePath();
    }
    if (size != nullptr) {
        *size = fileInfo.size();
    }
    if (modifiedMs != nullptr) {
        *modifiedMs = fileInfo.lastModified().toUTC().toMSecsSinceEpoch();
    }
    return true;
}

bool readCachedInfo(
    const QString& path,
    TiffStackInfo* info,
    const std::function<bool()>& shouldCancel)
{
    if (shouldCancel && shouldCancel()) {
        return false;
    }

    QString key;
    qint64 size = -1;
    qint64 modifiedMs = -1;
    if (!fileFingerprint(path, &key, &size, &modifiedMs)) {
        return false;
    }

    QMutexLocker locker(&gInfoCacheMutex);
    const auto iterator = gInfoCache.constFind(key);
    if (iterator == gInfoCache.constEnd()) {
        return false;
    }
    if (iterator->size != size || iterator->modifiedMs != modifiedMs) {
        gInfoCache.remove(key);
        gInfoCacheLru.removeAll(key);
        return false;
    }

    if (info != nullptr) {
        *info = iterator->info;
        info->path = path;
        info->fromCache = true;
        info->elapsedMs = 0;
    }
    touchInfoCacheKey(key);
    return true;
}

void storeCachedInfo(const QString& path, const TiffStackInfo& info)
{
    if (!info.indexComplete || info.frameCount <= 0 || info.directoryOffsets.empty()) {
        return;
    }

    QString key;
    qint64 size = -1;
    qint64 modifiedMs = -1;
    if (!fileFingerprint(path, &key, &size, &modifiedMs)) {
        return;
    }

    TiffStackInfo cachedInfo = info;
    cachedInfo.path = key;
    cachedInfo.fromCache = false;

    QMutexLocker locker(&gInfoCacheMutex);
    gInfoCache.insert(key, InfoCacheEntry {cachedInfo, size, modifiedMs});
    touchInfoCacheKey(key);
    while (gInfoCacheLru.size() > MaxInfoCacheEntries) {
        const QString staleKey = gInfoCacheLru.takeLast();
        gInfoCache.remove(staleKey);
    }
}

QString unsupportedFormatError(const TiffStackInfo& info)
{
    if (info.samplesPerPixel != 1) {
        return QStringLiteral("Native image processor v1 supports grayscale TIFF stacks only.");
    }
    if (info.sampleFormat == SAMPLEFORMAT_UINT && (info.bitsPerSample == 8 || info.bitsPerSample == 16)) {
        return {};
    }
    if (info.sampleFormat == SAMPLEFORMAT_INT && info.bitsPerSample == 16) {
        return {};
    }
    if (info.sampleFormat == SAMPLEFORMAT_IEEEFP && info.bitsPerSample == 32) {
        return {};
    }
    return QStringLiteral("Native image processor supports uint8, uint16, int16, and float32 grayscale TIFF stacks only.");
}

} // namespace

QString TiffStackInfo::pixelType() const
{
    QString scalarType;
    if (sampleFormat == SAMPLEFORMAT_INT) {
        scalarType = QStringLiteral("int%1").arg(bitsPerSample);
    } else if (sampleFormat == SAMPLEFORMAT_IEEEFP) {
        scalarType = QStringLiteral("float%1").arg(bitsPerSample);
    } else {
        scalarType = QStringLiteral("uint%1").arg(bitsPerSample);
    }

    if (samplesPerPixel == 1) {
        return scalarType;
    }
    return QStringLiteral("%1 x %2").arg(samplesPerPixel).arg(scalarType);
}

bool TiffStackInfo::hasDirectoryOffsets() const
{
    return indexComplete && frameCount > 0 && directoryOffsets.size() == static_cast<size_t>(frameCount);
}

namespace {

bool readInfoInternal(
    const QString& path,
    TiffStackInfo* info,
    QString* error,
    const std::function<bool()>& shouldCancel,
    int maxDirectories)
{
    QElapsedTimer timer;
    timer.start();

    const auto cancelled = [&shouldCancel, error]() {
        if (shouldCancel && shouldCancel()) {
            if (error != nullptr) {
                *error = QStringLiteral("TIFF open was cancelled.");
            }
            return true;
        }
        return false;
    };

    if (cancelled()) {
        return false;
    }

    if (path.isEmpty()) {
        if (error != nullptr) {
            *error = QStringLiteral("No TIFF path was provided.");
        }
        return false;
    }

    TiffHandle tif(openTiff(path));
    if (tif == nullptr) {
        if (error != nullptr) {
            *error = QStringLiteral("Could not open TIFF: %1").arg(path);
        }
        return false;
    }
    if (cancelled()) {
        return false;
    }

    TiffStackInfo localInfo;
    localInfo.path = path;
    if (!readDirectoryFields(tif.get(), &localInfo, error)) {
        return false;
    }
    if (cancelled()) {
        return false;
    }

    int frameCount = 0;
    std::vector<uint64_t> directoryOffsets;
    do {
        if (cancelled()) {
            return false;
        }
        directoryOffsets.push_back(static_cast<uint64_t>(TIFFCurrentDirOffset(tif.get())));
        ++frameCount;
        if (maxDirectories > 0 && frameCount >= maxDirectories) {
            break;
        }
    } while (TIFFReadDirectory(tif.get()) == 1);
    localInfo.frameCount = frameCount;
    localInfo.indexComplete = maxDirectories <= 0;
    if (directoryOffsets.size() == static_cast<size_t>(frameCount)) {
        localInfo.directoryOffsets = std::move(directoryOffsets);
    }
    localInfo.elapsedMs = timer.elapsed();

    if (info != nullptr) {
        *info = localInfo;
    }
    return true;
}

} // namespace

bool TiffStack::readPreviewInfo(const QString& path, TiffStackInfo* info, QString* error)
{
    return readPreviewInfo(path, info, error, {});
}

bool TiffStack::readPreviewInfo(
    const QString& path,
    TiffStackInfo* info,
    QString* error,
    const std::function<bool()>& shouldCancel)
{
    return readInfoInternal(path, info, error, shouldCancel, 1);
}

bool TiffStack::readInfo(const QString& path, TiffStackInfo* info, QString* error)
{
    return readInfo(path, info, error, {});
}

bool TiffStack::readInfo(
    const QString& path,
    TiffStackInfo* info,
    QString* error,
    const std::function<bool()>& shouldCancel)
{
    if (readCachedInfo(path, info, shouldCancel)) {
        if (error != nullptr) {
            error->clear();
        }
        return true;
    }

    TiffStackInfo localInfo;
    const bool ok = readInfoInternal(path, &localInfo, error, shouldCancel, 0);
    if (ok) {
        storeCachedInfo(path, localInfo);
        if (info != nullptr) {
            *info = localInfo;
        }
    }
    return ok;
}

bool TiffFrameResult::hasSamples() const
{
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (sampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 8) {
        return samples8.size() == static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    if (sampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 16) {
        return samples16.size() == static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    if ((sampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 16)
        || (sampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 32)) {
        return samplesFloat.size() == static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    return false;
}

namespace {

TiffFrameResult readFrameIndexed(
    const TiffStackInfo* indexedInfo,
    const QString& path,
    int frameIndex,
    const std::function<bool()>& shouldCancel)
{
    QElapsedTimer timer;
    timer.start();

    TiffFrameResult result;
    const auto cancelled = [&]() {
        if (shouldCancel && shouldCancel()) {
            result.cancelled = true;
            result.error = QStringLiteral("Frame load was cancelled.");
            result.elapsedMs = timer.elapsed();
            return true;
        }
        return false;
    };

    if (cancelled()) {
        return result;
    }

    if (frameIndex < 0) {
        result.error = QStringLiteral("Frame index must be non-negative.");
        return result;
    }

    TiffHandle tif(openTiff(path));
    if (tif == nullptr) {
        result.error = QStringLiteral("Could not open TIFF: %1").arg(path);
        return result;
    }
    if (cancelled()) {
        return result;
    }

    bool selectedDirectory = false;
    if (indexedInfo != nullptr
        && frameIndex < static_cast<int>(indexedInfo->directoryOffsets.size())
        && indexedInfo->directoryOffsets[static_cast<size_t>(frameIndex)] > 0) {
        selectedDirectory = TIFFSetSubDirectory(
                                tif.get(),
                                static_cast<toff_t>(indexedInfo->directoryOffsets[static_cast<size_t>(frameIndex)]))
            == 1;
        result.usedDirectoryOffset = selectedDirectory;
    }

    if (cancelled()) {
        return result;
    }

    if (!selectedDirectory && TIFFSetDirectory(tif.get(), static_cast<tdir_t>(frameIndex)) != 1) {
        result.error = QStringLiteral("TIFF does not contain frame %1.").arg(frameIndex + 1);
        return result;
    }
    if (cancelled()) {
        return result;
    }

    TiffStackInfo info;
    QString error;
    if (!readDirectoryFields(tif.get(), &info, &error)) {
        result.error = error;
        return result;
    }
    if (cancelled()) {
        return result;
    }

    const QString unsupported = unsupportedFormatError(info);
    if (!unsupported.isEmpty()) {
        result.error = unsupported;
        return result;
    }

    uint16_t photometric = PHOTOMETRIC_MINISBLACK;
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PHOTOMETRIC, &photometric);
    const bool invert = photometric == PHOTOMETRIC_MINISWHITE;

    const size_t pixelCount = static_cast<size_t>(info.width) * static_cast<size_t>(info.height);
    if (pixelCount == 0) {
        result.error = QStringLiteral("TIFF dimensions are invalid.");
        return result;
    }

    result.width = info.width;
    result.height = info.height;
    result.bitsPerSample = info.bitsPerSample;
    result.sampleFormat = info.sampleFormat;

    if (info.tiled) {
        uint32_t tileWidth = 0;
        uint32_t tileHeight = 0;
        if (TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tileWidth) != 1
            || TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &tileHeight) != 1
            || tileWidth == 0
            || tileHeight == 0) {
            result.error = QStringLiteral("TIFF tile dimensions are invalid.");
            return result;
        }

        const tmsize_t tileSize = TIFFTileSize(tif.get());
        if (tileSize <= 0) {
            result.error = QStringLiteral("TIFF tile size is invalid.");
            return result;
        }

        const size_t tilePixels = static_cast<size_t>(tileWidth) * static_cast<size_t>(tileHeight);
        if (tilePixels == 0) {
            result.error = QStringLiteral("TIFF tile dimensions are invalid.");
            return result;
        }

        if (info.sampleFormat == SAMPLEFORMAT_UINT && info.bitsPerSample == 8) {
            if (static_cast<size_t>(tileSize) < tilePixels) {
                result.error = QStringLiteral("8-bit TIFF tile is smaller than expected.");
                return result;
            }

            std::vector<uint8_t> tileBuffer(static_cast<size_t>(tileSize));
            result.samples8.resize(pixelCount);
            uint8_t minValue = std::numeric_limits<uint8_t>::max();
            uint8_t maxValue = 0;

            for (uint32_t tileY = 0; tileY < static_cast<uint32_t>(info.height); tileY += tileHeight) {
                for (uint32_t tileX = 0; tileX < static_cast<uint32_t>(info.width); tileX += tileWidth) {
                    if (cancelled()) {
                        return result;
                    }
                    if (TIFFReadTile(tif.get(), tileBuffer.data(), tileX, tileY, 0, 0) < 0) {
                        result.error =
                            QStringLiteral("Could not read TIFF tile at %1,%2.").arg(tileX).arg(tileY);
                        return result;
                    }

                    const uint32_t copyWidth =
                        std::min(tileWidth, static_cast<uint32_t>(info.width) - tileX);
                    const uint32_t copyHeight =
                        std::min(tileHeight, static_cast<uint32_t>(info.height) - tileY);
                    for (uint32_t localY = 0; localY < copyHeight; ++localY) {
                        for (uint32_t localX = 0; localX < copyWidth; ++localX) {
                            const size_t tileOffset =
                                static_cast<size_t>(localY) * static_cast<size_t>(tileWidth)
                                + static_cast<size_t>(localX);
                            const uint8_t rawValue = tileBuffer[tileOffset];
                            const uint8_t value = invert ? static_cast<uint8_t>(255U - rawValue) : rawValue;
                            result.samples8[static_cast<size_t>(tileY + localY) * static_cast<size_t>(info.width)
                                + static_cast<size_t>(tileX + localX)] = value;
                            minValue = std::min(minValue, value);
                            maxValue = std::max(maxValue, value);
                        }
                    }
                }
            }

            result.observedMin = static_cast<int>(minValue);
            result.observedMax = static_cast<int>(maxValue);
        } else if (info.sampleFormat == SAMPLEFORMAT_UINT && info.bitsPerSample == 16) {
            const size_t expectedTileBytes = tilePixels * sizeof(uint16_t);
            if (static_cast<size_t>(tileSize) < expectedTileBytes) {
                result.error = QStringLiteral("16-bit TIFF tile is smaller than expected.");
                return result;
            }

            std::vector<uint16_t> tileBuffer(
                static_cast<size_t>(tileSize + static_cast<tmsize_t>(sizeof(uint16_t)) - 1)
                / sizeof(uint16_t));
            result.samples16.resize(pixelCount);
            uint16_t minValue = std::numeric_limits<uint16_t>::max();
            uint16_t maxValue = 0;

            for (uint32_t tileY = 0; tileY < static_cast<uint32_t>(info.height); tileY += tileHeight) {
                for (uint32_t tileX = 0; tileX < static_cast<uint32_t>(info.width); tileX += tileWidth) {
                    if (cancelled()) {
                        return result;
                    }
                    if (TIFFReadTile(tif.get(), tileBuffer.data(), tileX, tileY, 0, 0) < 0) {
                        result.error =
                            QStringLiteral("Could not read TIFF tile at %1,%2.").arg(tileX).arg(tileY);
                        return result;
                    }

                    const uint32_t copyWidth =
                        std::min(tileWidth, static_cast<uint32_t>(info.width) - tileX);
                    const uint32_t copyHeight =
                        std::min(tileHeight, static_cast<uint32_t>(info.height) - tileY);
                    for (uint32_t localY = 0; localY < copyHeight; ++localY) {
                        for (uint32_t localX = 0; localX < copyWidth; ++localX) {
                            const size_t tileOffset =
                                static_cast<size_t>(localY) * static_cast<size_t>(tileWidth)
                                + static_cast<size_t>(localX);
                            const uint16_t rawValue = tileBuffer[tileOffset];
                            const uint16_t value = invert ? static_cast<uint16_t>(65535U - rawValue) : rawValue;
                            result.samples16[static_cast<size_t>(tileY + localY) * static_cast<size_t>(info.width)
                                + static_cast<size_t>(tileX + localX)] = value;
                            minValue = std::min(minValue, value);
                            maxValue = std::max(maxValue, value);
                        }
                    }
                }
            }

            result.observedMin = static_cast<int>(minValue);
            result.observedMax = static_cast<int>(maxValue);
        } else if (info.sampleFormat == SAMPLEFORMAT_INT && info.bitsPerSample == 16) {
            const size_t expectedTileBytes = tilePixels * sizeof(int16_t);
            if (static_cast<size_t>(tileSize) < expectedTileBytes) {
                result.error = QStringLiteral("signed 16-bit TIFF tile is smaller than expected.");
                return result;
            }

            std::vector<int16_t> tileBuffer(
                static_cast<size_t>(tileSize + static_cast<tmsize_t>(sizeof(int16_t)) - 1)
                / sizeof(int16_t));
            result.samplesFloat.resize(pixelCount);
            double minValue = std::numeric_limits<double>::max();
            double maxValue = std::numeric_limits<double>::lowest();

            for (uint32_t tileY = 0; tileY < static_cast<uint32_t>(info.height); tileY += tileHeight) {
                for (uint32_t tileX = 0; tileX < static_cast<uint32_t>(info.width); tileX += tileWidth) {
                    if (cancelled()) {
                        return result;
                    }
                    if (TIFFReadTile(tif.get(), tileBuffer.data(), tileX, tileY, 0, 0) < 0) {
                        result.error =
                            QStringLiteral("Could not read TIFF tile at %1,%2.").arg(tileX).arg(tileY);
                        return result;
                    }

                    const uint32_t copyWidth =
                        std::min(tileWidth, static_cast<uint32_t>(info.width) - tileX);
                    const uint32_t copyHeight =
                        std::min(tileHeight, static_cast<uint32_t>(info.height) - tileY);
                    for (uint32_t localY = 0; localY < copyHeight; ++localY) {
                        for (uint32_t localX = 0; localX < copyWidth; ++localX) {
                            const size_t tileOffset =
                                static_cast<size_t>(localY) * static_cast<size_t>(tileWidth)
                                + static_cast<size_t>(localX);
                            const float value = static_cast<float>(tileBuffer[tileOffset]);
                            result.samplesFloat[static_cast<size_t>(tileY + localY) * static_cast<size_t>(info.width)
                                + static_cast<size_t>(tileX + localX)] = value;
                            minValue = std::min(minValue, static_cast<double>(value));
                            maxValue = std::max(maxValue, static_cast<double>(value));
                        }
                    }
                }
            }

            result.observedMin = minValue;
            result.observedMax = maxValue;
        } else {
            const size_t expectedTileBytes = tilePixels * sizeof(float);
            if (static_cast<size_t>(tileSize) < expectedTileBytes) {
                result.error = QStringLiteral("float32 TIFF tile is smaller than expected.");
                return result;
            }

            std::vector<float> tileBuffer(
                static_cast<size_t>(tileSize + static_cast<tmsize_t>(sizeof(float)) - 1)
                / sizeof(float));
            result.samplesFloat.resize(pixelCount);
            double minValue = std::numeric_limits<double>::max();
            double maxValue = std::numeric_limits<double>::lowest();
            bool hasFiniteValue = false;

            for (uint32_t tileY = 0; tileY < static_cast<uint32_t>(info.height); tileY += tileHeight) {
                for (uint32_t tileX = 0; tileX < static_cast<uint32_t>(info.width); tileX += tileWidth) {
                    if (cancelled()) {
                        return result;
                    }
                    if (TIFFReadTile(tif.get(), tileBuffer.data(), tileX, tileY, 0, 0) < 0) {
                        result.error =
                            QStringLiteral("Could not read TIFF tile at %1,%2.").arg(tileX).arg(tileY);
                        return result;
                    }

                    const uint32_t copyWidth =
                        std::min(tileWidth, static_cast<uint32_t>(info.width) - tileX);
                    const uint32_t copyHeight =
                        std::min(tileHeight, static_cast<uint32_t>(info.height) - tileY);
                    for (uint32_t localY = 0; localY < copyHeight; ++localY) {
                        for (uint32_t localX = 0; localX < copyWidth; ++localX) {
                            const size_t tileOffset =
                                static_cast<size_t>(localY) * static_cast<size_t>(tileWidth)
                                + static_cast<size_t>(localX);
                            const float value = tileBuffer[tileOffset];
                            result.samplesFloat[static_cast<size_t>(tileY + localY) * static_cast<size_t>(info.width)
                                + static_cast<size_t>(tileX + localX)] = value;
                            if (std::isfinite(value)) {
                                minValue = std::min(minValue, static_cast<double>(value));
                                maxValue = std::max(maxValue, static_cast<double>(value));
                                hasFiniteValue = true;
                            }
                        }
                    }
                }
            }

            result.observedMin = hasFiniteValue ? minValue : 0.0;
            result.observedMax = hasFiniteValue ? maxValue : 0.0;
        }

        result.ok = true;
        result.elapsedMs = timer.elapsed();
        return result;
    }

    const tmsize_t scanlineSize = TIFFScanlineSize(tif.get());
    if (scanlineSize <= 0) {
        result.error = QStringLiteral("TIFF scanline size is invalid.");
        return result;
    }
    std::vector<uint8_t> scanline(static_cast<size_t>(scanlineSize));

    if (info.sampleFormat == SAMPLEFORMAT_UINT && info.bitsPerSample == 8) {
        if (scanlineSize < info.width) {
            result.error = QStringLiteral("TIFF scanline is smaller than the image width.");
            return result;
        }

        result.samples8.resize(pixelCount);
        uint8_t minValue = std::numeric_limits<uint8_t>::max();
        uint8_t maxValue = 0;
        for (int row = 0; row < info.height; ++row) {
            if (cancelled()) {
                return result;
            }
            if (TIFFReadScanline(tif.get(), scanline.data(), static_cast<uint32_t>(row), 0) < 0) {
                result.error = QStringLiteral("Could not read TIFF scanline %1.").arg(row + 1);
                return result;
            }
            for (int column = 0; column < info.width; ++column) {
                const uint8_t rawValue = scanline[static_cast<size_t>(column)];
                const uint8_t value = invert ? static_cast<uint8_t>(255U - rawValue) : rawValue;
                result.samples8[static_cast<size_t>(row) * static_cast<size_t>(info.width)
                    + static_cast<size_t>(column)] = value;
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
        }
        result.observedMin = static_cast<int>(minValue);
        result.observedMax = static_cast<int>(maxValue);
    } else if (info.sampleFormat == SAMPLEFORMAT_UINT && info.bitsPerSample == 16) {
        if (pixelCount == 0 || pixelCount > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
            result.error = QStringLiteral("TIFF dimensions are invalid.");
            return result;
        }
        if (scanlineSize < static_cast<tmsize_t>(info.width) * static_cast<tmsize_t>(sizeof(uint16_t))) {
            result.error = QStringLiteral("16-bit TIFF scanline is smaller than expected.");
            return result;
        }

        result.samples16.resize(pixelCount);
        std::vector<uint16_t> scanline16(
            static_cast<size_t>(scanlineSize + static_cast<tmsize_t>(sizeof(uint16_t)) - 1)
            / sizeof(uint16_t));
        uint16_t minValue = std::numeric_limits<uint16_t>::max();
        uint16_t maxValue = 0;

        for (int row = 0; row < info.height; ++row) {
            if (cancelled()) {
                return result;
            }
            if (TIFFReadScanline(tif.get(), scanline16.data(), static_cast<uint32_t>(row), 0) < 0) {
                result.error = QStringLiteral("Could not read TIFF scanline %1.").arg(row + 1);
                return result;
            }

            const auto* rowValues = scanline16.data();
            for (int column = 0; column < info.width; ++column) {
                const uint16_t rawValue = rowValues[column];
                const uint16_t value = invert ? static_cast<uint16_t>(65535U - rawValue) : rawValue;
                result.samples16[static_cast<size_t>(row) * static_cast<size_t>(info.width)
                    + static_cast<size_t>(column)] = value;
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
            }
        }
        result.observedMin = static_cast<int>(minValue);
        result.observedMax = static_cast<int>(maxValue);
    } else if (info.sampleFormat == SAMPLEFORMAT_INT && info.bitsPerSample == 16) {
        if (pixelCount == 0 || pixelCount > std::numeric_limits<size_t>::max() / sizeof(int16_t)) {
            result.error = QStringLiteral("TIFF dimensions are invalid.");
            return result;
        }
        if (scanlineSize < static_cast<tmsize_t>(info.width) * static_cast<tmsize_t>(sizeof(int16_t))) {
            result.error = QStringLiteral("signed 16-bit TIFF scanline is smaller than expected.");
            return result;
        }

        result.samplesFloat.resize(pixelCount);
        std::vector<int16_t> scanline16(
            static_cast<size_t>(scanlineSize + static_cast<tmsize_t>(sizeof(int16_t)) - 1)
            / sizeof(int16_t));
        double minValue = std::numeric_limits<double>::max();
        double maxValue = std::numeric_limits<double>::lowest();

        for (int row = 0; row < info.height; ++row) {
            if (cancelled()) {
                return result;
            }
            if (TIFFReadScanline(tif.get(), scanline16.data(), static_cast<uint32_t>(row), 0) < 0) {
                result.error = QStringLiteral("Could not read TIFF scanline %1.").arg(row + 1);
                return result;
            }

            const auto* rowValues = scanline16.data();
            for (int column = 0; column < info.width; ++column) {
                const float value = static_cast<float>(rowValues[column]);
                result.samplesFloat[static_cast<size_t>(row) * static_cast<size_t>(info.width)
                    + static_cast<size_t>(column)] = value;
                minValue = std::min(minValue, static_cast<double>(value));
                maxValue = std::max(maxValue, static_cast<double>(value));
            }
        }
        result.observedMin = minValue;
        result.observedMax = maxValue;
    } else {
        if (pixelCount == 0 || pixelCount > std::numeric_limits<size_t>::max() / sizeof(float)) {
            result.error = QStringLiteral("TIFF dimensions are invalid.");
            return result;
        }
        if (scanlineSize < static_cast<tmsize_t>(info.width) * static_cast<tmsize_t>(sizeof(float))) {
            result.error = QStringLiteral("float32 TIFF scanline is smaller than expected.");
            return result;
        }

        result.samplesFloat.resize(pixelCount);
        std::vector<float> scanlineFloat(
            static_cast<size_t>(scanlineSize + static_cast<tmsize_t>(sizeof(float)) - 1)
            / sizeof(float));
        double minValue = std::numeric_limits<double>::max();
        double maxValue = std::numeric_limits<double>::lowest();
        bool hasFiniteValue = false;

        for (int row = 0; row < info.height; ++row) {
            if (cancelled()) {
                return result;
            }
            if (TIFFReadScanline(tif.get(), scanlineFloat.data(), static_cast<uint32_t>(row), 0) < 0) {
                result.error = QStringLiteral("Could not read TIFF scanline %1.").arg(row + 1);
                return result;
            }

            const auto* rowValues = scanlineFloat.data();
            for (int column = 0; column < info.width; ++column) {
                const float value = rowValues[column];
                result.samplesFloat[static_cast<size_t>(row) * static_cast<size_t>(info.width)
                    + static_cast<size_t>(column)] = value;
                if (std::isfinite(value)) {
                    minValue = std::min(minValue, static_cast<double>(value));
                    maxValue = std::max(maxValue, static_cast<double>(value));
                    hasFiniteValue = true;
                }
            }
        }
        result.observedMin = hasFiniteValue ? minValue : 0.0;
        result.observedMax = hasFiniteValue ? maxValue : 0.0;
    }

    result.ok = true;
    result.elapsedMs = timer.elapsed();
    return result;
}

} // namespace

TiffFrameResult TiffStack::readFrame(const QString& path, int frameIndex)
{
    return readFrame(path, frameIndex, {});
}

TiffFrameResult TiffStack::readFrame(
    const QString& path,
    int frameIndex,
    const std::function<bool()>& shouldCancel)
{
    return readFrameIndexed(nullptr, path, frameIndex, shouldCancel);
}

TiffFrameResult TiffStack::readFrame(const TiffStackInfo& info, int frameIndex)
{
    return readFrame(info, frameIndex, {});
}

TiffFrameResult TiffStack::readFrame(
    const TiffStackInfo& info,
    int frameIndex,
    const std::function<bool()>& shouldCancel)
{
    return readFrameIndexed(&info, info.path, frameIndex, shouldCancel);
}
