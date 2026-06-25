#include "MainWindow.h"
#include "NativeMetadata.h"
#include "TiffStack.h"
#include "TiffViewerWidget.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QStringList>
#include <QTextStream>
#include <QTimer>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

uint8_t scaleToByte(double value, double black, double scale)
{
    const int scaled = static_cast<int>((value - black) * scale + 0.5);
    return static_cast<uint8_t>(std::clamp(scaled, 0, 255));
}

std::vector<uint8_t> buildIntegerLookup(int maximum, double black, double white)
{
    std::vector<uint8_t> lookup(static_cast<size_t>(maximum) + 1);
    if (white <= black) {
        return lookup;
    }

    const double scale = 255.0 / (white - black);
    for (int value = 0; value <= maximum; ++value) {
        lookup[static_cast<size_t>(value)] = scaleToByte(static_cast<double>(value), black, scale);
    }
    return lookup;
}

int runProbe(const QString& path)
{
    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 2;
    }

    const TiffFrameResult firstFrame = TiffStack::readFrame(info, 0);
    if (!firstFrame.ok) {
        QTextStream(stderr) << firstFrame.error << Qt::endl;
        return 3;
    }

    TiffFrameResult lastFrame;
    if (info.frameCount > 1) {
        lastFrame = TiffStack::readFrame(info, info.frameCount - 1);
        if (!lastFrame.ok) {
            QTextStream(stderr) << lastFrame.error << Qt::endl;
            return 4;
        }
    }

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(QFileInfo(path).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("frames=%1").arg(info.frameCount) << Qt::endl;
    out << QStringLiteral("directory_offsets=%1").arg(static_cast<int>(info.directoryOffsets.size())) << Qt::endl;
    out << QStringLiteral("info_ms=%1").arg(info.elapsedMs) << Qt::endl;
    out << QStringLiteral("info_cached=%1").arg(info.fromCache ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("size=%1x%2").arg(info.width).arg(info.height) << Qt::endl;
    out << QStringLiteral("pixel_type=%1").arg(info.pixelType()) << Qt::endl;
    out << QStringLiteral("bigtiff=%1").arg(info.bigTiff ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("tiled=%1").arg(info.tiled ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("first_observed=%1..%2").arg(firstFrame.observedMin).arg(firstFrame.observedMax) << Qt::endl;
    out << QStringLiteral("first_frame_ms=%1").arg(firstFrame.elapsedMs) << Qt::endl;
    out << QStringLiteral("first_indexed=%1").arg(firstFrame.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    if (info.frameCount > 1) {
        out << QStringLiteral("last_observed=%1..%2").arg(lastFrame.observedMin).arg(lastFrame.observedMax) << Qt::endl;
        out << QStringLiteral("last_frame_ms=%1").arg(lastFrame.elapsedMs) << Qt::endl;
        out << QStringLiteral("last_indexed=%1").arg(lastFrame.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
    }
    return 0;
}

int runProbeInfoCache(const QString& path)
{
    TiffStackInfo firstInfo;
    QString firstError;
    QElapsedTimer firstTimer;
    firstTimer.start();
    if (!TiffStack::readInfo(path, &firstInfo, &firstError)) {
        QTextStream(stderr) << firstError << Qt::endl;
        return 2;
    }
    const qint64 firstWallMs = firstTimer.elapsed();

    TiffStackInfo secondInfo;
    QString secondError;
    QElapsedTimer secondTimer;
    secondTimer.start();
    if (!TiffStack::readInfo(path, &secondInfo, &secondError)) {
        QTextStream(stderr) << secondError << Qt::endl;
        return 3;
    }
    const qint64 secondWallMs = secondTimer.elapsed();

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(QFileInfo(path).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("first_frames=%1").arg(firstInfo.frameCount) << Qt::endl;
    out << QStringLiteral("second_frames=%1").arg(secondInfo.frameCount) << Qt::endl;
    out << QStringLiteral("first_offsets=%1").arg(static_cast<int>(firstInfo.directoryOffsets.size())) << Qt::endl;
    out << QStringLiteral("second_offsets=%1").arg(static_cast<int>(secondInfo.directoryOffsets.size())) << Qt::endl;
    out << QStringLiteral("first_cached=%1").arg(firstInfo.fromCache ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("second_cached=%1").arg(secondInfo.fromCache ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("first_info_ms=%1").arg(firstInfo.elapsedMs) << Qt::endl;
    out << QStringLiteral("second_info_ms=%1").arg(secondInfo.elapsedMs) << Qt::endl;
    out << QStringLiteral("first_wall_ms=%1").arg(firstWallMs) << Qt::endl;
    out << QStringLiteral("second_wall_ms=%1").arg(secondWallMs) << Qt::endl;

    if (firstInfo.frameCount != secondInfo.frameCount
        || firstInfo.directoryOffsets.size() != secondInfo.directoryOffsets.size()) {
        QTextStream(stderr) << QStringLiteral("Cached TIFF info does not match the original read.") << Qt::endl;
        return 4;
    }
    if (!secondInfo.fromCache) {
        QTextStream(stderr) << QStringLiteral("Second TIFF info read was not served from cache.") << Qt::endl;
        return 5;
    }
    return 0;
}

int runProbeFrameCache(const QStringList& args)
{
    int frames = 8;
    int maxFrames = 8;
    int width = 2048;
    int height = 2048;
    int bits = 16;
    int budgetMb = 32;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            bool ok = false;
            const int parsed = args.at(index + 1).toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, args.at(index + 1))
                                    << Qt::endl;
                return false;
            }
            *value = parsed;
            ++index;
            return true;
        };

        if (option == QStringLiteral("--frames")) {
            if (!requirePositiveInt(option, &frames)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-frames")) {
            if (!requirePositiveInt(option, &maxFrames)) {
                return 2;
            }
        } else if (option == QStringLiteral("--width")) {
            if (!requirePositiveInt(option, &width)) {
                return 2;
            }
        } else if (option == QStringLiteral("--height")) {
            if (!requirePositiveInt(option, &height)) {
                return 2;
            }
        } else if (option == QStringLiteral("--bits")) {
            if (!requirePositiveInt(option, &bits)) {
                return 2;
            }
        } else if (option == QStringLiteral("--budget-mb")) {
            if (!requirePositiveInt(option, &budgetMb)) {
                return 2;
            }
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (bits != 8 && bits != 16 && bits != 32) {
        QTextStream(stderr) << QStringLiteral("Bits must be 8, 16, or 32.") << Qt::endl;
        return 2;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t maxBytes = static_cast<size_t>(budgetMb) * 1024ULL * 1024ULL;
    FrameCache cache(maxFrames, maxBytes);
    int inserted = 0;
    size_t frameBytes = 0;

    for (int frameIndex = 0; frameIndex < frames; ++frameIndex) {
        CachedFrame frame;
        frame.width = width;
        frame.height = height;
        frame.bitsPerSample = bits;
        frame.sampleFormat = bits == 32 ? 3 : 1;
        if (bits == 8) {
            auto samples = std::make_shared<std::vector<uint8_t>>(pixelCount, static_cast<uint8_t>(frameIndex));
            frameBytes = samples->size() * sizeof(uint8_t);
            frame.samples8 = samples;
        } else if (bits == 16) {
            auto samples = std::make_shared<std::vector<uint16_t>>(pixelCount, static_cast<uint16_t>(frameIndex));
            frameBytes = samples->size() * sizeof(uint16_t);
            frame.samples16 = samples;
        } else {
            auto samples = std::make_shared<std::vector<float>>(pixelCount, static_cast<float>(frameIndex));
            frameBytes = samples->size() * sizeof(float);
            frame.samplesFloat = samples;
        }

        cache.put(frameIndex, frame);
        ++inserted;
    }

    QTextStream out(stdout);
    out << QStringLiteral("frames_inserted=%1").arg(inserted) << Qt::endl;
    out << QStringLiteral("cached_frames=%1").arg(cache.size()) << Qt::endl;
    out << QStringLiteral("frame_bytes=%1").arg(static_cast<qulonglong>(frameBytes)) << Qt::endl;
    out << QStringLiteral("total_bytes=%1").arg(static_cast<qulonglong>(cache.totalBytes())) << Qt::endl;
    out << QStringLiteral("max_bytes=%1").arg(static_cast<qulonglong>(cache.maxBytes())) << Qt::endl;
    out << QStringLiteral("within_budget=%1").arg(cache.totalBytes() <= cache.maxBytes() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("oldest_present=%1").arg(cache.contains(0) ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("newest_present=%1").arg(cache.contains(frames - 1) ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (cache.totalBytes() > cache.maxBytes()) {
        QTextStream(stderr) << QStringLiteral("Cache exceeded its byte budget.") << Qt::endl;
        return 3;
    }
    if (cache.size() > maxFrames) {
        QTextStream(stderr) << QStringLiteral("Cache exceeded its frame-count budget.") << Qt::endl;
        return 4;
    }
    if (frameBytes <= maxBytes && !cache.contains(frames - 1)) {
        QTextStream(stderr) << QStringLiteral("Newest cacheable frame was not retained.") << Qt::endl;
        return 5;
    }
    return 0;
}

int runProbeSequence(const QStringList& paths)
{
    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF paths were provided.") << Qt::endl;
        return 2;
    }

    QTextStream out(stdout);
    out << QStringLiteral("sequence_count=%1").arg(paths.size()) << Qt::endl;
    for (int index = 0; index < paths.size(); ++index) {
        TiffStackInfo info;
        QString error;
        if (!TiffStack::readInfo(paths.at(index), &info, &error)) {
            QTextStream(stderr) << QStringLiteral("path %1: %2").arg(index + 1).arg(error) << Qt::endl;
            return 3;
        }

        const TiffFrameResult frame = TiffStack::readFrame(info, 0);
        if (!frame.ok) {
            QTextStream(stderr) << QStringLiteral("path %1: %2").arg(index + 1).arg(frame.error) << Qt::endl;
            return 4;
        }

        out << QStringLiteral("[%1] path=%2").arg(index + 1).arg(QFileInfo(paths.at(index)).absoluteFilePath())
            << Qt::endl;
        out << QStringLiteral("[%1] frames=%2").arg(index + 1).arg(info.frameCount) << Qt::endl;
        out << QStringLiteral("[%1] directory_offsets=%2").arg(index + 1).arg(static_cast<int>(info.directoryOffsets.size()))
            << Qt::endl;
        out << QStringLiteral("[%1] size=%2x%3").arg(index + 1).arg(info.width).arg(info.height) << Qt::endl;
        out << QStringLiteral("[%1] pixel_type=%2").arg(index + 1).arg(info.pixelType()) << Qt::endl;
        out << QStringLiteral("[%1] bigtiff=%2")
                   .arg(index + 1)
                   .arg(info.bigTiff ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
        out << QStringLiteral("[%1] tiled=%2")
                   .arg(index + 1)
                   .arg(info.tiled ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
        out << QStringLiteral("[%1] first_frame_ms=%2").arg(index + 1).arg(frame.elapsedMs) << Qt::endl;
        out << QStringLiteral("[%1] first_indexed=%2")
                   .arg(index + 1)
                   .arg(frame.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
    }
    return 0;
}

int runProbeMetadata(const QString& path)
{
    const NativeMetadataResult metadata = NativeMetadataReader::readForTiff(path);
    if (!metadata.ok) {
        QTextStream(stderr) << metadata.error << Qt::endl;
        return 2;
    }

    QTextStream out(stdout);
    out << QStringLiteral("metadata=%1").arg(metadata.overlay.metadataPath) << Qt::endl;
    out << QStringLiteral("size=%1x%2").arg(metadata.overlay.width).arg(metadata.overlay.height) << Qt::endl;
    out << QStringLiteral("roi_count=%1").arg(metadata.overlay.roiCount) << Qt::endl;
    out << QStringLiteral("grid=%1x%2").arg(metadata.overlay.columns).arg(metadata.overlay.rows) << Qt::endl;
    out << QStringLiteral("roi_size=%1x%2").arg(metadata.overlay.roiWidth).arg(metadata.overlay.roiHeight) << Qt::endl;
    if (metadata.overlay.hasSamplingRate) {
        out << QStringLiteral("sampling_rate_hz=%1").arg(metadata.overlay.samplingRateHz, 0, 'g', 6) << Qt::endl;
    }
    for (const NativeRoiBox& box : metadata.overlay.boxes) {
        out << QStringLiteral("roi=%1,%2,%3,%4,%5")
                   .arg(box.roiIndex)
                   .arg(box.left)
                   .arg(box.upper)
                   .arg(box.right)
                   .arg(box.lower)
            << Qt::endl;
    }
    return 0;
}

int runProbeScrub(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    const QString path = args.at(0);
    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }

    QList<int> frameIndices;
    for (int index = 1; index < args.size(); ++index) {
        bool ok = false;
        const int frameIndex = args.at(index).toInt(&ok);
        if (!ok) {
            QTextStream(stderr) << QStringLiteral("Invalid frame index: %1").arg(args.at(index)) << Qt::endl;
            return 4;
        }
        frameIndices.append(frameIndex);
    }

    if (frameIndices.isEmpty()) {
        const int probeCount = std::min(info.frameCount, 8);
        for (int index = 0; index < probeCount; ++index) {
            frameIndices.append(index);
        }
    }

    QTextStream out(stdout);
    QElapsedTimer totalTimer;
    totalTimer.start();
    out << QStringLiteral("scrub_count=%1").arg(frameIndices.size()) << Qt::endl;
    for (int frameIndex : frameIndices) {
        if (frameIndex < 0 || frameIndex >= info.frameCount) {
            QTextStream(stderr) << QStringLiteral("Frame index %1 is outside 0..%2.")
                                       .arg(frameIndex)
                                       .arg(std::max(0, info.frameCount - 1))
                                << Qt::endl;
            return 5;
        }

        const TiffFrameResult frame = TiffStack::readFrame(info, frameIndex);
        if (!frame.ok) {
            QTextStream(stderr) << frame.error << Qt::endl;
            return 6;
        }
        out << QStringLiteral("frame=%1 ms=%2 indexed=%3 size=%4x%5 bits=%6 observed=%7..%8")
                   .arg(frameIndex)
                   .arg(frame.elapsedMs)
                   .arg(frame.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
                   .arg(frame.width)
                   .arg(frame.height)
                   .arg(frame.bitsPerSample)
                   .arg(frame.observedMin)
                   .arg(frame.observedMax)
            << Qt::endl;
    }
    out << QStringLiteral("total_ms=%1").arg(totalTimer.elapsed()) << Qt::endl;
    return 0;
}

int runProbeFrameAccess(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    const QString path = args.at(0);
    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }

    bool frameOk = false;
    int frameIndex = args.size() >= 2 ? args.at(1).toInt(&frameOk) : std::max(0, info.frameCount - 1);
    if (args.size() < 2) {
        frameOk = true;
    }
    if (!frameOk || frameIndex < 0 || frameIndex >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Frame index must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 4;
    }

    const TiffFrameResult direct = TiffStack::readFrame(path, frameIndex);
    if (!direct.ok) {
        QTextStream(stderr) << direct.error << Qt::endl;
        return 5;
    }

    const TiffFrameResult indexed = TiffStack::readFrame(info, frameIndex);
    if (!indexed.ok) {
        QTextStream(stderr) << indexed.error << Qt::endl;
        return 6;
    }

    QTextStream(stdout)
        << QStringLiteral("frame=%1").arg(frameIndex) << Qt::endl
        << QStringLiteral("frames=%1").arg(info.frameCount) << Qt::endl
        << QStringLiteral("directory_offsets=%1").arg(static_cast<int>(info.directoryOffsets.size())) << Qt::endl
        << QStringLiteral("info_ms=%1").arg(info.elapsedMs) << Qt::endl
        << QStringLiteral("direct_ms=%1").arg(direct.elapsedMs) << Qt::endl
        << QStringLiteral("direct_indexed=%1").arg(direct.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl
        << QStringLiteral("indexed_ms=%1").arg(indexed.elapsedMs) << Qt::endl
        << QStringLiteral("indexed_indexed=%1")
               .arg(indexed.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    return 0;
}

qint64 renderFrameToGrayscaleMs(const TiffFrameResult& frame, QString* error)
{
    if (error != nullptr) {
        error->clear();
    }
    if (!frame.ok || !frame.hasSamples()) {
        if (error != nullptr) {
            *error = frame.error.isEmpty() ? QStringLiteral("Frame has no samples.") : frame.error;
        }
        return -1;
    }

    QElapsedTimer renderTimer;
    renderTimer.start();
    QImage image(frame.width, frame.height, QImage::Format_Grayscale8);
    if (image.isNull()) {
        if (error != nullptr) {
            *error = QStringLiteral("Could not allocate render image.");
        }
        return -1;
    }

    const double black = std::isfinite(frame.observedMin) ? frame.observedMin : 0.0;
    double white = std::isfinite(frame.observedMax) ? frame.observedMax : black + 1.0;
    if (white <= black) {
        white = black + 1.0;
    }

    if (!frame.samples8.empty()) {
        const auto lookup = buildIntegerLookup(255, black, white);
        for (int row = 0; row < frame.height; ++row) {
            const auto* source =
                frame.samples8.data() + static_cast<size_t>(row) * static_cast<size_t>(frame.width);
            uint8_t* destination = image.scanLine(row);
            for (int column = 0; column < frame.width; ++column) {
                destination[column] = lookup[source[column]];
            }
        }
        return renderTimer.elapsed();
    }

    if (!frame.samples16.empty()) {
        const auto lookup = buildIntegerLookup(65535, black, white);
        for (int row = 0; row < frame.height; ++row) {
            const auto* source =
                frame.samples16.data() + static_cast<size_t>(row) * static_cast<size_t>(frame.width);
            uint8_t* destination = image.scanLine(row);
            for (int column = 0; column < frame.width; ++column) {
                destination[column] = lookup[source[column]];
            }
        }
        return renderTimer.elapsed();
    }

    const double scale = 255.0 / (white - black);
    for (int row = 0; row < frame.height; ++row) {
        uint8_t* destination = image.scanLine(row);
        for (int column = 0; column < frame.width; ++column) {
            const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(frame.width)
                + static_cast<size_t>(column);
            double value = black;
            if (!frame.samplesFloat.empty()) {
                value = static_cast<double>(frame.samplesFloat[offset]);
                if (!std::isfinite(value)) {
                    value = black;
                }
            }
            const int scaled = static_cast<int>((value - black) * scale + 0.5);
            destination[column] = static_cast<uint8_t>(std::clamp(scaled, 0, 255));
        }
    }
    return renderTimer.elapsed();
}

QList<int> sampleFrameIndices(int frameCount, int requestedSamples)
{
    QList<int> indices;
    if (frameCount <= 0 || requestedSamples <= 0) {
        return indices;
    }
    if (requestedSamples == 1 || frameCount == 1) {
        indices.append(0);
        return indices;
    }

    const int sampleCount = std::min(frameCount, requestedSamples);
    for (int sample = 0; sample < sampleCount; ++sample) {
        const int frameIndex = static_cast<int>(
            std::llround(static_cast<double>(frameCount - 1) * static_cast<double>(sample)
                         / static_cast<double>(sampleCount - 1)));
        if (!indices.contains(frameIndex)) {
            indices.append(frameIndex);
        }
    }
    return indices;
}

int runProbeRenderFrame(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    const QString path = args.at(0);
    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }

    bool frameOk = false;
    const int frameIndex = args.size() >= 2 ? args.at(1).toInt(&frameOk) : std::max(0, info.frameCount - 1);
    if (args.size() < 2) {
        frameOk = true;
    }
    if (!frameOk || frameIndex < 0 || frameIndex >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Frame index must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 4;
    }

    const TiffFrameResult frame = TiffStack::readFrame(info, frameIndex);
    if (!frame.ok || !frame.hasSamples()) {
        QTextStream(stderr) << (frame.error.isEmpty() ? QStringLiteral("Frame has no samples.") : frame.error)
                            << Qt::endl;
        return 5;
    }

    QString renderError;
    const qint64 renderMs = renderFrameToGrayscaleMs(frame, &renderError);
    if (renderMs < 0) {
        QTextStream(stderr) << renderError << Qt::endl;
        return 6;
    }

    const double black = std::isfinite(frame.observedMin) ? frame.observedMin : 0.0;
    double white = std::isfinite(frame.observedMax) ? frame.observedMax : black + 1.0;
    if (white <= black) {
        white = black + 1.0;
    }
    QTextStream(stdout)
        << QStringLiteral("frame=%1").arg(frameIndex) << Qt::endl
        << QStringLiteral("frames=%1").arg(info.frameCount) << Qt::endl
        << QStringLiteral("size=%1x%2").arg(frame.width).arg(frame.height) << Qt::endl
        << QStringLiteral("pixels=%1").arg(static_cast<qulonglong>(frame.width) * static_cast<qulonglong>(frame.height))
        << Qt::endl
        << QStringLiteral("read_ms=%1").arg(frame.elapsedMs) << Qt::endl
        << QStringLiteral("render_ms=%1").arg(renderMs) << Qt::endl
        << QStringLiteral("indexed=%1").arg(frame.usedDirectoryOffset ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl
        << QStringLiteral("levels=%1..%2").arg(black).arg(white) << Qt::endl;
    return 0;
}

int runProbeFolderScan(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }

    const QString folderPath = args.at(0);
    const bool recursive = !args.contains(QStringLiteral("--flat"));
    QElapsedTimer timer;
    timer.start();
    const QStringList paths = MainWindow::scanFolderPaths(folderPath, recursive);
    const qint64 elapsedMs = timer.elapsed();
    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF files were found.") << Qt::endl;
        return 3;
    }

    QTextStream out(stdout);
    out << QStringLiteral("folder=%1").arg(QFileInfo(folderPath).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("recursive=%1").arg(recursive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("tiffs=%1").arg(paths.size()) << Qt::endl;
    out << QStringLiteral("scan_ms=%1").arg(elapsedMs) << Qt::endl;
    out << QStringLiteral("first=%1").arg(paths.first()) << Qt::endl;
    out << QStringLiteral("last=%1").arg(paths.last()) << Qt::endl;
    return 0;
}

int runProbeCompatFolder(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }

    QString folderPath;
    bool recursive = true;
    bool fullIndex = false;
    int limit = 100;
    int maxPreviewMsAllowed = 1000;
    int maxFirstFrameMsAllowed = 1000;
    int maxRenderMsAllowed = 1000;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--flat")) {
            recursive = false;
        } else if (option == QStringLiteral("--full-index")) {
            fullIndex = true;
        } else if (option == QStringLiteral("--limit")) {
            if (!requirePositiveInt(option, &limit)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-preview-ms")) {
            if (!requirePositiveInt(option, &maxPreviewMsAllowed)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-first-frame-ms")) {
            if (!requirePositiveInt(option, &maxFirstFrameMsAllowed)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-render-ms")) {
            if (!requirePositiveInt(option, &maxRenderMsAllowed)) {
                return 2;
            }
        } else if (folderPath.isEmpty()) {
            folderPath = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (folderPath.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }
    if (!QFileInfo(folderPath).isDir()) {
        QTextStream(stderr) << QStringLiteral("Folder does not exist: %1").arg(folderPath) << Qt::endl;
        return 3;
    }

    QElapsedTimer scanTimer;
    scanTimer.start();
    QStringList paths = MainWindow::scanFolderPaths(folderPath, recursive);
    const qint64 scanMs = scanTimer.elapsed();
    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF files were found.") << Qt::endl;
        return 4;
    }

    const int foundCount = paths.size();
    if (paths.size() > limit) {
        paths = paths.mid(0, limit);
    }

    int metadataOk = 0;
    int metadataErrors = 0;
    int firstFrameOk = 0;
    int firstFrameErrors = 0;
    int renderOk = 0;
    int renderErrors = 0;
    int slowPreview = 0;
    int slowFirstFrame = 0;
    int slowRender = 0;
    qint64 maxPreviewMs = -1;
    qint64 maxFullInfoMs = -1;
    qint64 maxFirstFrameMs = -1;
    qint64 maxRenderMs = -1;
    qint64 totalPreviewMs = 0;
    qint64 totalFirstFrameMs = 0;
    qint64 totalRenderMs = 0;
    int maxFrameCount = 0;
    qulonglong maxPixels = 0;
    QString maxPreviewPath;
    QString maxFullInfoPath;
    QString maxFirstFramePath;
    QString maxRenderPath;

    QTextStream out(stdout);
    out << QStringLiteral("folder=%1").arg(QFileInfo(folderPath).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("recursive=%1").arg(recursive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("full_index=%1").arg(fullIndex ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("tiffs_found=%1").arg(foundCount) << Qt::endl;
    out << QStringLiteral("tiffs_checked=%1").arg(paths.size()) << Qt::endl;
    out << QStringLiteral("scan_ms=%1").arg(scanMs) << Qt::endl;

    for (int pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        const QString path = paths.at(pathIndex);
        TiffStackInfo info;
        QString error;
        QElapsedTimer previewTimer;
        previewTimer.start();
        const bool previewOk = TiffStack::readPreviewInfo(path, &info, &error);
        const qint64 previewMs = previewTimer.elapsed();
        totalPreviewMs += previewMs;
        if (maxPreviewPath.isEmpty() || previewMs > maxPreviewMs) {
            maxPreviewMs = previewMs;
            maxPreviewPath = path;
        }
        if (previewMs > maxPreviewMsAllowed) {
            ++slowPreview;
        }

        if (!previewOk) {
            ++metadataErrors;
            out << QStringLiteral("[%1] status=metadata_error preview_ms=%2 path=%3")
                       .arg(pathIndex + 1)
                       .arg(previewMs)
                       .arg(QFileInfo(path).absoluteFilePath())
                << Qt::endl;
            out << QStringLiteral("[%1] error=%2").arg(pathIndex + 1).arg(error) << Qt::endl;
            continue;
        }

        ++metadataOk;
        qint64 fullInfoMs = -1;
        if (fullIndex) {
            TiffStackInfo fullInfo;
            QString fullError;
            QElapsedTimer fullTimer;
            fullTimer.start();
            const bool fullOk = TiffStack::readInfo(path, &fullInfo, &fullError);
            fullInfoMs = fullTimer.elapsed();
            if (maxFullInfoPath.isEmpty() || fullInfoMs > maxFullInfoMs) {
                maxFullInfoMs = fullInfoMs;
                maxFullInfoPath = path;
            }
            if (fullOk) {
                info = fullInfo;
            } else {
                out << QStringLiteral("[%1] warning=full_index_error full_index_ms=%2 error=%3")
                           .arg(pathIndex + 1)
                           .arg(fullInfoMs)
                           .arg(fullError)
                    << Qt::endl;
            }
        }

        maxFrameCount = std::max(maxFrameCount, info.frameCount);
        maxPixels = std::max(
            maxPixels,
            static_cast<qulonglong>(std::max(0, info.width))
                * static_cast<qulonglong>(std::max(0, info.height)));

        const TiffFrameResult frame = TiffStack::readFrame(info, 0);
        if (!frame.ok || !frame.hasSamples()) {
            ++firstFrameErrors;
            out << QStringLiteral("[%1] status=first_frame_error path=%2")
                       .arg(pathIndex + 1)
                       .arg(QFileInfo(path).absoluteFilePath())
                << Qt::endl;
            out << QStringLiteral("[%1] frames=%2 size=%3x%4 pixel_type=%5 preview_ms=%6 full_index_ms=%7")
                       .arg(pathIndex + 1)
                       .arg(info.frameCount)
                       .arg(info.width)
                       .arg(info.height)
                       .arg(info.pixelType())
                       .arg(previewMs)
                       .arg(fullInfoMs)
                << Qt::endl;
            out << QStringLiteral("[%1] error=%2")
                       .arg(pathIndex + 1)
                       .arg(frame.error.isEmpty() ? QStringLiteral("Frame has no samples.") : frame.error)
                << Qt::endl;
            continue;
        }

        ++firstFrameOk;
        totalFirstFrameMs += frame.elapsedMs;
        if (maxFirstFramePath.isEmpty() || frame.elapsedMs > maxFirstFrameMs) {
            maxFirstFrameMs = frame.elapsedMs;
            maxFirstFramePath = path;
        }
        if (frame.elapsedMs > maxFirstFrameMsAllowed) {
            ++slowFirstFrame;
        }

        QString renderError;
        const qint64 renderMs = renderFrameToGrayscaleMs(frame, &renderError);
        if (renderMs < 0) {
            ++renderErrors;
            out << QStringLiteral("[%1] status=render_error path=%2")
                       .arg(pathIndex + 1)
                       .arg(QFileInfo(path).absoluteFilePath())
                << Qt::endl;
            out << QStringLiteral("[%1] error=%2").arg(pathIndex + 1).arg(renderError) << Qt::endl;
            continue;
        }

        ++renderOk;
        totalRenderMs += renderMs;
        if (maxRenderPath.isEmpty() || renderMs > maxRenderMs) {
            maxRenderMs = renderMs;
            maxRenderPath = path;
        }
        if (renderMs > maxRenderMsAllowed) {
            ++slowRender;
        }

        out << QStringLiteral("[%1] status=ok path=%2")
                   .arg(pathIndex + 1)
                   .arg(QFileInfo(path).absoluteFilePath())
            << Qt::endl;
        out << QStringLiteral("[%1] frames=%2 size=%3x%4 pixel_type=%5 bigtiff=%6 tiled=%7")
                   .arg(pathIndex + 1)
                   .arg(info.frameCount)
                   .arg(info.width)
                   .arg(info.height)
                   .arg(info.pixelType())
                   .arg(info.bigTiff ? QStringLiteral("yes") : QStringLiteral("no"))
                   .arg(info.tiled ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
        out << QStringLiteral("[%1] preview_ms=%2 full_index_ms=%3 first_frame_ms=%4 render_ms=%5")
                   .arg(pathIndex + 1)
                   .arg(previewMs)
                   .arg(fullInfoMs)
                   .arg(frame.elapsedMs)
                   .arg(renderMs)
            << Qt::endl;
    }

    const double avgPreviewMs = paths.isEmpty()
        ? 0.0
        : static_cast<double>(totalPreviewMs) / static_cast<double>(paths.size());
    const double avgFirstFrameMs = firstFrameOk <= 0
        ? 0.0
        : static_cast<double>(totalFirstFrameMs) / static_cast<double>(firstFrameOk);
    const double avgRenderMs = renderOk <= 0
        ? 0.0
        : static_cast<double>(totalRenderMs) / static_cast<double>(renderOk);
    const bool compatible = metadataErrors == 0 && firstFrameErrors == 0 && renderErrors == 0;
    const bool withinThresholds = slowPreview == 0 && slowFirstFrame == 0 && slowRender == 0;

    out << QStringLiteral("metadata_ok=%1").arg(metadataOk) << Qt::endl;
    out << QStringLiteral("metadata_errors=%1").arg(metadataErrors) << Qt::endl;
    out << QStringLiteral("first_frame_ok=%1").arg(firstFrameOk) << Qt::endl;
    out << QStringLiteral("first_frame_errors=%1").arg(firstFrameErrors) << Qt::endl;
    out << QStringLiteral("render_ok=%1").arg(renderOk) << Qt::endl;
    out << QStringLiteral("render_errors=%1").arg(renderErrors) << Qt::endl;
    out << QStringLiteral("slow_preview=%1").arg(slowPreview) << Qt::endl;
    out << QStringLiteral("slow_first_frame=%1").arg(slowFirstFrame) << Qt::endl;
    out << QStringLiteral("slow_render=%1").arg(slowRender) << Qt::endl;
    out << QStringLiteral("avg_preview_ms=%1").arg(avgPreviewMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("avg_first_frame_ms=%1").arg(avgFirstFrameMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("avg_render_ms=%1").arg(avgRenderMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("max_preview_ms=%1 path=%2").arg(maxPreviewMs).arg(QFileInfo(maxPreviewPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_full_index_ms=%1 path=%2")
               .arg(maxFullInfoMs)
               .arg(maxFullInfoPath.isEmpty() ? QString() : QFileInfo(maxFullInfoPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_first_frame_ms=%1 path=%2")
               .arg(maxFirstFrameMs)
               .arg(maxFirstFramePath.isEmpty() ? QString() : QFileInfo(maxFirstFramePath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_render_ms=%1 path=%2")
               .arg(maxRenderMs)
               .arg(maxRenderPath.isEmpty() ? QString() : QFileInfo(maxRenderPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_frame_count=%1").arg(maxFrameCount) << Qt::endl;
    out << QStringLiteral("max_pixels=%1").arg(maxPixels) << Qt::endl;
    out << QStringLiteral("compatible=%1").arg(compatible ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("within_thresholds=%1")
               .arg(withinThresholds ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (!compatible) {
        return 8;
    }
    return withinThresholds ? 0 : 9;
}

int runProbePerfFolder(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }

    const QString folderPath = args.at(0);
    if (!QFileInfo(folderPath).isDir()) {
        QTextStream(stderr) << QStringLiteral("Folder does not exist: %1").arg(folderPath) << Qt::endl;
        return 3;
    }

    bool recursive = true;
    int limit = 20;
    int framesPerStack = 3;
    int maxReadMsAllowed = 250;
    int maxRenderMsAllowed = 250;

    for (int index = 1; index < args.size(); ++index) {
        const QString option = args.at(index);
        if (option == QStringLiteral("--flat")) {
            recursive = false;
            continue;
        }

        if ((option == QStringLiteral("--limit")
             || option == QStringLiteral("--frames-per-stack")
             || option == QStringLiteral("--max-read-ms")
             || option == QStringLiteral("--max-render-ms"))
            && index + 1 < args.size()) {
            bool ok = false;
            const int value = args.at(index + 1).toInt(&ok);
            if (!ok || value <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(option, args.at(index + 1))
                                    << Qt::endl;
                return 4;
            }
            if (option == QStringLiteral("--limit")) {
                limit = value;
            } else if (option == QStringLiteral("--frames-per-stack")) {
                framesPerStack = value;
            } else if (option == QStringLiteral("--max-read-ms")) {
                maxReadMsAllowed = value;
            } else {
                maxRenderMsAllowed = value;
            }
            ++index;
            continue;
        }

        QTextStream(stderr) << QStringLiteral("Unknown or incomplete option: %1").arg(option) << Qt::endl;
        return 4;
    }

    QElapsedTimer scanTimer;
    scanTimer.start();
    QStringList paths = MainWindow::scanFolderPaths(folderPath, recursive);
    const qint64 scanMs = scanTimer.elapsed();
    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF files were found.") << Qt::endl;
        return 5;
    }

    const int foundCount = paths.size();
    if (paths.size() > limit) {
        paths = paths.mid(0, limit);
    }

    qint64 totalInfoMs = 0;
    qint64 totalReadMs = 0;
    qint64 totalRenderMs = 0;
    qint64 maxInfoMs = 0;
    qint64 maxReadMs = 0;
    qint64 maxRenderMs = 0;
    int measuredFrames = 0;
    int totalFrames = 0;
    qulonglong maxPixels = 0;
    QString maxInfoPath;
    QString maxReadPath;
    QString maxRenderPath;
    int maxReadFrame = 0;
    int maxRenderFrame = 0;

    QTextStream out(stdout);
    out << QStringLiteral("folder=%1").arg(QFileInfo(folderPath).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("recursive=%1").arg(recursive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("tiffs_found=%1").arg(foundCount) << Qt::endl;
    out << QStringLiteral("tiffs_measured=%1").arg(paths.size()) << Qt::endl;
    out << QStringLiteral("frames_per_stack=%1").arg(framesPerStack) << Qt::endl;
    out << QStringLiteral("scan_ms=%1").arg(scanMs) << Qt::endl;

    for (int pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        const QString path = paths.at(pathIndex);
        TiffStackInfo info;
        QString error;
        if (!TiffStack::readInfo(path, &info, &error)) {
            QTextStream(stderr) << QStringLiteral("Could not read TIFF metadata for %1: %2").arg(path, error)
                                << Qt::endl;
            return 6;
        }

        totalInfoMs += info.elapsedMs;
        totalFrames += info.frameCount;
        if (maxInfoPath.isEmpty() || info.elapsedMs > maxInfoMs) {
            maxInfoMs = info.elapsedMs;
            maxInfoPath = path;
        }
        maxPixels = std::max(
            maxPixels,
            static_cast<qulonglong>(std::max(0, info.width))
                * static_cast<qulonglong>(std::max(0, info.height)));

        const QList<int> sampledFrames = sampleFrameIndices(info.frameCount, framesPerStack);
        qint64 stackReadMs = 0;
        qint64 stackRenderMs = 0;
        qint64 stackMaxReadMs = 0;
        qint64 stackMaxRenderMs = 0;
        for (int frameIndex : sampledFrames) {
            const TiffFrameResult frame = TiffStack::readFrame(info, frameIndex);
            if (!frame.ok || !frame.hasSamples()) {
                QTextStream(stderr) << QStringLiteral("Could not read frame %1 for %2: %3")
                                           .arg(frameIndex)
                                           .arg(path, frame.error)
                                    << Qt::endl;
                return 7;
            }

            QString renderError;
            const qint64 renderMs = renderFrameToGrayscaleMs(frame, &renderError);
            if (renderMs < 0) {
                QTextStream(stderr) << QStringLiteral("Could not render frame %1 for %2: %3")
                                           .arg(frameIndex)
                                           .arg(path, renderError)
                                    << Qt::endl;
                return 8;
            }

            ++measuredFrames;
            totalReadMs += frame.elapsedMs;
            totalRenderMs += renderMs;
            stackReadMs += frame.elapsedMs;
            stackRenderMs += renderMs;
            stackMaxReadMs = std::max(stackMaxReadMs, frame.elapsedMs);
            stackMaxRenderMs = std::max(stackMaxRenderMs, renderMs);
            if (maxReadPath.isEmpty() || frame.elapsedMs > maxReadMs) {
                maxReadMs = frame.elapsedMs;
                maxReadPath = path;
                maxReadFrame = frameIndex;
            }
            if (maxRenderPath.isEmpty() || renderMs > maxRenderMs) {
                maxRenderMs = renderMs;
                maxRenderPath = path;
                maxRenderFrame = frameIndex;
            }
        }

        const double stackReadAvg = sampledFrames.isEmpty()
            ? 0.0
            : static_cast<double>(stackReadMs) / static_cast<double>(sampledFrames.size());
        const double stackRenderAvg = sampledFrames.isEmpty()
            ? 0.0
            : static_cast<double>(stackRenderMs) / static_cast<double>(sampledFrames.size());
        out << QStringLiteral("[%1] path=%2").arg(pathIndex + 1).arg(QFileInfo(path).absoluteFilePath())
            << Qt::endl;
        out << QStringLiteral("[%1] frames=%2 size=%3x%4 pixel_type=%5 indexed=%6 samples=%7")
                   .arg(pathIndex + 1)
                   .arg(info.frameCount)
                   .arg(info.width)
                   .arg(info.height)
                   .arg(info.pixelType())
                   .arg(info.hasDirectoryOffsets() ? QStringLiteral("yes") : QStringLiteral("no"))
                   .arg(sampledFrames.size())
            << Qt::endl;
        out << QStringLiteral("[%1] bigtiff=%2 tiled=%3")
                   .arg(pathIndex + 1)
                   .arg(info.bigTiff ? QStringLiteral("yes") : QStringLiteral("no"))
                   .arg(info.tiled ? QStringLiteral("yes") : QStringLiteral("no"))
            << Qt::endl;
        out << QStringLiteral("[%1] info_ms=%2 read_avg_ms=%3 read_max_ms=%4 render_avg_ms=%5 render_max_ms=%6")
                   .arg(pathIndex + 1)
                   .arg(info.elapsedMs)
                   .arg(stackReadAvg, 0, 'f', 2)
                   .arg(stackMaxReadMs)
                   .arg(stackRenderAvg, 0, 'f', 2)
                   .arg(stackMaxRenderMs)
            << Qt::endl;
    }

    const double avgInfoMs = paths.isEmpty()
        ? 0.0
        : static_cast<double>(totalInfoMs) / static_cast<double>(paths.size());
    const double avgReadMs = measuredFrames <= 0
        ? 0.0
        : static_cast<double>(totalReadMs) / static_cast<double>(measuredFrames);
    const double avgRenderMs = measuredFrames <= 0
        ? 0.0
        : static_cast<double>(totalRenderMs) / static_cast<double>(measuredFrames);
    const bool readOk = maxReadMs <= maxReadMsAllowed;
    const bool renderOk = maxRenderMs <= maxRenderMsAllowed;

    out << QStringLiteral("measured_frames=%1").arg(measuredFrames) << Qt::endl;
    out << QStringLiteral("total_stack_frames=%1").arg(totalFrames) << Qt::endl;
    out << QStringLiteral("max_pixels=%1").arg(maxPixels) << Qt::endl;
    out << QStringLiteral("avg_info_ms=%1").arg(avgInfoMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("avg_read_ms=%1").arg(avgReadMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("avg_render_ms=%1").arg(avgRenderMs, 0, 'f', 2) << Qt::endl;
    out << QStringLiteral("max_info_ms=%1 path=%2").arg(maxInfoMs).arg(QFileInfo(maxInfoPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_read_ms=%1 frame=%2 path=%3")
               .arg(maxReadMs)
               .arg(maxReadFrame)
               .arg(QFileInfo(maxReadPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_render_ms=%1 frame=%2 path=%3")
               .arg(maxRenderMs)
               .arg(maxRenderFrame)
               .arg(QFileInfo(maxRenderPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("max_read_allowed_ms=%1").arg(maxReadMsAllowed) << Qt::endl;
    out << QStringLiteral("max_render_allowed_ms=%1").arg(maxRenderMsAllowed) << Qt::endl;
    out << QStringLiteral("within_thresholds=%1")
               .arg(readOk && renderOk ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    return readOk && renderOk ? 0 : 9;
}

int runProbeInfoCancel(const QString& path)
{
    int checks = 0;
    TiffStackInfo info;
    QString error;
    const bool ok = TiffStack::readInfo(path, &info, &error, [&checks]() {
        ++checks;
        return checks >= 2;
    });
    if (ok) {
        QTextStream(stderr) << QStringLiteral("Metadata scan was not cancelled.") << Qt::endl;
        return 3;
    }

    QTextStream(stdout)
        << QStringLiteral("cancelled=%1").arg(error.contains(QStringLiteral("cancelled")) ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl
        << QStringLiteral("checks=%1").arg(checks) << Qt::endl
        << QStringLiteral("error=%1").arg(error) << Qt::endl;
    return error.contains(QStringLiteral("cancelled")) ? 0 : 4;
}

int runProbeFrameCancel(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    const QString path = args.at(0);
    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }

    bool frameOk = false;
    const int frameIndex = args.size() >= 2 ? args.at(1).toInt(&frameOk) : 0;
    if (args.size() < 2) {
        frameOk = true;
    }
    if (!frameOk || frameIndex < 0 || frameIndex >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Frame index must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 4;
    }

    int checks = 0;
    const TiffFrameResult frame = TiffStack::readFrame(info, frameIndex, [&checks]() {
        ++checks;
        return checks >= 4;
    });
    if (frame.ok || !frame.cancelled) {
        QTextStream(stderr) << QStringLiteral("Frame load was not cancelled.") << Qt::endl;
        return 5;
    }

    QTextStream(stdout)
        << QStringLiteral("cancelled=%1").arg(frame.cancelled ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl
        << QStringLiteral("checks=%1").arg(checks) << Qt::endl
        << QStringLiteral("error=%1").arg(frame.error) << Qt::endl
        << QStringLiteral("elapsed_ms=%1").arg(frame.elapsedMs) << Qt::endl;
    return 0;
}

int runProbeManifestWorking(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No manifest path was provided.") << Qt::endl;
        return 2;
    }

    QString error;
    const QString currentSourcePath = args.size() >= 2 ? args.at(1) : QString();
    const QString relativeRootPath = args.size() >= 3 ? args.at(2) : QString();
    const QString workingPath =
        MainWindow::findWorkingTiffFromManifest(args.at(0), currentSourcePath, &error, relativeRootPath);
    if (!error.isEmpty()) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }
    if (workingPath.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No different working TIFF was found.") << Qt::endl;
        return 4;
    }

    QTextStream(stdout) << QStringLiteral("working_tiff=%1").arg(workingPath) << Qt::endl;
    return 0;
}

int runProbeManifestDisplay(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No manifest path was provided.") << Qt::endl;
        return 2;
    }

    QString error;
    const QString currentSourcePath = args.size() >= 2 ? args.at(1) : QString();
    const QString relativeRootPath = args.size() >= 3 ? args.at(2) : QString();
    const QString displayPath =
        MainWindow::findPreferredDisplayPathFromManifest(args.at(0), currentSourcePath, &error, relativeRootPath);
    if (!error.isEmpty()) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }
    if (displayPath.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No preferred display output was found.") << Qt::endl;
        return 4;
    }

    QTextStream(stdout) << QStringLiteral("display_path=%1").arg(displayPath) << Qt::endl;
    return 0;
}

int runProbeManifestResults(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No manifest path was provided.") << Qt::endl;
        return 2;
    }

    QString error;
    const QString relativeRootPath = args.size() >= 2 ? args.at(1) : QString();
    const QStringList paths = MainWindow::findResultPathsFromManifest(args.at(0), &error, relativeRootPath);
    if (!error.isEmpty()) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }
    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No result paths were found.") << Qt::endl;
        return 4;
    }

    QTextStream out(stdout);
    out << QStringLiteral("result_count=%1").arg(paths.size()) << Qt::endl;
    for (const QString& path : paths) {
        out << QStringLiteral("result=%1").arg(path) << Qt::endl;
    }
    return 0;
}

int runProbeProgressLine(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No progress line was provided.") << Qt::endl;
        return 2;
    }

    const QString line = args.join(QLatin1Char(' '));
    QString stage;
    int done = 0;
    int total = 0;
    if (!MainWindow::parseAnalysisProgressLine(line, &stage, &done, &total)) {
        QTextStream(stderr) << QStringLiteral("No progress was parsed.") << Qt::endl;
        return 3;
    }

    QTextStream(stdout)
        << QStringLiteral("stage=%1").arg(stage) << Qt::endl
        << QStringLiteral("done=%1").arg(done) << Qt::endl
        << QStringLiteral("total=%1").arg(total) << Qt::endl;
    return 0;
}

int runProbeResultOpenMode(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No result path was provided.") << Qt::endl;
        return 2;
    }

    QElapsedTimer timer;
    timer.start();
    const MainWindow::ResultOpenMode mode = MainWindow::resultOpenModeForPath(args.at(0));
    QTextStream(stdout)
        << QStringLiteral("mode=%1").arg(MainWindow::resultOpenModeName(mode)) << Qt::endl
        << QStringLiteral("elapsed_ms=%1").arg(timer.elapsed()) << Qt::endl;
    return mode == MainWindow::ResultOpenMode::Missing ? 3 : 0;
}

int runProbeResultActionLabel(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No result path was provided.") << Qt::endl;
        return 2;
    }

    QElapsedTimer timer;
    timer.start();
    const MainWindow::ResultOpenMode mode = MainWindow::resultOpenModeForPath(args.at(0));
    QTextStream(stdout)
        << QStringLiteral("label=%1").arg(MainWindow::resultActionLabel(mode)) << Qt::endl
        << QStringLiteral("elapsed_ms=%1").arg(timer.elapsed()) << Qt::endl;
    return mode == MainWindow::ResultOpenMode::Missing ? 3 : 0;
}

int runProbeImage(const QString& path)
{
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage image = reader.read();
    if (image.isNull()) {
        QTextStream(stderr) << reader.errorString() << Qt::endl;
        return 2;
    }

    QTextStream(stdout)
        << QStringLiteral("path=%1").arg(QFileInfo(path).absoluteFilePath()) << Qt::endl
        << QStringLiteral("size=%1x%2").arg(image.width()).arg(image.height()) << Qt::endl
        << QStringLiteral("format=%1").arg(QString::fromLatin1(reader.format())) << Qt::endl;
    return 0;
}

int runProbeText(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QTextStream(stderr) << file.errorString() << Qt::endl;
        return 2;
    }

    const QByteArray bytes = file.read(4096);
    const QString text = QString::fromUtf8(bytes);
    const QString firstLine = text.split(QLatin1Char('\n')).value(0).trimmed();
    QTextStream(stdout)
        << QStringLiteral("path=%1").arg(QFileInfo(path).absoluteFilePath()) << Qt::endl
        << QStringLiteral("bytes=%1").arg(QFileInfo(path).size()) << Qt::endl
        << QStringLiteral("first_line=%1").arg(firstLine) << Qt::endl;
    return 0;
}

int runProbeResultLogLine(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No log line was provided.") << Qt::endl;
        return 2;
    }

    const QString line = args.join(QLatin1Char(' '));
    const QString path = MainWindow::resultPathFromLogLine(line, QDir::currentPath());
    if (path.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No result path was parsed.") << Qt::endl;
        return 3;
    }

    QTextStream(stdout) << QStringLiteral("result=%1").arg(path) << Qt::endl;
    return 0;
}

int runProbeMainWindowConstruct(const QStringList&)
{
    QElapsedTimer elapsed;
    elapsed.start();
    {
        QTextStream(stdout) << QStringLiteral("constructing=yes") << Qt::endl;
        MainWindow window;
        QTextStream(stdout) << QStringLiteral("constructed=yes") << Qt::endl;
        QTextStream(stdout) << QStringLiteral("construct_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    }
    QTextStream(stdout) << QStringLiteral("destroyed=yes") << Qt::endl;
    QTextStream(stdout) << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    return 0;
}

int runProbeQueueUi(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }

    const QString folderPath = args.at(0);
    if (!QFileInfo(folderPath).isDir()) {
        QTextStream(stderr) << QStringLiteral("Folder does not exist: %1").arg(folderPath) << Qt::endl;
        return 3;
    }

    const bool recursive = !args.contains(QStringLiteral("--flat"));
    const bool requireFirstDisplay = args.contains(QStringLiteral("--require-first-display"));
    const bool directPaths = args.contains(QStringLiteral("--direct-paths"));
    int timeoutMs = 15000;
    int maxAllowedGapMs = 250;
    for (int index = 1; index < args.size(); ++index) {
        const QString option = args.at(index);
        if ((option == QStringLiteral("--timeout-ms") || option == QStringLiteral("--max-gap-ms"))
            && index + 1 < args.size()) {
            bool ok = false;
            const int value = args.at(index + 1).toInt(&ok);
            if (!ok || value <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(option, args.at(index + 1))
                                    << Qt::endl;
                return 4;
            }
            if (option == QStringLiteral("--timeout-ms")) {
                timeoutMs = value;
            } else {
                maxAllowedGapMs = value;
            }
            ++index;
        } else if (option == QStringLiteral("--flat")
                   || option == QStringLiteral("--require-first-display")
                   || option == QStringLiteral("--direct-paths")) {
            continue;
        }
    }

    const QStringList expectedPaths = MainWindow::scanFolderPaths(folderPath, recursive);
    if (expectedPaths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF files were found.") << Qt::endl;
        return 5;
    }

    MainWindow window;
    window.setRecursiveFolderScan(recursive);

    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    qint64 firstDisplayMs = -1;
    qint64 queueDoneMs = -1;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (firstDisplayMs < 0 && window.viewerHasDisplayedFrame()) {
            firstDisplayMs = elapsed.elapsed();
        }
        const bool queueDone = !window.isFolderScanPending() && !window.isQueuePopulationPending();
        if (queueDone && queueDoneMs < 0) {
            queueDoneMs = elapsed.elapsed();
        }
        if (queueDone && (!requireFirstDisplay || firstDisplayMs >= 0)) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    if (directPaths) {
        window.openPaths(expectedPaths);
    } else {
        window.openFolder(folderPath);
    }
    QCoreApplication::exec();

    const int queuedCount = window.queuedTiffCount();
    const bool countMatches = queuedCount == expectedPaths.size();
    const bool responsive = maxGapMs <= maxAllowedGapMs;
    const bool firstDisplayed = firstDisplayMs >= 0 || window.viewerHasDisplayedFrame();
    if (firstDisplayMs < 0 && firstDisplayed) {
        firstDisplayMs = elapsed.elapsed();
    }

    QTextStream out(stdout);
    out << QStringLiteral("folder=%1").arg(QFileInfo(folderPath).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("source=%1").arg(directPaths ? QStringLiteral("paths") : QStringLiteral("folder"))
        << Qt::endl;
    out << QStringLiteral("recursive=%1").arg(recursive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("expected=%1").arg(expectedPaths.size()) << Qt::endl;
    out << QStringLiteral("queued=%1").arg(queuedCount) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("queue_done_ms=%1").arg(queueDoneMs) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("first_displayed=%1").arg(firstDisplayed ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("first_display_ms=%1").arg(firstDisplayMs) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("status=%1").arg(window.queueStatusText()) << Qt::endl;
    out << QStringLiteral("folder_pending=%1")
               .arg(window.isFolderScanPending() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("queue_pending=%1")
               .arg(window.isQueuePopulationPending() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for queue population.") << Qt::endl;
        return 6;
    }
    if (!countMatches) {
        QTextStream(stderr) << QStringLiteral("Queued TIFF count did not match scan result.") << Qt::endl;
        return 7;
    }
    if (requireFirstDisplay && !firstDisplayed) {
        QTextStream(stderr) << QStringLiteral("No frame was displayed after queue population.") << Qt::endl;
        return 8;
    }
    return responsive ? 0 : 9;
}

int runProbeViewerSwitchUi(const QStringList& args)
{
    QString folderPath;
    QStringList paths;
    bool recursive = true;
    int limit = 20;
    int rounds = 3;
    int intervalMs = 1;
    int timeoutMs = 15000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--folder")) {
            if (!requireValue(option, &folderPath)) {
                return 2;
            }
        } else if (option == QStringLiteral("--flat")) {
            recursive = false;
        } else if (option == QStringLiteral("--limit")) {
            if (!requirePositiveInt(option, &limit)) {
                return 2;
            }
        } else if (option == QStringLiteral("--rounds")) {
            if (!requirePositiveInt(option, &rounds)) {
                return 2;
            }
        } else if (option == QStringLiteral("--interval-ms")) {
            if (!requirePositiveInt(option, &intervalMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else {
            paths.append(option);
        }
    }

    if (!folderPath.isEmpty()) {
        if (!QFileInfo(folderPath).isDir()) {
            QTextStream(stderr) << QStringLiteral("Folder does not exist: %1").arg(folderPath) << Qt::endl;
            return 3;
        }
        paths = MainWindow::scanFolderPaths(folderPath, recursive);
        if (paths.size() > limit) {
            paths = paths.mid(0, limit);
        }
    }

    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF paths were provided.") << Qt::endl;
        return 4;
    }

    QStringList sequence;
    sequence.reserve(paths.size() * rounds);
    for (int round = 0; round < rounds; ++round) {
        sequence.append(paths);
    }

    for (const QString& path : sequence) {
        const QFileInfo info(path);
        const QString suffix = info.suffix().toLower();
        if (!info.isFile() || (suffix != QStringLiteral("tif") && suffix != QStringLiteral("tiff"))) {
            QTextStream(stderr) << QStringLiteral("Invalid TIFF path: %1").arg(path) << Qt::endl;
            return 5;
        }
    }

    TiffViewerWidget viewer;
    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    int scheduledLoads = 0;
    int nextLoadIndex = 0;
    int loadedSignals = 0;
    int renderedSignals = 0;
    bool timedOut = false;
    QString lastLoadedPath;
    QString lastRenderedPath;
    const QString finalPath = QFileInfo(sequence.last()).absoluteFilePath();

    QObject::connect(&viewer, &TiffViewerWidget::fileLoaded, [&](const QString& path) {
        ++loadedSignals;
        lastLoadedPath = QFileInfo(path).absoluteFilePath();
    });
    QObject::connect(&viewer, &TiffViewerWidget::frameRendered, [&](const QString& path, int frameIndex) {
        Q_UNUSED(frameIndex);
        ++renderedSignals;
        lastRenderedPath = QFileInfo(path).absoluteFilePath();
    });

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (scheduledLoads >= sequence.size()
            && lastLoadedPath.compare(finalPath, Qt::CaseInsensitive) == 0
            && lastRenderedPath.compare(finalPath, Qt::CaseInsensitive) == 0) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    QTimer switchTimer;
    switchTimer.setInterval(intervalMs);
    QObject::connect(&switchTimer, &QTimer::timeout, [&]() {
        if (nextLoadIndex >= sequence.size()) {
            switchTimer.stop();
            return;
        }

        viewer.loadFile(sequence.at(nextLoadIndex));
        ++scheduledLoads;
        ++nextLoadIndex;
        if (nextLoadIndex >= sequence.size()) {
            switchTimer.stop();
        }
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    switchTimer.start();
    QCoreApplication::exec();

    const bool finalLoaded = lastLoadedPath.compare(finalPath, Qt::CaseInsensitive) == 0;
    const QString currentPath = QFileInfo(viewer.currentFilePath()).absoluteFilePath();
    const bool finalCurrent = currentPath.compare(finalPath, Qt::CaseInsensitive) == 0;
    const bool finalRendered = lastRenderedPath.compare(finalPath, Qt::CaseInsensitive) == 0;
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("sequence=%1").arg(sequence.size()) << Qt::endl;
    out << QStringLiteral("unique_paths=%1").arg(paths.size()) << Qt::endl;
    out << QStringLiteral("rounds=%1").arg(rounds) << Qt::endl;
    out << QStringLiteral("scheduled=%1").arg(scheduledLoads) << Qt::endl;
    out << QStringLiteral("loaded_signals=%1").arg(loadedSignals) << Qt::endl;
    out << QStringLiteral("rendered_signals=%1").arg(renderedSignals) << Qt::endl;
    out << QStringLiteral("final_path=%1").arg(finalPath) << Qt::endl;
    out << QStringLiteral("current_path=%1").arg(currentPath) << Qt::endl;
    out << QStringLiteral("last_loaded=%1").arg(lastLoadedPath) << Qt::endl;
    out << QStringLiteral("last_rendered=%1").arg(lastRenderedPath) << Qt::endl;
    out << QStringLiteral("displayed_frame=%1").arg(viewer.hasDisplayedFrame() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("status=%1").arg(viewer.statusText()) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for final TIFF display.") << Qt::endl;
        return 6;
    }
    if (!finalLoaded || !finalCurrent || !finalRendered) {
        QTextStream(stderr) << QStringLiteral("Final TIFF did not settle in the viewer.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

int runProbeViewerOpenLatency(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    QString path;
    int timeoutMs = 20000;
    int maxFirstRenderMs = 1000;
    bool requirePreviewBeforeFull = false;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-first-render-ms")) {
            if (!requirePositiveInt(option, &maxFirstRenderMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--require-preview-before-full")) {
            requirePreviewBeforeFull = true;
        } else if (path.isEmpty()) {
            path = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (path.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }
    const QFileInfo fileInfo(path);
    if (!fileInfo.isFile()) {
        QTextStream(stderr) << QStringLiteral("TIFF path does not exist: %1").arg(path) << Qt::endl;
        return 3;
    }

    TiffViewerWidget viewer;
    QElapsedTimer elapsed;
    elapsed.start();
    qint64 firstRenderMs = -1;
    qint64 fullLoadMs = -1;
    bool timedOut = false;
    QString loadedPath;
    QString renderedPath;
    const QString expectedPath = fileInfo.absoluteFilePath();

    QObject::connect(&viewer, &TiffViewerWidget::fileLoaded, [&](const QString& loaded) {
        if (loadedPath.isEmpty()) {
            fullLoadMs = elapsed.elapsed();
            loadedPath = QFileInfo(loaded).absoluteFilePath();
        }
    });
    QObject::connect(&viewer, &TiffViewerWidget::frameRendered, [&](const QString& rendered, int frameIndex) {
        Q_UNUSED(frameIndex);
        if (renderedPath.isEmpty()) {
            firstRenderMs = elapsed.elapsed();
            renderedPath = QFileInfo(rendered).absoluteFilePath();
        }
    });

    QTimer poll;
    poll.setInterval(5);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (firstRenderMs >= 0 && fullLoadMs >= 0) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    poll.start();
    timeout.start(timeoutMs);
    viewer.loadFile(path);
    QCoreApplication::exec();

    const bool renderedExpected = renderedPath.compare(expectedPath, Qt::CaseInsensitive) == 0;
    const bool loadedExpected = loadedPath.compare(expectedPath, Qt::CaseInsensitive) == 0;
    const bool previewBeforeFull = firstRenderMs >= 0 && fullLoadMs >= 0 && firstRenderMs <= fullLoadMs;

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(expectedPath) << Qt::endl;
    out << QStringLiteral("first_render_ms=%1").arg(firstRenderMs) << Qt::endl;
    out << QStringLiteral("full_load_ms=%1").arg(fullLoadMs) << Qt::endl;
    out << QStringLiteral("preview_before_full=%1").arg(previewBeforeFull ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("rendered_path=%1").arg(renderedPath) << Qt::endl;
    out << QStringLiteral("loaded_path=%1").arg(loadedPath) << Qt::endl;
    out << QStringLiteral("status=%1").arg(viewer.statusText()) << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for first render and full load.") << Qt::endl;
        return 4;
    }
    if (!renderedExpected || !loadedExpected) {
        QTextStream(stderr) << QStringLiteral("Viewer did not render and load the expected TIFF.") << Qt::endl;
        return 5;
    }
    if (firstRenderMs > maxFirstRenderMs) {
        QTextStream(stderr) << QStringLiteral("First frame render exceeded latency budget.") << Qt::endl;
        return 6;
    }
    if (requirePreviewBeforeFull && !previewBeforeFull) {
        QTextStream(stderr) << QStringLiteral("First frame did not render before full indexing completed.") << Qt::endl;
        return 7;
    }
    return 0;
}

int runProbeViewerPrefetchUi(const QStringList& args)
{
    QString path;
    int targetFrame = -1;
    int primeFrame = -1;
    int pivotFrame = -1;
    int waitMs = 500;
    int timeoutMs = 20000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };
        auto requireNonNegativeInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed < 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--target-frame")) {
            if (!requireNonNegativeInt(option, &targetFrame)) {
                return 2;
            }
        } else if (option == QStringLiteral("--prime-frame")) {
            if (!requireNonNegativeInt(option, &primeFrame)) {
                return 2;
            }
        } else if (option == QStringLiteral("--pivot-frame")) {
            if (!requireNonNegativeInt(option, &pivotFrame)) {
                return 2;
            }
        } else if (option == QStringLiteral("--wait-ms")) {
            if (!requirePositiveInt(option, &waitMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else if (path.isEmpty()) {
            path = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (path.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }
    const QFileInfo fileInfo(path);
    if (!fileInfo.isFile()) {
        QTextStream(stderr) << QStringLiteral("TIFF path does not exist: %1").arg(path) << Qt::endl;
        return 3;
    }

    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 4;
    }
    if (info.frameCount < 2) {
        QTextStream(stderr) << QStringLiteral("Prefetch probe needs a multi-frame TIFF.") << Qt::endl;
        return 5;
    }
    if (targetFrame < 0 && pivotFrame > 0) {
        targetFrame = pivotFrame - 1;
    } else if (targetFrame < 0) {
        targetFrame = info.frameCount > 2 ? 2 : 1;
    }
    if (targetFrame <= 0 || targetFrame >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Target frame must be inside 1..%1.")
                                   .arg(std::max(1, info.frameCount - 1))
                            << Qt::endl;
        return 6;
    }
    if (primeFrame >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Prime frame must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 6;
    }
    if (pivotFrame >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Pivot frame must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 6;
    }

    TiffViewerWidget viewer;
    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    int renderedSignals = 0;
    bool firstFrameRendered = false;
    bool primeRequested = false;
    bool primeRendered = primeFrame < 0;
    bool pivotRequested = false;
    bool pivotRendered = pivotFrame < 0;
    bool targetRequestScheduled = false;
    bool targetRequested = false;
    bool targetRendered = false;
    bool timedOut = false;
    qint64 targetRenderMs = -1;
    QString targetStatus;
    const QString expectedPath = fileInfo.absoluteFilePath();
    QString renderedPath;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    const auto scheduleTargetRequest = [&]() {
        if (targetRequestScheduled || targetRequested) {
            return;
        }
        targetRequestScheduled = true;
        QTimer::singleShot(waitMs, &viewer, [&]() {
            if (!targetRequested) {
                targetRequested = true;
                viewer.setCurrentFrameIndex(targetFrame);
            }
        });
    };

    QObject::connect(&viewer, &TiffViewerWidget::frameRendered, [&](const QString& rendered, int frameIndex) {
        ++renderedSignals;
        renderedPath = QFileInfo(rendered).absoluteFilePath();
        if (renderedPath.compare(expectedPath, Qt::CaseInsensitive) != 0) {
            return;
        }
        if (frameIndex == 0 && !firstFrameRendered) {
            firstFrameRendered = true;
            if (primeFrame >= 0) {
                primeRequested = true;
                viewer.setCurrentFrameIndex(primeFrame);
            } else if (pivotFrame >= 0) {
                pivotRequested = true;
                viewer.setCurrentFrameIndex(pivotFrame);
            } else {
                scheduleTargetRequest();
            }
        }
        if (primeRequested && !primeRendered && frameIndex == primeFrame) {
            primeRendered = true;
            if (pivotFrame >= 0) {
                pivotRequested = true;
                viewer.setCurrentFrameIndex(pivotFrame);
            } else {
                scheduleTargetRequest();
            }
        }
        if (pivotRequested && !pivotRendered && frameIndex == pivotFrame) {
            pivotRendered = true;
            scheduleTargetRequest();
        }
        if (targetRequested && frameIndex == targetFrame) {
            targetRendered = true;
            targetRenderMs = elapsed.elapsed();
            targetStatus = viewer.statusText();
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    timeout.start(timeoutMs);
    viewer.loadFile(path);
    QCoreApplication::exec();

    const bool renderedExpected = renderedPath.compare(expectedPath, Qt::CaseInsensitive) == 0;
    const bool prefetchHit = targetStatus.contains(QStringLiteral("prefetch"), Qt::CaseInsensitive);
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(expectedPath) << Qt::endl;
    out << QStringLiteral("frames=%1").arg(info.frameCount) << Qt::endl;
    out << QStringLiteral("prime_frame=%1").arg(primeFrame) << Qt::endl;
    out << QStringLiteral("pivot_frame=%1").arg(pivotFrame) << Qt::endl;
    out << QStringLiteral("target_frame=%1").arg(targetFrame) << Qt::endl;
    out << QStringLiteral("wait_ms=%1").arg(waitMs) << Qt::endl;
    out << QStringLiteral("prime_requested=%1").arg(primeRequested ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("prime_rendered=%1").arg(primeRendered ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("pivot_requested=%1").arg(pivotRequested ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("pivot_rendered=%1").arg(pivotRendered ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("target_requested=%1").arg(targetRequested ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("target_rendered=%1").arg(targetRendered ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("target_render_ms=%1").arg(targetRenderMs) << Qt::endl;
    out << QStringLiteral("rendered_signals=%1").arg(renderedSignals) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("status=%1").arg(targetStatus) << Qt::endl;
    out << QStringLiteral("prefetch_hit=%1").arg(prefetchHit ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for prefetch target frame.") << Qt::endl;
        return 7;
    }
    if (!renderedExpected || !primeRendered || !pivotRendered || !targetRendered) {
        QTextStream(stderr) << QStringLiteral("Viewer did not render the expected prefetch target frame.") << Qt::endl;
        return 8;
    }
    if (!prefetchHit) {
        QTextStream(stderr) << QStringLiteral("Target frame was not served from prefetch cache.") << Qt::endl;
        return 9;
    }
    return responsive ? 0 : 10;
}

int runProbeViewerScrubUi(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    QString path;
    int events = 80;
    int intervalMs = 1;
    int timeoutMs = 20000;
    int maxAllowedGapMs = 250;
    int finalFrame = -1;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };
        auto requireNonNegativeInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed < 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--events")) {
            if (!requirePositiveInt(option, &events)) {
                return 2;
            }
        } else if (option == QStringLiteral("--interval-ms")) {
            if (!requirePositiveInt(option, &intervalMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--final-frame")) {
            if (!requireNonNegativeInt(option, &finalFrame)) {
                return 2;
            }
        } else if (path.isEmpty()) {
            path = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (path.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    TiffStackInfo info;
    QString error;
    if (!TiffStack::readInfo(path, &info, &error)) {
        QTextStream(stderr) << error << Qt::endl;
        return 3;
    }
    if (info.frameCount <= 0) {
        QTextStream(stderr) << QStringLiteral("TIFF has no frames.") << Qt::endl;
        return 4;
    }

    if (finalFrame < 0) {
        finalFrame = std::max(0, info.frameCount - 1);
    }
    if (finalFrame >= info.frameCount) {
        QTextStream(stderr) << QStringLiteral("Final frame must be inside 0..%1.")
                                   .arg(std::max(0, info.frameCount - 1))
                            << Qt::endl;
        return 5;
    }

    QList<int> sequence;
    sequence.reserve(events + 1);
    for (int event = 0; event < events; ++event) {
        int frameIndex = 0;
        if (info.frameCount == 1) {
            frameIndex = 0;
        } else if (event % 4 == 0) {
            frameIndex = 0;
        } else if (event % 4 == 1) {
            frameIndex = info.frameCount - 1;
        } else if (event % 4 == 2) {
            frameIndex = (event * 7) % info.frameCount;
        } else {
            frameIndex = info.frameCount - 1 - ((event * 11) % info.frameCount);
        }
        sequence.append(std::clamp(frameIndex, 0, info.frameCount - 1));
    }
    sequence.append(finalFrame);

    TiffViewerWidget viewer;
    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    int loadedSignals = 0;
    int renderedSignals = 0;
    int scheduledFrames = 0;
    int nextFrameIndex = 0;
    int lastRenderedFrame = -1;
    bool startedScrub = false;
    bool timedOut = false;
    const QString expectedPath = QFileInfo(path).absoluteFilePath();
    QString loadedPath;
    QString renderedPath;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer scrubTimer;
    scrubTimer.setInterval(intervalMs);
    QObject::connect(&scrubTimer, &QTimer::timeout, [&]() {
        if (nextFrameIndex >= sequence.size()) {
            scrubTimer.stop();
            return;
        }
        viewer.setCurrentFrameIndex(sequence.at(nextFrameIndex));
        ++scheduledFrames;
        ++nextFrameIndex;
        if (nextFrameIndex >= sequence.size()) {
            scrubTimer.stop();
        }
    });

    QObject::connect(&viewer, &TiffViewerWidget::fileLoaded, [&](const QString& loaded) {
        ++loadedSignals;
        loadedPath = QFileInfo(loaded).absoluteFilePath();
        if (!startedScrub && loadedPath.compare(expectedPath, Qt::CaseInsensitive) == 0) {
            startedScrub = true;
            scrubTimer.start();
        }
    });

    QObject::connect(&viewer, &TiffViewerWidget::frameRendered, [&](const QString& rendered, int frameIndex) {
        ++renderedSignals;
        renderedPath = QFileInfo(rendered).absoluteFilePath();
        lastRenderedFrame = frameIndex;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (scheduledFrames >= sequence.size()
            && renderedPath.compare(expectedPath, Qt::CaseInsensitive) == 0
            && viewer.currentFrameIndex() == finalFrame
            && lastRenderedFrame == finalFrame) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    viewer.loadFile(path);
    QCoreApplication::exec();

    const bool loadedExpected = loadedPath.compare(expectedPath, Qt::CaseInsensitive) == 0;
    const bool renderedExpected = renderedPath.compare(expectedPath, Qt::CaseInsensitive) == 0;
    const bool finalCurrent = viewer.currentFrameIndex() == finalFrame;
    const bool finalRendered = lastRenderedFrame == finalFrame;
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(expectedPath) << Qt::endl;
    out << QStringLiteral("frames=%1").arg(info.frameCount) << Qt::endl;
    out << QStringLiteral("sequence=%1").arg(sequence.size()) << Qt::endl;
    out << QStringLiteral("scheduled=%1").arg(scheduledFrames) << Qt::endl;
    out << QStringLiteral("loaded_signals=%1").arg(loadedSignals) << Qt::endl;
    out << QStringLiteral("rendered_signals=%1").arg(renderedSignals) << Qt::endl;
    out << QStringLiteral("final_frame=%1").arg(finalFrame) << Qt::endl;
    out << QStringLiteral("current_frame=%1").arg(viewer.currentFrameIndex()) << Qt::endl;
    out << QStringLiteral("last_rendered_frame=%1").arg(lastRenderedFrame) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("status=%1").arg(viewer.statusText()) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for final scrub frame.") << Qt::endl;
        return 6;
    }
    if (!loadedExpected || !renderedExpected || !finalCurrent || !finalRendered) {
        QTextStream(stderr) << QStringLiteral("Viewer did not settle on the final scrub frame.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

int runProbeMainWindowBenchmark(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No folder path was provided.") << Qt::endl;
        return 2;
    }

    const QString folderPath = args.at(0);
    if (!QFileInfo(folderPath).isDir()) {
        QTextStream(stderr) << QStringLiteral("Folder does not exist: %1").arg(folderPath) << Qt::endl;
        return 3;
    }

    const bool recursive = !args.contains(QStringLiteral("--flat"));
    int timeoutMs = 30000;
    int maxAllowedGapMs = 250;
    for (int index = 1; index < args.size(); ++index) {
        const QString option = args.at(index);
        if ((option == QStringLiteral("--timeout-ms") || option == QStringLiteral("--max-gap-ms"))
            && index + 1 < args.size()) {
            bool ok = false;
            const int value = args.at(index + 1).toInt(&ok);
            if (!ok || value <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(option, args.at(index + 1))
                                    << Qt::endl;
                return 4;
            }
            if (option == QStringLiteral("--timeout-ms")) {
                timeoutMs = value;
            } else {
                maxAllowedGapMs = value;
            }
            ++index;
        }
    }

    MainWindow window;
    window.setRecursiveFolderScan(recursive);
    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    bool benchmarkStarted = false;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (!benchmarkStarted) {
            if (!window.isFolderScanPending() && !window.isQueuePopulationPending()) {
                if (window.queuedTiffCount() <= 0) {
                    QCoreApplication::quit();
                    return;
                }
                benchmarkStarted = true;
                window.runQueueBenchmark();
            }
            return;
        }

        if (!window.isQueueBenchmarkActive()) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    window.openFolder(folderPath);
    QCoreApplication::exec();

    const QString log = window.processLogText();
    const bool hasSummary = log.contains(QStringLiteral("Benchmark summary:"));
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("folder=%1").arg(QFileInfo(folderPath).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("recursive=%1").arg(recursive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("queued=%1").arg(window.queuedTiffCount()) << Qt::endl;
    out << QStringLiteral("benchmark_started=%1").arg(benchmarkStarted ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("benchmark_active=%1").arg(window.isQueueBenchmarkActive() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("summary=%1").arg(hasSummary ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for MainWindow benchmark.") << Qt::endl;
        return 5;
    }
    if (window.queuedTiffCount() <= 0) {
        QTextStream(stderr) << QStringLiteral("No TIFF files were queued.") << Qt::endl;
        return 6;
    }
    if (!benchmarkStarted || window.isQueueBenchmarkActive() || !hasSummary) {
        QTextStream(stderr) << QStringLiteral("MainWindow benchmark did not finish with a summary.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

int runProbePythonAverageUi(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    QString path;
    QString outputFolder;
    QString pythonPath;
    int timeoutMs = 30000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--output")) {
            if (!requireValue(option, &outputFolder)) {
                return 2;
            }
        } else if (option == QStringLiteral("--python")) {
            if (!requireValue(option, &pythonPath)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else if (path.isEmpty()) {
            path = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    const QFileInfo tiffInfo(path);
    if (!tiffInfo.isFile()) {
        QTextStream(stderr) << QStringLiteral("TIFF path does not exist: %1").arg(path) << Qt::endl;
        return 3;
    }

    if (outputFolder.isEmpty()) {
        outputFolder = QDir(QCoreApplication::applicationDirPath())
                           .absoluteFilePath(QStringLiteral("python_average_probe"));
    }
    if (!QDir().mkpath(outputFolder)) {
        QTextStream(stderr) << QStringLiteral("Could not create output folder: %1").arg(outputFolder) << Qt::endl;
        return 4;
    }

    if (!pythonPath.isEmpty()) {
        const QFileInfo pythonInfo(pythonPath);
        if (!pythonInfo.isFile()) {
            QTextStream(stderr) << QStringLiteral("Python path does not exist: %1").arg(pythonPath) << Qt::endl;
            return 4;
        }
        pythonPath = pythonInfo.absoluteFilePath();
    }

    MainWindow window;
    window.setPythonBackendPath(pythonPath);
    window.setOutputFolder(outputFolder);

    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    bool started = false;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (!started
            && !window.isFolderScanPending()
            && !window.isQueuePopulationPending()
            && window.queuedTiffCount() > 0) {
            started = true;
            window.runAveragePngForCurrent();
            return;
        }

        const QString log = window.processLogText();
        if (started
            && !window.isAnalysisProcessRunning()
            && (log.contains(QStringLiteral("Process finished:"), Qt::CaseInsensitive)
                || log.contains(QStringLiteral("Process error:"), Qt::CaseInsensitive))) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    QTimer::singleShot(0, &window, [&window, path = tiffInfo.absoluteFilePath()]() {
        window.queuePathForAnalysis(path);
    });
    QCoreApplication::exec();

    const QString log = window.processLogText();
    const bool okResult = log.contains(QStringLiteral("[OK]"), Qt::CaseInsensitive)
        && log.contains(QStringLiteral("Process finished: exit code 0"), Qt::CaseInsensitive);
    const bool responsive = maxGapMs <= maxAllowedGapMs;
    const QStringList logLines = log.split(QLatin1Char('\n'), Qt::SkipEmptyParts);

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(tiffInfo.absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("output=%1").arg(QFileInfo(outputFolder).absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("python=%1").arg(pythonPath.isEmpty() ? QStringLiteral("(default)") : QFileInfo(pythonPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("started=%1").arg(started ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("queued=%1").arg(window.queuedTiffCount()) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("process_ok=%1").arg(okResult ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    for (const QString& line : logLines) {
        if (line.contains(QStringLiteral("Python backend:"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("Python candidate failed"), Qt::CaseInsensitive)) {
            out << QStringLiteral("backend_log=%1").arg(line) << Qt::endl;
        }
    }
    if (!okResult) {
        const int firstLine = std::max(0, static_cast<int>(logLines.size()) - 8);
        for (int lineIndex = firstLine; lineIndex < static_cast<int>(logLines.size()); ++lineIndex) {
            out << QStringLiteral("log_tail=%1").arg(logLines.at(lineIndex)) << Qt::endl;
        }
    }

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for Python average probe.") << Qt::endl;
        return 5;
    }
    if (!started) {
        QTextStream(stderr) << QStringLiteral("Python average probe did not start.") << Qt::endl;
        return 6;
    }
    if (!okResult) {
        QTextStream(stderr) << QStringLiteral("Python average probe did not complete successfully.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

int runProbeValidateBackendUi(const QStringList& args)
{
    QString pythonPath;
    QString motionHook;
    QString roiHook;
    int timeoutMs = 30000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--python")) {
            if (!requireValue(option, &pythonPath)) {
                return 2;
            }
        } else if (option == QStringLiteral("--motion-hook")) {
            if (!requireValue(option, &motionHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--roi-hook")) {
            if (!requireValue(option, &roiHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    if (!pythonPath.isEmpty()) {
        const QFileInfo pythonInfo(pythonPath);
        if (!pythonInfo.isFile()) {
            QTextStream(stderr) << QStringLiteral("Python path does not exist: %1").arg(pythonPath) << Qt::endl;
            return 3;
        }
        pythonPath = pythonInfo.absoluteFilePath();
    }

    MainWindow window;
    window.setPythonBackendPath(pythonPath);
    window.setMotionHookSpec(motionHook);
    window.setRoiHookSpec(roiHook);

    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    bool started = false;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (!started) {
            started = true;
            window.runBackendValidation();
            return;
        }

        const QString log = window.processLogText();
        if (!window.isAnalysisProcessRunning()
            && (log.contains(QStringLiteral("Process finished:"), Qt::CaseInsensitive)
                || log.contains(QStringLiteral("Process error:"), Qt::CaseInsensitive))) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    QCoreApplication::exec();

    const QString log = window.processLogText();
    const QStringList logLines = log.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    const bool processOk = log.contains(QStringLiteral("Process finished: exit code 0"), Qt::CaseInsensitive);
    const bool validationOk =
        log.contains(QStringLiteral("Backend and hook validation succeeded."), Qt::CaseInsensitive);
    const bool importOk =
        log.contains(QStringLiteral("[OK] backend import numpy"), Qt::CaseInsensitive)
        && log.contains(QStringLiteral("[OK] backend import Pillow"), Qt::CaseInsensitive)
        && log.contains(QStringLiteral("[OK] backend import openpyxl"), Qt::CaseInsensitive);
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("python=%1")
               .arg(pythonPath.isEmpty() ? QStringLiteral("(default)") : QFileInfo(pythonPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("motion_hook=%1")
               .arg(motionHook.isEmpty() ? QStringLiteral("(none)") : motionHook)
        << Qt::endl;
    out << QStringLiteral("roi_hook=%1")
               .arg(roiHook.isEmpty() ? QStringLiteral("(none)") : roiHook)
        << Qt::endl;
    out << QStringLiteral("started=%1").arg(started ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("imports_ok=%1").arg(importOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("validation_ok=%1").arg(validationOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("process_ok=%1").arg(processOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    for (const QString& line : logLines) {
        if (line.contains(QStringLiteral("Python backend:"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("Python candidate failed"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("backend import"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("hook"), Qt::CaseInsensitive)) {
            out << QStringLiteral("validation_log=%1").arg(line) << Qt::endl;
        }
    }
    if (!processOk || !validationOk || !importOk) {
        const int firstLine = std::max(0, static_cast<int>(logLines.size()) - 10);
        for (int lineIndex = firstLine; lineIndex < static_cast<int>(logLines.size()); ++lineIndex) {
            out << QStringLiteral("log_tail=%1").arg(logLines.at(lineIndex)) << Qt::endl;
        }
    }

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for backend validation probe.") << Qt::endl;
        return 4;
    }
    if (!started) {
        QTextStream(stderr) << QStringLiteral("Backend validation probe did not start.") << Qt::endl;
        return 5;
    }
    if (!processOk || !validationOk || !importOk) {
        QTextStream(stderr) << QStringLiteral("Backend validation probe did not complete successfully.") << Qt::endl;
        return 6;
    }
    return responsive ? 0 : 7;
}

int runProbeCurrentPipelineUi(const QStringList& args)
{
    if (args.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF path was provided.") << Qt::endl;
        return 2;
    }

    QString path;
    QString outputFolder;
    QString pythonPath;
    QString motionHook;
    QString roiHook;
    QStringList motionHookParamEntries;
    QStringList roiHookParamEntries;
    QString maskBackgroundMode = QStringLiteral("none");
    QString expectedWorkingTiff;
    QString switchToTiff;
    QString expectedViewerTiff;
    bool doAverage = true;
    bool doSignals = true;
    bool doSplit = true;
    int timeoutMs = 60000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--output")) {
            if (!requireValue(option, &outputFolder)) {
                return 2;
            }
        } else if (option == QStringLiteral("--python")) {
            if (!requireValue(option, &pythonPath)) {
                return 2;
            }
        } else if (option == QStringLiteral("--motion-hook")) {
            if (!requireValue(option, &motionHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--motion-hook-param")) {
            QString value;
            if (!requireValue(option, &value)) {
                return 2;
            }
            motionHookParamEntries.append(value);
        } else if (option == QStringLiteral("--roi-hook")) {
            if (!requireValue(option, &roiHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--roi-hook-param")) {
            QString value;
            if (!requireValue(option, &value)) {
                return 2;
            }
            roiHookParamEntries.append(value);
        } else if (option == QStringLiteral("--mask-background")) {
            if (!requireValue(option, &maskBackgroundMode)) {
                return 2;
            }
        } else if (option == QStringLiteral("--expect-working-tiff")) {
            if (!requireValue(option, &expectedWorkingTiff)) {
                return 2;
            }
        } else if (option == QStringLiteral("--switch-to-tiff")) {
            if (!requireValue(option, &switchToTiff)) {
                return 2;
            }
        } else if (option == QStringLiteral("--expect-viewer-tiff")) {
            if (!requireValue(option, &expectedViewerTiff)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--no-average")) {
            doAverage = false;
        } else if (option == QStringLiteral("--no-signals")) {
            doSignals = false;
        } else if (option == QStringLiteral("--no-split")) {
            doSplit = false;
        } else if (path.isEmpty()) {
            path = option;
        } else {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        }
    }

    const QFileInfo tiffInfo(path);
    if (!tiffInfo.isFile()) {
        QTextStream(stderr) << QStringLiteral("TIFF path does not exist: %1").arg(path) << Qt::endl;
        return 3;
    }

    if (outputFolder.isEmpty()) {
        outputFolder = QDir(QCoreApplication::applicationDirPath())
                           .absoluteFilePath(QStringLiteral("current_pipeline_probe"));
    }
    if (!QDir().mkpath(outputFolder)) {
        QTextStream(stderr) << QStringLiteral("Could not create output folder: %1").arg(outputFolder) << Qt::endl;
        return 4;
    }
    const QString absoluteOutputFolder = QFileInfo(outputFolder).absoluteFilePath();
    const QString manifestPath = QDir(absoluteOutputFolder).absoluteFilePath(QStringLiteral("native_pipeline_manifest.json"));
    const QString expectedWorkingPath = expectedWorkingTiff.isEmpty()
        ? QString()
        : QFileInfo(expectedWorkingTiff).absoluteFilePath();
    if (!expectedWorkingPath.isEmpty() && !QFileInfo(expectedWorkingPath).isFile()) {
        QTextStream(stderr) << QStringLiteral("Expected working TIFF does not exist: %1").arg(expectedWorkingTiff)
                            << Qt::endl;
        return 4;
    }
    const QString switchToPath = switchToTiff.isEmpty()
        ? QString()
        : QFileInfo(switchToTiff).absoluteFilePath();
    if (!switchToPath.isEmpty() && !QFileInfo(switchToPath).isFile()) {
        QTextStream(stderr) << QStringLiteral("Switch target TIFF does not exist: %1").arg(switchToTiff)
                            << Qt::endl;
        return 4;
    }
    const QString expectedViewerPath = expectedViewerTiff.isEmpty()
        ? expectedWorkingPath
        : QFileInfo(expectedViewerTiff).absoluteFilePath();
    if (!expectedViewerPath.isEmpty() && !QFileInfo(expectedViewerPath).isFile()) {
        QTextStream(stderr) << QStringLiteral("Expected viewer TIFF does not exist: %1").arg(expectedViewerTiff)
                            << Qt::endl;
        return 4;
    }

    if (!pythonPath.isEmpty()) {
        const QFileInfo pythonInfo(pythonPath);
        if (!pythonInfo.isFile()) {
            QTextStream(stderr) << QStringLiteral("Python path does not exist: %1").arg(pythonPath) << Qt::endl;
            return 4;
        }
        pythonPath = pythonInfo.absoluteFilePath();
    }

    MainWindow window;
    window.setPythonBackendPath(pythonPath);
    window.setOutputFolder(absoluteOutputFolder);
    window.setMotionHookSpec(motionHook);
    window.setMotionHookParams(motionHookParamEntries.join(QStringLiteral("; ")));
    window.setRoiHookSpec(roiHook);
    window.setRoiHookParams(roiHookParamEntries.join(QStringLiteral("; ")));
    window.setMaskBackgroundMode(maskBackgroundMode);
    window.setPipelineStepSelection(doAverage, doSignals, doSplit);

    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    bool started = false;
    bool switchedDuringRun = false;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (!started
            && !window.isFolderScanPending()
            && !window.isQueuePopulationPending()
            && window.queuedTiffCount() > 0) {
            started = true;
            window.runPipelineForCurrent();
            return;
        }

        if (started
            && !switchedDuringRun
            && !switchToPath.isEmpty()
            && window.isAnalysisProcessRunning()) {
            switchedDuringRun = true;
            window.openPath(switchToPath);
            return;
        }

        const QString log = window.processLogText();
        if (started
            && !window.isAnalysisProcessRunning()
            && (log.contains(QStringLiteral("Process finished:"), Qt::CaseInsensitive)
                || log.contains(QStringLiteral("Process error:"), Qt::CaseInsensitive))) {
            if (!expectedViewerPath.isEmpty()
                && log.contains(QStringLiteral("Process finished: exit code 0"), Qt::CaseInsensitive)) {
                const QString viewerPath = QFileInfo(window.currentViewerPath()).absoluteFilePath();
                if (viewerPath.compare(expectedViewerPath, Qt::CaseInsensitive) == 0
                    && window.viewerHasDisplayedFrame()) {
                    QCoreApplication::quit();
                }
                return;
            }
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    const bool probeNeedsInitialViewer = !expectedViewerPath.isEmpty() || !switchToPath.isEmpty();
    QTimer::singleShot(0, &window, [&window, path = tiffInfo.absoluteFilePath(), probeNeedsInitialViewer]() {
        if (probeNeedsInitialViewer) {
            window.openPath(path);
        } else {
            window.queuePathForAnalysis(path);
        }
    });
    QCoreApplication::exec();

    const QString log = window.processLogText();
    const QStringList logLines = log.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    const bool processOk = log.contains(QStringLiteral("Process finished: exit code 0"), Qt::CaseInsensitive);
    const bool manifestMentioned = log.contains(QStringLiteral("[OK] manifest:"), Qt::CaseInsensitive);
    bool manifestExists = QFileInfo::exists(manifestPath);
    bool manifestOk = false;
    if (manifestExists) {
        QFile manifestFile(manifestPath);
        if (manifestFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            const QString manifestText = QString::fromUtf8(manifestFile.readAll());
            manifestOk = manifestText.contains(QStringLiteral("\"ok\": true"), Qt::CaseInsensitive);
        }
    }
    const bool averageOk = !doAverage || log.contains(QStringLiteral("average PNG"), Qt::CaseInsensitive);
    const bool signalsOk = !doSignals || log.contains(QStringLiteral("signals"), Qt::CaseInsensitive);
    const bool splitOk = !doSplit || log.contains(QStringLiteral("split"), Qt::CaseInsensitive);
    const QString viewerPath = QFileInfo(window.currentViewerPath()).absoluteFilePath();
    const bool viewerOk = expectedViewerPath.isEmpty()
        || (viewerPath.compare(expectedViewerPath, Qt::CaseInsensitive) == 0 && window.viewerHasDisplayedFrame());
    const bool workingTiffOk = expectedWorkingPath.isEmpty()
        || QFileInfo::exists(expectedWorkingPath);
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("path=%1").arg(tiffInfo.absoluteFilePath()) << Qt::endl;
    out << QStringLiteral("output=%1").arg(absoluteOutputFolder) << Qt::endl;
    out << QStringLiteral("manifest=%1").arg(manifestPath) << Qt::endl;
    out << QStringLiteral("python=%1")
               .arg(pythonPath.isEmpty() ? QStringLiteral("(default)") : QFileInfo(pythonPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("motion_hook=%1")
               .arg(motionHook.isEmpty() ? QStringLiteral("(none)") : motionHook)
        << Qt::endl;
    out << QStringLiteral("roi_hook=%1")
               .arg(roiHook.isEmpty() ? QStringLiteral("(none)") : roiHook)
        << Qt::endl;
    out << QStringLiteral("expected_working_tiff=%1")
               .arg(expectedWorkingPath.isEmpty() ? QStringLiteral("(none)") : expectedWorkingPath)
        << Qt::endl;
    out << QStringLiteral("switch_to_tiff=%1")
               .arg(switchToPath.isEmpty() ? QStringLiteral("(none)") : switchToPath)
        << Qt::endl;
    out << QStringLiteral("expected_viewer_tiff=%1")
               .arg(expectedViewerPath.isEmpty() ? QStringLiteral("(none)") : expectedViewerPath)
        << Qt::endl;
    out << QStringLiteral("steps=%1,%2,%3")
               .arg(doAverage ? QStringLiteral("average") : QStringLiteral("no-average"),
                    doSignals ? QStringLiteral("signals") : QStringLiteral("no-signals"),
                    doSplit ? QStringLiteral("split") : QStringLiteral("no-split"))
        << Qt::endl;
    out << QStringLiteral("started=%1").arg(started ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("switched_during_run=%1")
               .arg(switchedDuringRun ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("queued=%1").arg(window.queuedTiffCount()) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("process_ok=%1").arg(processOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_mentioned=%1").arg(manifestMentioned ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_exists=%1").arg(manifestExists ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_ok=%1").arg(manifestOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("average_ok=%1").arg(averageOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("signals_ok=%1").arg(signalsOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("split_ok=%1").arg(splitOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("viewer_path=%1").arg(viewerPath) << Qt::endl;
    out << QStringLiteral("viewer_displayed=%1").arg(window.viewerHasDisplayedFrame() ? QStringLiteral("yes") : QStringLiteral("no"))
        << Qt::endl;
    out << QStringLiteral("viewer_ok=%1").arg(viewerOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("working_tiff_ok=%1").arg(workingTiffOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    for (const QString& line : logLines) {
        if (line.contains(QStringLiteral("Python backend:"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("Python candidate failed"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("hook"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("[OK] manifest:"), Qt::CaseInsensitive)) {
            out << QStringLiteral("pipeline_log=%1").arg(line) << Qt::endl;
        }
    }
    if (!processOk || !manifestMentioned || !manifestExists || !manifestOk || !averageOk || !signalsOk || !splitOk
        || !viewerOk || !workingTiffOk) {
        const int firstLine = std::max(0, static_cast<int>(logLines.size()) - 12);
        for (int lineIndex = firstLine; lineIndex < static_cast<int>(logLines.size()); ++lineIndex) {
            out << QStringLiteral("log_tail=%1").arg(logLines.at(lineIndex)) << Qt::endl;
        }
    }

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for current pipeline probe.") << Qt::endl;
        return 5;
    }
    if (!started) {
        QTextStream(stderr) << QStringLiteral("Current pipeline probe did not start.") << Qt::endl;
        return 6;
    }
    if (!processOk || !manifestMentioned || !manifestExists || !manifestOk || !averageOk || !signalsOk || !splitOk
        || !viewerOk || !workingTiffOk) {
        QTextStream(stderr) << QStringLiteral("Current pipeline probe did not complete successfully.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

int runProbeQueuePipelineUi(const QStringList& args)
{
    QStringList paths;
    QString outputFolder;
    QString pythonPath;
    QString motionHook;
    QString roiHook;
    QStringList motionHookParamEntries;
    QStringList roiHookParamEntries;
    QString maskBackgroundMode = QStringLiteral("none");
    bool doAverage = true;
    bool doSignals = true;
    bool doSplit = true;
    int timeoutMs = 60000;
    int maxAllowedGapMs = 250;

    for (int index = 0; index < args.size(); ++index) {
        const QString option = args.at(index);
        auto requireValue = [&](const QString& name, QString* value) -> bool {
            if (index + 1 >= args.size()) {
                QTextStream(stderr) << QStringLiteral("Missing value for %1.").arg(name) << Qt::endl;
                return false;
            }
            *value = args.at(index + 1);
            ++index;
            return true;
        };
        auto requirePositiveInt = [&](const QString& name, int* value) -> bool {
            QString text;
            if (!requireValue(name, &text)) {
                return false;
            }
            bool ok = false;
            const int parsed = text.toInt(&ok);
            if (!ok || parsed <= 0) {
                QTextStream(stderr) << QStringLiteral("Invalid value for %1: %2").arg(name, text) << Qt::endl;
                return false;
            }
            *value = parsed;
            return true;
        };

        if (option == QStringLiteral("--output")) {
            if (!requireValue(option, &outputFolder)) {
                return 2;
            }
        } else if (option == QStringLiteral("--python")) {
            if (!requireValue(option, &pythonPath)) {
                return 2;
            }
        } else if (option == QStringLiteral("--motion-hook")) {
            if (!requireValue(option, &motionHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--motion-hook-param")) {
            QString value;
            if (!requireValue(option, &value)) {
                return 2;
            }
            motionHookParamEntries.append(value);
        } else if (option == QStringLiteral("--roi-hook")) {
            if (!requireValue(option, &roiHook)) {
                return 2;
            }
        } else if (option == QStringLiteral("--roi-hook-param")) {
            QString value;
            if (!requireValue(option, &value)) {
                return 2;
            }
            roiHookParamEntries.append(value);
        } else if (option == QStringLiteral("--mask-background")) {
            if (!requireValue(option, &maskBackgroundMode)) {
                return 2;
            }
        } else if (option == QStringLiteral("--timeout-ms")) {
            if (!requirePositiveInt(option, &timeoutMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--max-gap-ms")) {
            if (!requirePositiveInt(option, &maxAllowedGapMs)) {
                return 2;
            }
        } else if (option == QStringLiteral("--no-average")) {
            doAverage = false;
        } else if (option == QStringLiteral("--no-signals")) {
            doSignals = false;
        } else if (option == QStringLiteral("--no-split")) {
            doSplit = false;
        } else if (option.startsWith(QLatin1Char('-'))) {
            QTextStream(stderr) << QStringLiteral("Unexpected argument: %1").arg(option) << Qt::endl;
            return 2;
        } else {
            paths.append(option);
        }
    }

    if (paths.isEmpty()) {
        QTextStream(stderr) << QStringLiteral("No TIFF paths were provided.") << Qt::endl;
        return 2;
    }

    QStringList absolutePaths;
    for (const QString& path : paths) {
        const QFileInfo info(path);
        if (!info.isFile()) {
            QTextStream(stderr) << QStringLiteral("TIFF path does not exist: %1").arg(path) << Qt::endl;
            return 3;
        }
        absolutePaths.append(info.absoluteFilePath());
    }

    if (outputFolder.isEmpty()) {
        outputFolder = QDir(QCoreApplication::applicationDirPath())
                           .absoluteFilePath(QStringLiteral("queue_pipeline_probe"));
    }
    if (!QDir().mkpath(outputFolder)) {
        QTextStream(stderr) << QStringLiteral("Could not create output folder: %1").arg(outputFolder) << Qt::endl;
        return 4;
    }
    const QString absoluteOutputFolder = QFileInfo(outputFolder).absoluteFilePath();
    const QString manifestPath = QDir(absoluteOutputFolder).absoluteFilePath(QStringLiteral("native_pipeline_manifest.json"));
    if (QFileInfo::exists(manifestPath)) {
        QFile::remove(manifestPath);
    }

    if (!pythonPath.isEmpty()) {
        const QFileInfo pythonInfo(pythonPath);
        if (!pythonInfo.isFile()) {
            QTextStream(stderr) << QStringLiteral("Python path does not exist: %1").arg(pythonPath) << Qt::endl;
            return 4;
        }
        pythonPath = pythonInfo.absoluteFilePath();
    }

    MainWindow window;
    window.setPythonBackendPath(pythonPath);
    window.setOutputFolder(absoluteOutputFolder);
    window.setMotionHookSpec(motionHook);
    window.setMotionHookParams(motionHookParamEntries.join(QStringLiteral("; ")));
    window.setRoiHookSpec(roiHook);
    window.setRoiHookParams(roiHookParamEntries.join(QStringLiteral("; ")));
    window.setMaskBackgroundMode(maskBackgroundMode);
    window.setPipelineStepSelection(doAverage, doSignals, doSplit);

    QElapsedTimer elapsed;
    elapsed.start();
    qint64 lastTickMs = 0;
    qint64 maxGapMs = 0;
    int ticks = 0;
    bool started = false;
    bool timedOut = false;

    QTimer heartbeat;
    heartbeat.setInterval(10);
    QObject::connect(&heartbeat, &QTimer::timeout, [&]() {
        const qint64 now = elapsed.elapsed();
        if (ticks > 0) {
            maxGapMs = std::max(maxGapMs, now - lastTickMs);
        }
        lastTickMs = now;
        ++ticks;
    });

    QTimer poll;
    poll.setInterval(10);
    QObject::connect(&poll, &QTimer::timeout, [&]() {
        if (!started
            && !window.isFolderScanPending()
            && !window.isQueuePopulationPending()
            && window.queuedTiffCount() == absolutePaths.size()) {
            started = true;
            window.runPipelineForQueue();
            return;
        }

        const QString log = window.processLogText();
        if (started
            && !window.isAnalysisProcessRunning()
            && (log.contains(QStringLiteral("Process finished:"), Qt::CaseInsensitive)
                || log.contains(QStringLiteral("Process error:"), Qt::CaseInsensitive))) {
            QCoreApplication::quit();
        }
    });

    QTimer timeout;
    timeout.setSingleShot(true);
    QObject::connect(&timeout, &QTimer::timeout, [&]() {
        timedOut = true;
        QCoreApplication::quit();
    });

    heartbeat.start();
    poll.start();
    timeout.start(timeoutMs);
    QTimer::singleShot(0, &window, [&window, absolutePaths]() {
        window.queuePathsForAnalysis(absolutePaths);
    });
    QCoreApplication::exec();

    const QString log = window.processLogText();
    const QStringList logLines = log.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    const bool processOk = log.contains(QStringLiteral("Process finished: exit code 0"), Qt::CaseInsensitive);
    const bool manifestMentioned = log.contains(QStringLiteral("[OK] manifest:"), Qt::CaseInsensitive);
    bool manifestExists = QFileInfo::exists(manifestPath);
    bool manifestOk = false;
    bool allInputsMentioned = false;
    if (manifestExists) {
        QFile manifestFile(manifestPath);
        if (manifestFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            const QString manifestText = QString::fromUtf8(manifestFile.readAll());
            manifestOk = manifestText.contains(QStringLiteral("\"ok\": true"), Qt::CaseInsensitive);
            allInputsMentioned = true;
            for (const QString& absolutePath : absolutePaths) {
                if (!manifestText.contains(QFileInfo(absolutePath).fileName(), Qt::CaseInsensitive)) {
                    allInputsMentioned = false;
                    break;
                }
            }
        }
    }
    const bool averageOk = !doAverage || log.contains(QStringLiteral("average PNG"), Qt::CaseInsensitive);
    const bool signalsOk = !doSignals || log.contains(QStringLiteral("signals"), Qt::CaseInsensitive);
    const bool splitOk = !doSplit || log.contains(QStringLiteral("split"), Qt::CaseInsensitive);
    const bool responsive = maxGapMs <= maxAllowedGapMs;

    QTextStream out(stdout);
    out << QStringLiteral("paths=%1").arg(absolutePaths.join(QLatin1Char(';'))) << Qt::endl;
    out << QStringLiteral("output=%1").arg(absoluteOutputFolder) << Qt::endl;
    out << QStringLiteral("manifest=%1").arg(manifestPath) << Qt::endl;
    out << QStringLiteral("python=%1")
               .arg(pythonPath.isEmpty() ? QStringLiteral("(default)") : QFileInfo(pythonPath).absoluteFilePath())
        << Qt::endl;
    out << QStringLiteral("motion_hook=%1")
               .arg(motionHook.isEmpty() ? QStringLiteral("(none)") : motionHook)
        << Qt::endl;
    out << QStringLiteral("roi_hook=%1")
               .arg(roiHook.isEmpty() ? QStringLiteral("(none)") : roiHook)
        << Qt::endl;
    out << QStringLiteral("steps=%1,%2,%3")
               .arg(doAverage ? QStringLiteral("average") : QStringLiteral("no-average"),
                    doSignals ? QStringLiteral("signals") : QStringLiteral("no-signals"),
                    doSplit ? QStringLiteral("split") : QStringLiteral("no-split"))
        << Qt::endl;
    out << QStringLiteral("started=%1").arg(started ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("queued=%1").arg(window.queuedTiffCount()) << Qt::endl;
    out << QStringLiteral("elapsed_ms=%1").arg(elapsed.elapsed()) << Qt::endl;
    out << QStringLiteral("timer_ticks=%1").arg(ticks) << Qt::endl;
    out << QStringLiteral("max_event_gap_ms=%1").arg(maxGapMs) << Qt::endl;
    out << QStringLiteral("max_allowed_gap_ms=%1").arg(maxAllowedGapMs) << Qt::endl;
    out << QStringLiteral("process_ok=%1").arg(processOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_mentioned=%1").arg(manifestMentioned ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_exists=%1").arg(manifestExists ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("manifest_ok=%1").arg(manifestOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("all_inputs_mentioned=%1").arg(allInputsMentioned ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("average_ok=%1").arg(averageOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("signals_ok=%1").arg(signalsOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("split_ok=%1").arg(splitOk ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    out << QStringLiteral("responsive=%1").arg(responsive ? QStringLiteral("yes") : QStringLiteral("no")) << Qt::endl;
    for (const QString& line : logLines) {
        if (line.contains(QStringLiteral("Python backend:"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("Python candidate failed"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("hook"), Qt::CaseInsensitive)
            || line.contains(QStringLiteral("[OK] manifest:"), Qt::CaseInsensitive)) {
            out << QStringLiteral("pipeline_log=%1").arg(line) << Qt::endl;
        }
    }
    if (!processOk || !manifestMentioned || !manifestExists || !manifestOk || !allInputsMentioned
        || !averageOk || !signalsOk || !splitOk) {
        const int firstLine = std::max(0, static_cast<int>(logLines.size()) - 12);
        for (int lineIndex = firstLine; lineIndex < static_cast<int>(logLines.size()); ++lineIndex) {
            out << QStringLiteral("log_tail=%1").arg(logLines.at(lineIndex)) << Qt::endl;
        }
    }

    if (timedOut) {
        QTextStream(stderr) << QStringLiteral("Timed out while waiting for queue pipeline probe.") << Qt::endl;
        return 5;
    }
    if (!started) {
        QTextStream(stderr) << QStringLiteral("Queue pipeline probe did not start.") << Qt::endl;
        return 6;
    }
    if (!processOk || !manifestMentioned || !manifestExists || !manifestOk || !allInputsMentioned
        || !averageOk || !signalsOk || !splitOk) {
        QTextStream(stderr) << QStringLiteral("Queue pipeline probe did not complete successfully.") << Qt::endl;
        return 7;
    }
    return responsive ? 0 : 8;
}

} // namespace

int main(int argc, char* argv[])
{
    QStringList args;
    for (int index = 0; index < argc; ++index) {
        args << QString::fromLocal8Bit(argv[index]);
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe")) {
        QCoreApplication app(argc, argv);
        return runProbe(args.at(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-sequence")) {
        QCoreApplication app(argc, argv);
        return runProbeSequence(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-info-cache")) {
        QCoreApplication app(argc, argv);
        return runProbeInfoCache(args.at(2));
    }

    if (args.size() >= 2 && args.at(1) == QStringLiteral("--probe-frame-cache")) {
        QCoreApplication app(argc, argv);
        return runProbeFrameCache(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-metadata")) {
        QCoreApplication app(argc, argv);
        return runProbeMetadata(args.at(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-scrub")) {
        QCoreApplication app(argc, argv);
        return runProbeScrub(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-frame-access")) {
        QCoreApplication app(argc, argv);
        return runProbeFrameAccess(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-render-frame")) {
        QCoreApplication app(argc, argv);
        return runProbeRenderFrame(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-folder-scan")) {
        QCoreApplication app(argc, argv);
        return runProbeFolderScan(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-compat-folder")) {
        QCoreApplication app(argc, argv);
        return runProbeCompatFolder(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-perf-folder")) {
        QCoreApplication app(argc, argv);
        return runProbePerfFolder(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-info-cancel")) {
        QCoreApplication app(argc, argv);
        return runProbeInfoCancel(args.at(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-frame-cancel")) {
        QCoreApplication app(argc, argv);
        return runProbeFrameCancel(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-manifest-working")) {
        QCoreApplication app(argc, argv);
        return runProbeManifestWorking(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-manifest-display")) {
        QCoreApplication app(argc, argv);
        return runProbeManifestDisplay(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-manifest-results")) {
        QCoreApplication app(argc, argv);
        return runProbeManifestResults(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-progress-line")) {
        QCoreApplication app(argc, argv);
        return runProbeProgressLine(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-result-open-mode")) {
        QCoreApplication app(argc, argv);
        return runProbeResultOpenMode(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-result-action-label")) {
        QCoreApplication app(argc, argv);
        return runProbeResultActionLabel(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-image")) {
        QCoreApplication app(argc, argv);
        return runProbeImage(args.at(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-text")) {
        QCoreApplication app(argc, argv);
        return runProbeText(args.at(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-result-log-line")) {
        QCoreApplication app(argc, argv);
        return runProbeResultLogLine(args.mid(2));
    }

    if (args.size() >= 2 && args.at(1) == QStringLiteral("--probe-mainwindow-construct")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeMainWindowConstruct(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-queue-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeQueueUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-viewer-switch-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeViewerSwitchUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-viewer-open-latency")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeViewerOpenLatency(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-viewer-prefetch-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeViewerPrefetchUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-viewer-scrub-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeViewerScrubUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-mainwindow-benchmark")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeMainWindowBenchmark(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-python-average-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbePythonAverageUi(args.mid(2));
    }

    if (args.size() >= 2 && args.at(1) == QStringLiteral("--probe-validate-backend-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeValidateBackendUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-current-pipeline-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeCurrentPipelineUi(args.mid(2));
    }

    if (args.size() >= 3 && args.at(1) == QStringLiteral("--probe-queue-pipeline-ui")) {
        QApplication app(argc, argv);
        QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
        QApplication::setOrganizationName(QStringLiteral("Spike Detector"));
        return runProbeQueuePipelineUi(args.mid(2));
    }

    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("Femtonics Image Processor"));
    QApplication::setOrganizationName(QStringLiteral("Spike Detector"));

    MainWindow window;
    if (args.size() >= 2) {
        window.openPaths(args.mid(1));
    }
    window.show();
    return app.exec();
}
