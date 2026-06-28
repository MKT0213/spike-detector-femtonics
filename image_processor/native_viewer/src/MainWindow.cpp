#include "MainWindow.h"

#include "TiffStack.h"
#include "TiffViewerWidget.h"

#include <QAction>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QDirIterator>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QFile>
#include <QIODevice>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QKeySequence>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMenuBar>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QRunnable>
#include <QSettings>
#include <QSet>
#include <QSignalBlocker>
#include <QSplitter>
#include <QThreadPool>
#include <QVBoxLayout>
#include <QTimer>
#include <QUrl>
#include <QWidget>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr int AnalysisLogMaxLines = 1000;
constexpr int AnalysisLogFlushLines = 120;
constexpr int AnalysisLogFlushIntervalMs = 16;
constexpr int ResultFolderDirectScanLimit = 512;
constexpr int ResultFolderRecursiveScanLimit = 4096;
constexpr qint64 ResultFolderScanBudgetMs = 25;

QString builtInMotionHookSpec()
{
    return QStringLiteral("image_processor.motion_correction:motion_hook");
}

QString builtInMotionHookParams()
{
    return QStringLiteral(
        "method=shared_template_residual; max_shift_px=2; subpixel=false; "
        "residual_max_shift_px=1; residual_min_peak_ratio=1.10");
}

QString builtInRoiHookSpec()
{
    return QStringLiteral("image_processor.segmentation_hooks:roi_hook");
}

QString builtInTrackedRoiHookSpec()
{
    return QStringLiteral("image_processor.tracked_segmentation:roi_hook");
}

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

class FolderScanTask final : public QRunnable {
public:
    FolderScanTask(
        MainWindow* target,
        QString folderPath,
        bool recursive,
        quint64 generation,
        std::shared_ptr<std::atomic_bool> cancelFlag)
        : target_(target)
        , folderPath_(std::move(folderPath))
        , recursive_(recursive)
        , generation_(generation)
        , cancelFlag_(std::move(cancelFlag))
    {
    }

    void run() override
    {
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };
        const QStringList paths = MainWindow::scanFolderPaths(folderPath_, recursive_, shouldCancel);
        const bool cancelled = shouldCancel();
        const QPointer<MainWindow> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target,
             generation = generation_,
             folderPath = folderPath_,
             paths,
             cancelled]() {
                if (target != nullptr) {
                    target->completeFolderScan(generation, folderPath, paths, cancelled);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<MainWindow> target_;
    QString folderPath_;
    bool recursive_;
    quint64 generation_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
};

class QueueBenchmarkTask final : public QRunnable {
public:
    QueueBenchmarkTask(
        MainWindow* target,
        QStringList paths,
        quint64 generation,
        std::shared_ptr<std::atomic_bool> cancelFlag)
        : target_(target)
        , paths_(std::move(paths))
        , generation_(generation)
        , cancelFlag_(std::move(cancelFlag))
    {
    }

    void run() override
    {
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };

        QStringList lines;
        bool ok = true;
        bool cancelled = false;
        QElapsedTimer totalTimer;
        totalTimer.start();
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

        lines << QStringLiteral("Native queue benchmark...");
        lines << QStringLiteral("Queued TIFFs: %1").arg(paths_.size());
        lines << QStringLiteral("Sampling up to 3 frames per TIFF.");

        for (int pathIndex = 0; pathIndex < paths_.size(); ++pathIndex) {
            if (shouldCancel()) {
                cancelled = true;
                break;
            }

            const QString path = paths_.at(pathIndex);
            TiffStackInfo info;
            QString error;
            const bool infoOk = TiffStack::readInfo(path, &info, &error, shouldCancel);
            if (!infoOk) {
                cancelled = shouldCancel() || error.contains(QStringLiteral("cancelled"), Qt::CaseInsensitive);
                if (!cancelled) {
                    lines << QStringLiteral("[ERROR] %1: %2").arg(path, error);
                    ok = false;
                }
                break;
            }

            totalInfoMs += info.elapsedMs;
            totalFrames += info.frameCount;
            maxPixels = std::max(
                maxPixels,
                static_cast<qulonglong>(std::max(0, info.width))
                    * static_cast<qulonglong>(std::max(0, info.height)));
            if (maxInfoPath.isEmpty() || info.elapsedMs > maxInfoMs) {
                maxInfoMs = info.elapsedMs;
                maxInfoPath = path;
            }

            const QList<int> sampledFrames = sampleFrameIndices(info.frameCount, 3);
            qint64 stackReadMs = 0;
            qint64 stackRenderMs = 0;
            qint64 stackMaxReadMs = 0;
            qint64 stackMaxRenderMs = 0;
            for (int frameIndex : sampledFrames) {
                if (shouldCancel()) {
                    cancelled = true;
                    break;
                }

                const TiffFrameResult frame = TiffStack::readFrame(info, frameIndex, shouldCancel);
                if (!frame.ok || !frame.hasSamples()) {
                    cancelled = frame.cancelled || shouldCancel();
                    if (!cancelled) {
                        lines << QStringLiteral("[ERROR] %1 frame %2: %3").arg(path).arg(frameIndex).arg(frame.error);
                        ok = false;
                    }
                    break;
                }

                QString renderError;
                const qint64 renderMs = renderFrameToGrayscaleMs(frame, &renderError);
                if (renderMs < 0) {
                    lines << QStringLiteral("[ERROR] %1 frame %2 render: %3").arg(path).arg(frameIndex).arg(renderError);
                    ok = false;
                    break;
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

            if (cancelled || !ok) {
                break;
            }

            const double stackReadAvg = sampledFrames.isEmpty()
                ? 0.0
                : static_cast<double>(stackReadMs) / static_cast<double>(sampledFrames.size());
            const double stackRenderAvg = sampledFrames.isEmpty()
                ? 0.0
                : static_cast<double>(stackRenderMs) / static_cast<double>(sampledFrames.size());
            lines << QStringLiteral("[%1/%2] %3 | %4 frames, %5x%6, %7, indexed=%8")
                         .arg(pathIndex + 1)
                         .arg(paths_.size())
                         .arg(QFileInfo(path).fileName())
                         .arg(info.frameCount)
                         .arg(info.width)
                         .arg(info.height)
                         .arg(info.pixelType())
                         .arg(info.hasDirectoryOffsets() ? QStringLiteral("yes") : QStringLiteral("no"));
            lines << QStringLiteral("  container %1 | layout %2")
                         .arg(info.bigTiff ? QStringLiteral("BigTIFF") : QStringLiteral("classic TIFF"))
                         .arg(info.tiled ? QStringLiteral("tiled") : QStringLiteral("strips"));
            lines << QStringLiteral("  info %1 ms | read avg %2 ms max %3 ms | render avg %4 ms max %5 ms")
                         .arg(info.elapsedMs)
                         .arg(stackReadAvg, 0, 'f', 2)
                         .arg(stackMaxReadMs)
                         .arg(stackRenderAvg, 0, 'f', 2)
                         .arg(stackMaxRenderMs);
        }

        const double avgInfoMs = paths_.isEmpty()
            ? 0.0
            : static_cast<double>(totalInfoMs) / static_cast<double>(paths_.size());
        const double avgReadMs = measuredFrames <= 0
            ? 0.0
            : static_cast<double>(totalReadMs) / static_cast<double>(measuredFrames);
        const double avgRenderMs = measuredFrames <= 0
            ? 0.0
            : static_cast<double>(totalRenderMs) / static_cast<double>(measuredFrames);

        if (cancelled) {
            lines << QStringLiteral("Benchmark cancelled.");
        } else if (ok) {
            lines << QStringLiteral("Benchmark summary:");
            lines << QStringLiteral("  elapsed %1 ms | measured frames %2 | total stack frames %3 | max pixels %4")
                         .arg(totalTimer.elapsed())
                         .arg(measuredFrames)
                         .arg(totalFrames)
                         .arg(maxPixels);
            lines << QStringLiteral("  avg info %1 ms | avg read %2 ms | avg render %3 ms")
                         .arg(avgInfoMs, 0, 'f', 2)
                         .arg(avgReadMs, 0, 'f', 2)
                         .arg(avgRenderMs, 0, 'f', 2);
            lines << QStringLiteral("  slowest info %1 ms: %2")
                         .arg(maxInfoMs)
                         .arg(QFileInfo(maxInfoPath).fileName());
            lines << QStringLiteral("  slowest read %1 ms frame %2: %3")
                         .arg(maxReadMs)
                         .arg(maxReadFrame)
                         .arg(QFileInfo(maxReadPath).fileName());
            lines << QStringLiteral("  slowest render %1 ms frame %2: %3")
                         .arg(maxRenderMs)
                         .arg(maxRenderFrame)
                         .arg(QFileInfo(maxRenderPath).fileName());
        }

        const QPointer<MainWindow> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target, generation = generation_, lines, cancelled, ok]() {
                if (target != nullptr) {
                    target->completeQueueBenchmark(generation, lines, cancelled, ok);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<MainWindow> target_;
    QStringList paths_;
    quint64 generation_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
};

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    auto* queuePanel = new QWidget(splitter);
    auto* queueLayout = new QVBoxLayout(queuePanel);
    queueLayout->setContentsMargins(8, 8, 8, 8);
    queueLayout->setSpacing(8);

    auto* queueButtons = new QHBoxLayout();
    auto* openTiffButton = new QPushButton(QStringLiteral("Open TIFF"));
    auto* openFolderButton = new QPushButton(QStringLiteral("Open Folder"));
    queueButtons->addWidget(openTiffButton);
    queueButtons->addWidget(openFolderButton);
    queueLayout->addLayout(queueButtons);

    recursiveCheck_ = new QCheckBox(QStringLiteral("Recursive folder scan"));
    recursiveCheck_->setChecked(true);
    queueLayout->addWidget(recursiveCheck_);

    queueLayout->addWidget(new QLabel(QStringLiteral("TIFF Queue")));
    queueList_ = new QListWidget();
    queueList_->setAlternatingRowColors(true);
    queueList_->setSelectionMode(QAbstractItemView::SingleSelection);
    queueLayout->addWidget(queueList_, 1);

    queueStatus_ = new QLabel(QStringLiteral("0 TIFFs"));
    queueLayout->addWidget(queueStatus_);

    auto* queueEditButtons = new QHBoxLayout();
    removeQueueButton_ = new QPushButton(QStringLiteral("Remove"));
    clearQueueButton_ = new QPushButton(QStringLiteral("Clear"));
    removeQueueButton_->setEnabled(false);
    clearQueueButton_->setEnabled(false);
    queueEditButtons->addWidget(removeQueueButton_);
    queueEditButtons->addWidget(clearQueueButton_);
    queueLayout->addLayout(queueEditButtons);

    auto* analysisGroup = new QGroupBox(QStringLiteral("Analysis"));
    auto* analysisLayout = new QVBoxLayout(analysisGroup);

    auto* outputLayout = new QHBoxLayout();
    outputFolderEdit_ = new QLineEdit();
    outputFolderEdit_->setReadOnly(true);
    outputFolderEdit_->setPlaceholderText(QStringLiteral("Default exports beside TIFF"));
    auto* outputButton = new QPushButton(QStringLiteral("Output"));
    openResultsButton_ = new QPushButton(QStringLiteral("Results"));
    openResultsButton_->setEnabled(false);
    openWorkingTiffButton_ = new QPushButton(QStringLiteral("Processed"));
    openWorkingTiffButton_->setEnabled(false);
    openWorkingTiffButton_->setToolTip(
        QStringLiteral("Open latest segmentation overlay or corrected/downstream TIFF"));
    outputLayout->addWidget(outputFolderEdit_, 1);
    outputLayout->addWidget(outputButton);
    outputLayout->addWidget(openResultsButton_);
    outputLayout->addWidget(openWorkingTiffButton_);
    analysisLayout->addLayout(outputLayout);

    useMetadataSamplingCheck_ = new QCheckBox(QStringLiteral("Use metadata sampling rate"));
    useMetadataSamplingCheck_->setChecked(true);
    samplingRateSpin_ = new QDoubleSpinBox();
    samplingRateSpin_->setRange(0.001, 1000000.0);
    samplingRateSpin_->setDecimals(3);
    samplingRateSpin_->setSingleStep(1.0);
    samplingRateSpin_->setValue(1.0);
    samplingRateSpin_->setEnabled(false);
    writeCsvCheck_ = new QCheckBox(QStringLiteral("Write CSV"));
    writeCsvCheck_->setChecked(true);
    pipelineAverageCheck_ = new QCheckBox(QStringLiteral("Pipeline average PNG"));
    pipelineAverageCheck_->setChecked(true);
    pipelineSignalsCheck_ = new QCheckBox(QStringLiteral("Pipeline signals"));
    pipelineSignalsCheck_->setChecked(true);
    maskBackgroundCombo_ = new QComboBox();
    maskBackgroundCombo_->addItem(QStringLiteral("None"), QStringLiteral("none"));
    maskBackgroundCombo_->addItem(QStringLiteral("Parent tile"), QStringLiteral("parent_tile"));
    maskBackgroundCombo_->addItem(QStringLiteral("Global frame"), QStringLiteral("global"));
    maskBackgroundCombo_->setToolTip(QStringLiteral(
        "Background subtraction for mask-based signal export when ROI hooks return segmentation masks"));
    pipelineSplitCheck_ = new QCheckBox(QStringLiteral("Pipeline split ROIs"));
    pipelineSplitCheck_->setChecked(true);
    runMotionCorrectionCheck_ = new QCheckBox(QStringLiteral("Motion correction"));
    runMotionCorrectionCheck_->setChecked(true);
    runMotionCorrectionCheck_->setToolTip(
        QStringLiteral("Run the configured motion hook before downstream analysis"));
    runSegmentationCheck_ = new QCheckBox(QStringLiteral("Segmentation / ROI detection"));
    runSegmentationCheck_->setChecked(true);
    runSegmentationCheck_->setToolTip(
        QStringLiteral("Run the configured ROI hook and use its masks for downstream signal export"));
    runDynamicSegmentationCheck_ = new QCheckBox(QStringLiteral("Dynamic tracked segmentation"));
    runDynamicSegmentationCheck_->setChecked(false);
    runDynamicSegmentationCheck_->setToolTip(
        QStringLiteral("Track segmentation masks frame by frame without motion-correcting the TIFF"));
    autoOpenWorkingTiffCheck_ = new QCheckBox(QStringLiteral("Auto-open processed output"));
    autoOpenWorkingTiffCheck_->setChecked(true);
    autoOpenWorkingTiffCheck_->setToolTip(
        QStringLiteral("After a pipeline run, preview the segmentation overlay or corrected/downstream TIFF"));

    auto* samplingLayout = new QFormLayout();
    samplingLayout->setContentsMargins(0, 0, 0, 0);
    samplingLayout->addRow(useMetadataSamplingCheck_);
    samplingLayout->addRow(QStringLiteral("Manual Hz"), samplingRateSpin_);
    samplingLayout->addRow(writeCsvCheck_);
    samplingLayout->addRow(runMotionCorrectionCheck_);
    samplingLayout->addRow(runSegmentationCheck_);
    samplingLayout->addRow(runDynamicSegmentationCheck_);
    samplingLayout->addRow(pipelineAverageCheck_);
    samplingLayout->addRow(pipelineSignalsCheck_);
    samplingLayout->addRow(QStringLiteral("Mask background"), maskBackgroundCombo_);
    samplingLayout->addRow(pipelineSplitCheck_);
    samplingLayout->addRow(autoOpenWorkingTiffCheck_);
    analysisLayout->addLayout(samplingLayout);

    auto* hookLayout = new QFormLayout();
    hookLayout->setContentsMargins(0, 0, 0, 0);
    pythonBackendEdit_ = new QLineEdit();
    pythonBackendEdit_->setPlaceholderText(QStringLiteral("auto-detect or path to python.exe"));
    choosePythonBackendButton_ = new QPushButton(QStringLiteral("..."));
    choosePythonBackendButton_->setFixedWidth(34);
    choosePythonBackendButton_->setToolTip(QStringLiteral("Select Python backend executable"));
    auto* pythonBackendRow = new QHBoxLayout();
    pythonBackendRow->setContentsMargins(0, 0, 0, 0);
    pythonBackendRow->addWidget(pythonBackendEdit_, 1);
    pythonBackendRow->addWidget(choosePythonBackendButton_);
    motionHookEdit_ = new QLineEdit();
    motionHookEdit_->setPlaceholderText(QStringLiteral("module:function or file.py:function"));
    motionHookParamsEdit_ = new QLineEdit();
    motionHookParamsEdit_->setPlaceholderText(
        QStringLiteral("method=shared_template_residual; max_shift_px=2; residual_min_peak_ratio=1.10"));
    motionHookParamsEdit_->setToolTip(QStringLiteral(
        "Extra motion hook keyword arguments as key=value entries separated by semicolons"));
    builtInMotionHookButton_ = new QPushButton(QStringLiteral("Built-in"));
    builtInMotionHookButton_->setToolTip(
        QStringLiteral("Use image_processor.motion_correction:motion_hook"));
    chooseMotionHookButton_ = new QPushButton(QStringLiteral("..."));
    chooseMotionHookButton_->setFixedWidth(34);
    chooseMotionHookButton_->setToolTip(QStringLiteral("Select Python hook file"));
    auto* motionHookRow = new QHBoxLayout();
    motionHookRow->setContentsMargins(0, 0, 0, 0);
    motionHookRow->addWidget(motionHookEdit_, 1);
    motionHookRow->addWidget(builtInMotionHookButton_);
    motionHookRow->addWidget(chooseMotionHookButton_);
    roiHookEdit_ = new QLineEdit();
    roiHookEdit_->setPlaceholderText(QStringLiteral("module:function or file.py:function"));
    roiHookParamsEdit_ = new QLineEdit();
    roiHookParamsEdit_->setPlaceholderText(QStringLiteral("min_area=8; percentile=95; std_factor=2.5"));
    roiHookParamsEdit_->setToolTip(QStringLiteral(
        "Extra ROI hook keyword arguments as key=value entries separated by semicolons"));
    builtInRoiHookButton_ = new QPushButton(QStringLiteral("Built-in"));
    builtInRoiHookButton_->setToolTip(
        QStringLiteral("Use image_processor.segmentation_hooks:roi_hook"));
    chooseRoiHookButton_ = new QPushButton(QStringLiteral("..."));
    chooseRoiHookButton_->setFixedWidth(34);
    chooseRoiHookButton_->setToolTip(QStringLiteral("Select Python hook file"));
    auto* roiHookRow = new QHBoxLayout();
    roiHookRow->setContentsMargins(0, 0, 0, 0);
    roiHookRow->addWidget(roiHookEdit_, 1);
    roiHookRow->addWidget(builtInRoiHookButton_);
    roiHookRow->addWidget(chooseRoiHookButton_);
    hookLayout->addRow(QStringLiteral("Python backend"), pythonBackendRow);
    hookLayout->addRow(QStringLiteral("Motion hook"), motionHookRow);
    hookLayout->addRow(QStringLiteral("Motion params"), motionHookParamsEdit_);
    hookLayout->addRow(QStringLiteral("ROI hook"), roiHookRow);
    hookLayout->addRow(QStringLiteral("ROI params"), roiHookParamsEdit_);
    analysisLayout->addLayout(hookLayout);

    validateHooksButton_ = new QPushButton(QStringLiteral("Validate Backend / Hooks"));
    validateHooksButton_->setToolTip(
        QStringLiteral("Check the selected Python backend, required packages, and configured motion/ROI hooks"));
    analysisLayout->addWidget(validateHooksButton_);

    splitButton_ = new QPushButton(QStringLiteral("Split TIFF"));
    splitButton_->setToolTip(QStringLiteral("Split the current TIFF into per-ROI TIFF stacks"));
    splitButton_->setVisible(false);
    exportSignalsButton_ = new QPushButton(QStringLiteral("Export Signals"));
    exportSignalsButton_->setToolTip(QStringLiteral("Export ROI mean-intensity signal workbook for the current TIFF"));
    exportSignalsButton_->setVisible(false);
    averagePngButton_ = new QPushButton(QStringLiteral("Average PNG"));
    averagePngButton_->setToolTip(QStringLiteral("Export an average-projection PNG for the current TIFF"));
    averagePngButton_->setVisible(false);
    runPipelineButton_ = new QPushButton(QStringLiteral("Export Current"));
    runPipelineButton_->setToolTip(
        QStringLiteral("Export signals, QC images, segmentation masks, and split ROI TIFFs for the current TIFF"));
    runQueuePipelineButton_ = new QPushButton(QStringLiteral("Export Queue"));
    runQueuePipelineButton_->setToolTip(QStringLiteral("Export consolidated pipeline outputs for every queued TIFF"));
    benchmarkQueueButton_ = new QPushButton(QStringLiteral("Benchmark Queue"));
    benchmarkQueueButton_->setToolTip(QStringLiteral("Measure native TIFF read/render timing for queued TIFFs"));
    cancelAnalysisButton_ = new QPushButton(QStringLiteral("Cancel"));
    cancelAnalysisButton_->setEnabled(false);
    auto* pipelineButtons = new QHBoxLayout();
    pipelineButtons->addWidget(runPipelineButton_);
    pipelineButtons->addWidget(runQueuePipelineButton_);
    pipelineButtons->addWidget(cancelAnalysisButton_);
    analysisLayout->addLayout(pipelineButtons);
    analysisLayout->addWidget(benchmarkQueueButton_);

    analysisProgressBar_ = new QProgressBar();
    analysisProgressBar_->setRange(0, 1);
    analysisProgressBar_->setValue(0);
    analysisProgressBar_->setFormat(QStringLiteral("Idle"));
    analysisLayout->addWidget(analysisProgressBar_);
    backendStatusLabel_ = new QLabel(QStringLiteral("Python backend: idle"));
    backendStatusLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    backendStatusLabel_->setWordWrap(true);
    analysisLayout->addWidget(backendStatusLabel_);

    analysisLayout->addWidget(new QLabel(QStringLiteral("Last Results")));
    resultsList_ = new QListWidget();
    resultsList_->setAlternatingRowColors(true);
    resultsList_->setSelectionMode(QAbstractItemView::SingleSelection);
    resultsList_->setMaximumHeight(110);
    analysisLayout->addWidget(resultsList_);
    openResultButton_ = new QPushButton(QStringLiteral("Open Selected"));
    openResultButton_->setEnabled(false);
    analysisLayout->addWidget(openResultButton_);
    queueLayout->addWidget(analysisGroup);

    queueLayout->addWidget(new QLabel(QStringLiteral("Process Log")));
    analysisLog_ = new QPlainTextEdit();
    analysisLog_->setReadOnly(true);
    analysisLog_->setMaximumBlockCount(AnalysisLogMaxLines);
    analysisLog_->setMaximumHeight(180);
    queueLayout->addWidget(analysisLog_);
    analysisLogFlushTimer_ = new QTimer(this);
    analysisLogFlushTimer_->setInterval(AnalysisLogFlushIntervalMs);
    connect(analysisLogFlushTimer_, &QTimer::timeout, this, [this]() {
        flushAnalysisLogBuffer(AnalysisLogFlushLines);
    });

    viewer_ = new TiffViewerWidget(this);
    analysisProcess_ = new QProcess(this);
    splitter->addWidget(queuePanel);
    splitter->addWidget(viewer_);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setSizes(QList<int>{260, 940});
    setCentralWidget(splitter);

    resize(1320, 860);
    setWindowTitle(QStringLiteral("Femtonics Image Processor"));

    auto* fileMenu = menuBar()->addMenu(QStringLiteral("&File"));
    auto* openAction = fileMenu->addAction(QStringLiteral("&Open TIFF..."));
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &MainWindow::openFileDialog);

    auto* openFolderAction = fileMenu->addAction(QStringLiteral("Open &Folder..."));
    connect(openFolderAction, &QAction::triggered, this, &MainWindow::openFolderDialog);

    auto* quitAction = fileMenu->addAction(QStringLiteral("E&xit"));
    quitAction->setShortcut(QKeySequence::Quit);
    connect(quitAction, &QAction::triggered, this, &QWidget::close);

    connect(openTiffButton, &QPushButton::clicked, this, &MainWindow::openFileDialog);
    connect(openFolderButton, &QPushButton::clicked, this, &MainWindow::openFolderDialog);
    connect(outputButton, &QPushButton::clicked, this, &MainWindow::chooseOutputFolder);
    connect(openResultsButton_, &QPushButton::clicked, this, &MainWindow::openResultsFolder);
    connect(openWorkingTiffButton_, &QPushButton::clicked, this, &MainWindow::openWorkingTiff);
    connect(openResultButton_, &QPushButton::clicked, this, &MainWindow::openSelectedResult);
    connect(resultsList_, &QListWidget::currentRowChanged, this, [this](int row) {
        Q_UNUSED(row);
        updateResultActionUi();
    });
    connect(resultsList_, &QListWidget::itemActivated, this, [this](QListWidgetItem*) {
        if (analysisProcess_ == nullptr || analysisProcess_->state() == QProcess::NotRunning) {
            openSelectedResult();
        }
    });
    connect(choosePythonBackendButton_, &QPushButton::clicked, this, &MainWindow::choosePythonBackendFile);
    connect(builtInMotionHookButton_, &QPushButton::clicked, this, &MainWindow::setBuiltInMotionHookDefaults);
    connect(chooseMotionHookButton_, &QPushButton::clicked, this, &MainWindow::chooseMotionHookFile);
    connect(builtInRoiHookButton_, &QPushButton::clicked, this, &MainWindow::setBuiltInRoiHookDefaults);
    connect(chooseRoiHookButton_, &QPushButton::clicked, this, &MainWindow::chooseRoiHookFile);
    connect(validateHooksButton_, &QPushButton::clicked, this, &MainWindow::validateHooks);
    connect(useMetadataSamplingCheck_, &QCheckBox::toggled, samplingRateSpin_, &QDoubleSpinBox::setDisabled);
    connect(runMotionCorrectionCheck_, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked && runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked()) {
            QSignalBlocker blocker(runDynamicSegmentationCheck_);
            runDynamicSegmentationCheck_->setChecked(false);
        }
        if (checked && motionHookEdit_ != nullptr && motionHookEdit_->text().trimmed().isEmpty()) {
            setBuiltInMotionHookDefaults();
        }
        updateProcessingStageUi();
    });
    connect(runSegmentationCheck_, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked && runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked()) {
            QSignalBlocker blocker(runDynamicSegmentationCheck_);
            runDynamicSegmentationCheck_->setChecked(false);
        }
        if (checked && roiHookEdit_ != nullptr && roiHookEdit_->text().trimmed().isEmpty()) {
            setBuiltInRoiHookDefaults();
        }
        updateProcessingStageUi();
    });
    connect(runDynamicSegmentationCheck_, &QCheckBox::toggled, this, [this](bool checked) {
        if (checked) {
            if (runMotionCorrectionCheck_ != nullptr) {
                QSignalBlocker blocker(runMotionCorrectionCheck_);
                runMotionCorrectionCheck_->setChecked(false);
            }
            if (runSegmentationCheck_ != nullptr) {
                QSignalBlocker blocker(runSegmentationCheck_);
                runSegmentationCheck_->setChecked(false);
            }
        }
        updateProcessingStageUi();
    });
    connect(pipelineSignalsCheck_, &QCheckBox::toggled, this, [this](bool checked) {
        Q_UNUSED(checked);
        updateProcessingStageUi();
    });
    connect(splitButton_, &QPushButton::clicked, this, &MainWindow::splitCurrentTiff);
    connect(exportSignalsButton_, &QPushButton::clicked, this, &MainWindow::exportCurrentSignals);
    connect(averagePngButton_, &QPushButton::clicked, this, &MainWindow::exportAveragePng);
    connect(runPipelineButton_, &QPushButton::clicked, this, &MainWindow::runCurrentPipeline);
    connect(runQueuePipelineButton_, &QPushButton::clicked, this, &MainWindow::runQueuePipeline);
    connect(benchmarkQueueButton_, &QPushButton::clicked, this, &MainWindow::runQueueBenchmark);
    connect(cancelAnalysisButton_, &QPushButton::clicked, this, &MainWindow::cancelAnalysis);
    connect(removeQueueButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedQueueItem);
    connect(clearQueueButton_, &QPushButton::clicked, this, &MainWindow::clearQueue);
    connect(queueList_, &QListWidget::currentRowChanged, this, &MainWindow::openQueueRow);
    connect(analysisProcess_, &QProcess::readyReadStandardOutput, this, [this]() {
        appendAnalysisProcessOutput(
            &pendingAnalysisStdoutText_,
            QString::fromLocal8Bit(analysisProcess_->readAllStandardOutput()));
    });
    connect(analysisProcess_, &QProcess::readyReadStandardError, this, [this]() {
        appendAnalysisProcessOutput(
            &pendingAnalysisStderrText_,
            QString::fromLocal8Bit(analysisProcess_->readAllStandardError()));
    });
    connect(
        analysisProcess_,
        QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
        this,
        [this](int exitCode, QProcess::ExitStatus exitStatus) {
            appendAnalysisProcessOutput(
                &pendingAnalysisStdoutText_,
                QString::fromLocal8Bit(analysisProcess_->readAllStandardOutput()));
            appendAnalysisProcessOutput(
                &pendingAnalysisStderrText_,
                QString::fromLocal8Bit(analysisProcess_->readAllStandardError()));
            flushAnalysisProcessOutput(&pendingAnalysisStdoutText_);
            flushAnalysisProcessOutput(&pendingAnalysisStderrText_);
            const QString status = exitStatus == QProcess::NormalExit
                ? QStringLiteral("exit code %1").arg(exitCode)
                : QStringLiteral("crashed");
            appendAnalysisLog(QStringLiteral("Process finished: %1").arg(status));
            if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                setAnalysisProgressIdle(QStringLiteral("Done"));
                setBackendStatus(QStringLiteral("Python backend: done"));
            } else {
                setAnalysisProgressIdle(QStringLiteral("Failed"));
                setBackendStatus(QStringLiteral("Python backend: failed"));
            }
            detectWorkingTiffFromManifest();
            detectPreferredDisplayPathFromManifest();
            updateResultListFromManifest();
            setAnalysisBusy(false);
            if (exitStatus == QProcess::NormalExit && exitCode == 0) {
                openWorkingTiffIfConfigured();
            }
    });
    connect(analysisProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        Q_UNUSED(error);
        appendAnalysisProcessOutput(
            &pendingAnalysisStdoutText_,
            QString::fromLocal8Bit(analysisProcess_->readAllStandardOutput()));
        appendAnalysisProcessOutput(
            &pendingAnalysisStderrText_,
            QString::fromLocal8Bit(analysisProcess_->readAllStandardError()));
        flushAnalysisProcessOutput(&pendingAnalysisStdoutText_);
        flushAnalysisProcessOutput(&pendingAnalysisStderrText_);
        appendAnalysisLog(QStringLiteral("Process error: %1").arg(analysisProcess_->errorString()));
        setAnalysisProgressIdle(QStringLiteral("Process error"));
        setBackendStatus(QStringLiteral("Python backend: process error"), analysisProcess_->errorString());
        setAnalysisBusy(false);
    });
    connect(viewer_, &TiffViewerWidget::fileLoaded, this, [this](const QString& path) {
        if (pendingQueuePaths_.isEmpty()) {
            addPaths(QStringList{path}, false);
        }
        selectQueuePath(path);
        setWindowTitle(QStringLiteral("%1 - Femtonics Image Processor").arg(QFileInfo(path).fileName()));
    });

    loadUserSettings();
    setAnalysisBusy(false);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    cancelFolderScan();
    cancelQueuedPathAdd();
    cancelQueueBenchmark();
    saveUserSettings();
    QMainWindow::closeEvent(event);
}

void MainWindow::openPath(const QString& path)
{
    const QFileInfo info(path);
    if (info.isDir()) {
        openFolder(path);
        return;
    }
    addPaths(QStringList{path}, true);
}

void MainWindow::openPaths(const QStringList& paths)
{
    if (paths.size() <= 1) {
        if (!paths.isEmpty()) {
            openPath(paths.first());
        }
        return;
    }
    startQueuedPathAdd(paths, true);
}

void MainWindow::queuePathForAnalysis(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }
    queuePathsForAnalysis(QStringList{path});
}

void MainWindow::queuePathsForAnalysis(const QStringList& paths)
{
    if (paths.isEmpty()) {
        setQueueStatus();
        return;
    }

    addPaths(paths, false);
    for (const QString& path : paths) {
        const QFileInfo info(path);
        if (info.isFile()) {
            selectQueuePath(info.absoluteFilePath());
            return;
        }
    }
    setQueueStatus();
}

void MainWindow::openFolder(const QString& folderPath)
{
    startFolderScan(folderPath);
}

int MainWindow::queuedTiffCount() const
{
    return queueList_ == nullptr ? 0 : queueList_->count();
}

bool MainWindow::isFolderScanPending() const
{
    return folderScanCancelFlag_ != nullptr;
}

bool MainWindow::isQueuePopulationPending() const
{
    return !pendingQueuePaths_.isEmpty();
}

QString MainWindow::queueStatusText() const
{
    return queueStatus_ == nullptr ? QString() : queueStatus_->text();
}

void MainWindow::setRecursiveFolderScan(bool recursive)
{
    if (recursiveCheck_ != nullptr) {
        recursiveCheck_->setChecked(recursive);
    }
}

bool MainWindow::isQueueBenchmarkActive() const
{
    return queueBenchmarkActive_;
}

bool MainWindow::isAnalysisProcessRunning() const
{
    return analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning;
}

QString MainWindow::processLogText() const
{
    return analysisLogMirrorLines_.join(QLatin1Char('\n'));
}

QString MainWindow::currentViewerPath() const
{
    return viewer_ == nullptr ? QString() : viewer_->currentFilePath();
}

bool MainWindow::viewerHasDisplayedFrame() const
{
    return viewer_ != nullptr && viewer_->hasDisplayedFrame();
}

void MainWindow::setOutputFolder(const QString& folderPath)
{
    if (outputFolderEdit_ != nullptr) {
        outputFolderEdit_->setText(folderPath);
    }
}

void MainWindow::setPythonBackendPath(const QString& pythonPath)
{
    if (pythonBackendEdit_ != nullptr) {
        pythonBackendEdit_->setText(pythonPath);
    }
}

void MainWindow::setMotionHookSpec(const QString& hookSpec)
{
    if (motionHookEdit_ != nullptr) {
        motionHookEdit_->setText(hookSpec);
    }
}

void MainWindow::setRoiHookSpec(const QString& hookSpec)
{
    if (roiHookEdit_ != nullptr) {
        roiHookEdit_->setText(hookSpec);
    }
}

void MainWindow::setMotionHookParams(const QString& params)
{
    if (motionHookParamsEdit_ != nullptr) {
        motionHookParamsEdit_->setText(params);
    }
}

void MainWindow::setRoiHookParams(const QString& params)
{
    if (roiHookParamsEdit_ != nullptr) {
        roiHookParamsEdit_->setText(params);
    }
}

void MainWindow::setBuiltInMotionHookDefaults()
{
    if (motionHookEdit_ != nullptr) {
        motionHookEdit_->setText(builtInMotionHookSpec());
    }
    if (motionHookParamsEdit_ != nullptr) {
        motionHookParamsEdit_->setText(builtInMotionHookParams());
    }
}

void MainWindow::setBuiltInRoiHookDefaults()
{
    if (roiHookEdit_ != nullptr) {
        roiHookEdit_->setText(builtInRoiHookSpec());
    }
}

void MainWindow::updateProcessingStageUi()
{
    const bool busy = analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning;
    const bool dynamicEnabled =
        runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked();
    const bool motionEnabled =
        !dynamicEnabled && (runMotionCorrectionCheck_ == nullptr || runMotionCorrectionCheck_->isChecked());
    const bool segmentationEnabled =
        !dynamicEnabled && (runSegmentationCheck_ == nullptr || runSegmentationCheck_->isChecked());
    const bool roiParamsEnabled = dynamicEnabled || segmentationEnabled;

    if (runMotionCorrectionCheck_ != nullptr) {
        runMotionCorrectionCheck_->setEnabled(!busy && !dynamicEnabled);
    }
    if (runSegmentationCheck_ != nullptr) {
        runSegmentationCheck_->setEnabled(!busy && !dynamicEnabled);
    }
    if (runDynamicSegmentationCheck_ != nullptr) {
        runDynamicSegmentationCheck_->setEnabled(!busy);
    }

    if (motionHookEdit_ != nullptr) {
        motionHookEdit_->setEnabled(!busy && motionEnabled);
    }
    if (motionHookParamsEdit_ != nullptr) {
        motionHookParamsEdit_->setEnabled(!busy && motionEnabled);
    }
    if (builtInMotionHookButton_ != nullptr) {
        builtInMotionHookButton_->setEnabled(!busy && motionEnabled);
    }
    if (chooseMotionHookButton_ != nullptr) {
        chooseMotionHookButton_->setEnabled(!busy && motionEnabled);
    }

    if (roiHookEdit_ != nullptr) {
        roiHookEdit_->setEnabled(!busy && segmentationEnabled);
    }
    if (roiHookParamsEdit_ != nullptr) {
        roiHookParamsEdit_->setEnabled(!busy && roiParamsEnabled);
    }
    if (builtInRoiHookButton_ != nullptr) {
        builtInRoiHookButton_->setEnabled(!busy && segmentationEnabled);
    }
    if (chooseRoiHookButton_ != nullptr) {
        chooseRoiHookButton_->setEnabled(!busy && segmentationEnabled);
    }
    if (maskBackgroundCombo_ != nullptr) {
        const bool signalsEnabled = pipelineSignalsCheck_ == nullptr || pipelineSignalsCheck_->isChecked();
        maskBackgroundCombo_->setEnabled(!busy && segmentationEnabled && signalsEnabled);
    }
}

void MainWindow::setMaskBackgroundMode(const QString& mode)
{
    if (maskBackgroundCombo_ == nullptr) {
        return;
    }
    const int index = maskBackgroundCombo_->findData(mode);
    maskBackgroundCombo_->setCurrentIndex(index >= 0 ? index : 0);
}

void MainWindow::setPipelineStepSelection(bool averagePng, bool exportSignals, bool splitRois)
{
    if (pipelineAverageCheck_ != nullptr) {
        pipelineAverageCheck_->setChecked(averagePng);
    }
    if (pipelineSignalsCheck_ != nullptr) {
        pipelineSignalsCheck_->setChecked(exportSignals);
    }
    if (pipelineSplitCheck_ != nullptr) {
        pipelineSplitCheck_->setChecked(splitRois);
    }
}

void MainWindow::runAveragePngForCurrent()
{
    exportAveragePng();
}

void MainWindow::runPipelineForCurrent()
{
    runCurrentPipeline();
}

void MainWindow::runPipelineForQueue()
{
    runQueuePipeline();
}

void MainWindow::runBackendValidation()
{
    validateHooks();
}

void MainWindow::startFolderScan(const QString& folderPath)
{
    if (folderPath.isEmpty()) {
        return;
    }

    cancelFolderScan();
    cancelQueuedPathAdd();
    const QFileInfo info(folderPath);
    const QString absolutePath = info.absoluteFilePath();
    folderScanCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    ++folderScanGeneration_;
    queueStatus_->setText(QStringLiteral("Scanning folder..."));
    QThreadPool::globalInstance()->start(new FolderScanTask(
        this,
        absolutePath,
        recursiveCheck_ != nullptr && recursiveCheck_->isChecked(),
        folderScanGeneration_,
        folderScanCancelFlag_));
}

void MainWindow::cancelFolderScan()
{
    if (folderScanCancelFlag_ != nullptr) {
        folderScanCancelFlag_->store(true);
        folderScanCancelFlag_.reset();
    }
}

void MainWindow::completeFolderScan(
    quint64 generation,
    const QString& folderPath,
    const QStringList& paths,
    bool cancelled)
{
    if (generation != folderScanGeneration_) {
        return;
    }
    folderScanCancelFlag_.reset();
    if (cancelled) {
        setQueueStatus();
        return;
    }
    if (paths.isEmpty()) {
        QMessageBox::information(
            this,
            QStringLiteral("Open Folder"),
            QStringLiteral("No TIFF files were found in:\n%1").arg(folderPath));
        setQueueStatus();
        return;
    }
    startQueuedPathAdd(paths, true);
}

void MainWindow::openFileDialog()
{
    const QStringList paths = QFileDialog::getOpenFileNames(
        this,
        QStringLiteral("Open TIFF stack"),
        QString(),
        QStringLiteral("TIFF files (*.tif *.tiff);;All files (*.*)"));
    if (!paths.isEmpty()) {
        openPaths(paths);
    }
}

void MainWindow::openFolderDialog()
{
    const QString folderPath = QFileDialog::getExistingDirectory(this, QStringLiteral("Open TIFF folder"));
    if (!folderPath.isEmpty()) {
        openFolder(folderPath);
    }
}

void MainWindow::chooseOutputFolder()
{
    const QString folderPath = QFileDialog::getExistingDirectory(this, QStringLiteral("Select output folder"));
    if (!folderPath.isEmpty()) {
        const QString absolutePath = QFileInfo(folderPath).absoluteFilePath();
        outputFolderEdit_->setText(absolutePath);
        setLastResultsPath(absolutePath);
    }
}

void MainWindow::openResultsFolder()
{
    if (lastResultsPath_.isEmpty()) {
        return;
    }

    QFileInfo info(lastResultsPath_);
    if (!info.exists()) {
        QMessageBox::information(
            this,
            QStringLiteral("Results"),
            QStringLiteral("Results folder does not exist yet:\n%1").arg(lastResultsPath_));
        return;
    }

    const QString folderPath = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(folderPath))) {
        QMessageBox::warning(
            this,
            QStringLiteral("Results"),
            QStringLiteral("Could not open results folder:\n%1").arg(folderPath));
    }
}

void MainWindow::openWorkingTiff()
{
    const QString displayPath = lastDisplayResultPath_.isEmpty() ? lastWorkingTiffPath_ : lastDisplayResultPath_;
    if (displayPath.isEmpty()) {
        return;
    }

    const QFileInfo info(displayPath);
    if (!info.isFile()) {
        QMessageBox::information(
            this,
            QStringLiteral("Processed Output"),
            QStringLiteral("Processed output does not exist yet:\n%1").arg(displayPath));
        lastDisplayResultPath_.clear();
        setLastWorkingTiffPath(QString());
        return;
    }

    openPath(info.absoluteFilePath());
}

void MainWindow::openWorkingTiffIfConfigured()
{
    if (autoOpenWorkingTiffCheck_ == nullptr || !autoOpenWorkingTiffCheck_->isChecked()) {
        return;
    }
    const QString displayPath = lastDisplayResultPath_.isEmpty() ? lastWorkingTiffPath_ : lastDisplayResultPath_;
    if (displayPath.isEmpty()) {
        return;
    }

    const QFileInfo info(displayPath);
    if (!info.isFile()) {
        return;
    }

    QString currentPath;
    const QString currentViewerItem = viewer_ == nullptr ? QString() : viewer_->currentFilePath();
    const QString currentQueueItem = currentQueuePath();
    const QString currentItem = currentViewerItem.isEmpty() ? currentQueueItem : currentViewerItem;
    if (!currentItem.isEmpty()) {
        const QFileInfo currentInfo(currentItem);
        currentPath = currentInfo.canonicalFilePath().isEmpty()
            ? currentInfo.absoluteFilePath()
            : currentInfo.canonicalFilePath();
    }
    const QString resolvedDisplayPath = info.canonicalFilePath().isEmpty()
        ? info.absoluteFilePath()
        : info.canonicalFilePath();
    if (!currentPath.isEmpty() && currentPath.compare(resolvedDisplayPath, Qt::CaseInsensitive) == 0) {
        return;
    }

    if (!analysisAutoOpenSourcePath_.isEmpty()) {
        const QFileInfo sourceInfo(analysisAutoOpenSourcePath_);
        const QString sourcePath = sourceInfo.canonicalFilePath().isEmpty()
            ? sourceInfo.absoluteFilePath()
            : sourceInfo.canonicalFilePath();
        if (!currentPath.isEmpty() && currentPath.compare(sourcePath, Qt::CaseInsensitive) != 0) {
            appendAnalysisLog(
                QStringLiteral("[OK] processed output ready, not auto-opened because the viewer moved to another file: %1")
                    .arg(resolvedDisplayPath));
            return;
        }
    }

    appendAnalysisLog(QStringLiteral("[OK] opening processed output: %1").arg(resolvedDisplayPath));
    const ResultOpenMode mode = resultOpenModeForPath(resolvedDisplayPath);
    if (mode == ResultOpenMode::Image) {
        viewer_->loadImageFile(resolvedDisplayPath);
        setWindowTitle(QStringLiteral("%1 - Femtonics Image Processor").arg(info.fileName()));
        return;
    }
    if (mode == ResultOpenMode::Text) {
        viewer_->loadTextFile(resolvedDisplayPath);
        setWindowTitle(QStringLiteral("%1 - Femtonics Image Processor").arg(info.fileName()));
        return;
    }
    openPath(resolvedDisplayPath);
}

void MainWindow::openSelectedResult()
{
    if (resultsList_ == nullptr) {
        return;
    }

    const QListWidgetItem* item = resultsList_->currentItem();
    if (item == nullptr) {
        return;
    }

    const QString path = item->data(Qt::UserRole).toString();
    const QFileInfo info(path);
    const ResultOpenMode mode = resultOpenModeForPath(path);
    if (mode == ResultOpenMode::Missing) {
        QMessageBox::information(
            this,
            QStringLiteral("Result"),
            QStringLiteral("Result does not exist yet:\n%1").arg(path));
        updateResultListFromManifest();
        return;
    }

    if (mode == ResultOpenMode::Viewer) {
        openPath(info.absoluteFilePath());
        return;
    }
    if (mode == ResultOpenMode::Image) {
        viewer_->loadImageFile(info.absoluteFilePath());
        setWindowTitle(QStringLiteral("%1 - Femtonics Image Processor").arg(info.fileName()));
        return;
    }
    if (mode == ResultOpenMode::Text) {
        viewer_->loadTextFile(info.absoluteFilePath());
        setWindowTitle(QStringLiteral("%1 - Femtonics Image Processor").arg(info.fileName()));
        return;
    }
    if (mode == ResultOpenMode::FolderTiffs) {
        openFolder(info.absoluteFilePath());
        return;
    }

    if (!QDesktopServices::openUrl(QUrl::fromLocalFile(info.absoluteFilePath()))) {
        QMessageBox::warning(
            this,
            QStringLiteral("Result"),
            QStringLiteral("Could not open result:\n%1").arg(info.absoluteFilePath()));
    }
}

void MainWindow::choosePythonBackendFile()
{
    if (pythonBackendEdit_ == nullptr) {
        return;
    }

    const QString pythonPath = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Select Python backend"),
        QString(),
        QStringLiteral("Python executable (python.exe);;Executables (*.exe);;All files (*.*)"));
    if (!pythonPath.isEmpty()) {
        pythonBackendEdit_->setText(QFileInfo(pythonPath).absoluteFilePath());
    }
}

void MainWindow::chooseMotionHookFile()
{
    chooseHookFileFor(motionHookEdit_, QStringLiteral("Select motion hook file"));
}

void MainWindow::chooseRoiHookFile()
{
    chooseHookFileFor(roiHookEdit_, QStringLiteral("Select ROI hook file"));
}

void MainWindow::chooseHookFileFor(QLineEdit* edit, const QString& title)
{
    if (edit == nullptr) {
        return;
    }

    const QString hookPath = QFileDialog::getOpenFileName(
        this,
        title,
        QString(),
        QStringLiteral("Python files (*.py);;All files (*.*)"));
    if (hookPath.isEmpty()) {
        return;
    }

    QString functionName = QStringLiteral("run");
    const QString currentText = edit->text().trimmed();
    const int suffixSeparator = currentText.lastIndexOf(QLatin1Char(':'));
    if (suffixSeparator >= 0 && suffixSeparator + 1 < currentText.size()) {
        const QString suffix = currentText.mid(suffixSeparator + 1).trimmed();
        if (!suffix.contains(QLatin1Char('/')) && !suffix.contains(QLatin1Char('\\'))) {
            functionName = suffix;
        }
    }

    edit->setText(QStringLiteral("%1:%2").arg(QFileInfo(hookPath).absoluteFilePath(), functionName));
}

void MainWindow::validateHooks()
{
    QStringList args;
    appendSelectedHookArgs(args);

    runPythonModule(
        QStringLiteral("image_processor.validate_hooks"),
        args,
        QStringLiteral("Validate Backend / Hooks"));
}

void MainWindow::addPaths(const QStringList& paths, bool openFirst)
{
    cancelQueuedPathAdd();
    if (paths.isEmpty()) {
        setQueueStatus();
        return;
    }

    QSet<QString> existingPaths;
    for (int row = 0; row < queueList_->count(); ++row) {
        existingPaths.insert(queueList_->item(row)->data(Qt::UserRole).toString());
    }

    int firstAddedRow = -1;
    QString firstUsablePath;
    {
        const QSignalBlocker blocker(queueList_);
        for (const QString& path : paths) {
            const QFileInfo info(path);
            if (!info.isFile()) {
                continue;
            }

            const QString suffix = info.suffix().toLower();
            if (suffix != QStringLiteral("tif") && suffix != QStringLiteral("tiff")) {
                continue;
            }

            const QString absolutePath = info.absoluteFilePath();
            if (existingPaths.contains(absolutePath)) {
                if (firstUsablePath.isEmpty()) {
                    firstUsablePath = absolutePath;
                }
                continue;
            }

            auto* item = new QListWidgetItem(info.fileName());
            item->setToolTip(absolutePath);
            item->setData(Qt::UserRole, absolutePath);
            queueList_->addItem(item);
            existingPaths.insert(absolutePath);
            if (firstAddedRow < 0) {
                firstAddedRow = queueList_->count() - 1;
                firstUsablePath = absolutePath;
            }
        }
    }

    setQueueStatus();

    if (!openFirst || queueList_->count() == 0) {
        return;
    }

    int rowToOpen = firstAddedRow;
    if (rowToOpen < 0 && !firstUsablePath.isEmpty()) {
        for (int row = 0; row < queueList_->count(); ++row) {
            if (queueList_->item(row)->data(Qt::UserRole).toString() == firstUsablePath) {
                rowToOpen = row;
                break;
            }
        }
    }
    if (rowToOpen < 0) {
        rowToOpen = 0;
    }

    {
        const QSignalBlocker blocker(queueList_);
        queueList_->setCurrentRow(rowToOpen);
    }
    openQueueRow(rowToOpen);
}

void MainWindow::startQueuedPathAdd(const QStringList& paths, bool openFirst)
{
    cancelQueuedPathAdd();
    if (paths.isEmpty()) {
        setQueueStatus();
        return;
    }

    pendingQueuePaths_ = paths;
    pendingQueueExistingPaths_.clear();
    pendingQueueOpenFirst_ = openFirst;
    pendingQueueOpenedFirst_ = false;
    pendingQueueIndex_ = 0;
    pendingQueueFirstAddedRow_ = -1;
    pendingQueueFirstUsablePath_.clear();

    for (int row = 0; row < queueList_->count(); ++row) {
        pendingQueueExistingPaths_.insert(queueList_->item(row)->data(Qt::UserRole).toString());
    }

    ++queueAddGeneration_;
    const quint64 generation = queueAddGeneration_;
    setQueueStatus();
    QTimer::singleShot(0, this, [this, generation]() {
        addNextQueuedPathBatch(generation);
    });
}

void MainWindow::cancelQueuedPathAdd()
{
    ++queueAddGeneration_;
    pendingQueuePaths_.clear();
    pendingQueueExistingPaths_.clear();
    pendingQueueOpenFirst_ = false;
    pendingQueueOpenedFirst_ = false;
    pendingQueueIndex_ = 0;
    pendingQueueFirstAddedRow_ = -1;
    pendingQueueFirstUsablePath_.clear();
    if (queueStatus_ != nullptr) {
        setQueueStatus();
    }
}

void MainWindow::addNextQueuedPathBatch(quint64 generation)
{
    if (generation != queueAddGeneration_) {
        return;
    }

    const int total = pendingQueuePaths_.size();
    if (pendingQueueIndex_ >= total) {
        const bool openFirst = pendingQueueOpenFirst_;
        const bool openedFirst = pendingQueueOpenedFirst_;
        int rowToOpen = pendingQueueFirstAddedRow_;
        const QString firstUsablePath = pendingQueueFirstUsablePath_;

        pendingQueuePaths_.clear();
        pendingQueueExistingPaths_.clear();
        pendingQueueOpenFirst_ = false;
        pendingQueueOpenedFirst_ = false;
        pendingQueueIndex_ = 0;
        pendingQueueFirstAddedRow_ = -1;
        pendingQueueFirstUsablePath_.clear();
        setQueueStatus();

        if (!openFirst || openedFirst || queueList_->count() == 0) {
            return;
        }

        if (rowToOpen < 0 && !firstUsablePath.isEmpty()) {
            for (int row = 0; row < queueList_->count(); ++row) {
                if (queueList_->item(row)->data(Qt::UserRole).toString() == firstUsablePath) {
                    rowToOpen = row;
                    break;
                }
            }
        }
        if (rowToOpen < 0) {
            rowToOpen = 0;
        }

        {
            const QSignalBlocker blocker(queueList_);
            queueList_->setCurrentRow(rowToOpen);
        }
        openQueueRow(rowToOpen);
        return;
    }

    constexpr int batchSize = 200;
    constexpr qint64 batchBudgetMs = 8;
    QElapsedTimer batchTimer;
    batchTimer.start();
    int processed = 0;
    {
        const QSignalBlocker blocker(queueList_);
        queueList_->setUpdatesEnabled(false);
        for (; pendingQueueIndex_ < total; ++pendingQueueIndex_) {
            if (processed >= batchSize || (processed > 0 && batchTimer.elapsed() >= batchBudgetMs)) {
                break;
            }
            ++processed;
            const QFileInfo info(pendingQueuePaths_.at(pendingQueueIndex_));
            if (!info.isFile()) {
                continue;
            }

            const QString suffix = info.suffix().toLower();
            if (suffix != QStringLiteral("tif") && suffix != QStringLiteral("tiff")) {
                continue;
            }

            const QString absolutePath = info.absoluteFilePath();
            if (pendingQueueExistingPaths_.contains(absolutePath)) {
                if (pendingQueueFirstUsablePath_.isEmpty()) {
                    pendingQueueFirstUsablePath_ = absolutePath;
                }
                continue;
            }

            auto* item = new QListWidgetItem(info.fileName());
            item->setToolTip(absolutePath);
            item->setData(Qt::UserRole, absolutePath);
            queueList_->addItem(item);
            pendingQueueExistingPaths_.insert(absolutePath);
            if (pendingQueueFirstAddedRow_ < 0) {
                pendingQueueFirstAddedRow_ = queueList_->count() - 1;
                pendingQueueFirstUsablePath_ = absolutePath;
            }
        }
        queueList_->setUpdatesEnabled(true);
    }

    openPendingQueueFirstUsable();
    setQueueStatus();
    QTimer::singleShot(0, this, [this, generation]() {
        addNextQueuedPathBatch(generation);
    });
}

void MainWindow::openPendingQueueFirstUsable()
{
    if (!pendingQueueOpenFirst_ || pendingQueueOpenedFirst_ || queueList_->count() == 0) {
        return;
    }

    int rowToOpen = pendingQueueFirstAddedRow_;
    if (rowToOpen < 0 && !pendingQueueFirstUsablePath_.isEmpty()) {
        for (int row = 0; row < queueList_->count(); ++row) {
            if (queueList_->item(row)->data(Qt::UserRole).toString() == pendingQueueFirstUsablePath_) {
                rowToOpen = row;
                break;
            }
        }
    }
    if (rowToOpen < 0) {
        return;
    }

    {
        const QSignalBlocker blocker(queueList_);
        queueList_->setCurrentRow(rowToOpen);
    }
    pendingQueueOpenedFirst_ = true;
    openQueueRow(rowToOpen);
}

void MainWindow::removeSelectedQueueItem()
{
    if (analysisProcess_->state() != QProcess::NotRunning) {
        return;
    }
    cancelQueuedPathAdd();

    const int row = queueList_->currentRow();
    if (row < 0 || row >= queueList_->count()) {
        setQueueStatus();
        return;
    }

    delete queueList_->takeItem(row);
    if (queueList_->count() <= 0) {
        viewer_->clearView();
        setWindowTitle(QStringLiteral("Femtonics Image Processor"));
        setQueueStatus();
        return;
    }

    const int nextRow = std::min(row, queueList_->count() - 1);
    queueList_->setCurrentRow(nextRow);
    openQueueRow(nextRow);
}

void MainWindow::clearQueue()
{
    if (analysisProcess_->state() != QProcess::NotRunning) {
        return;
    }

    cancelQueuedPathAdd();
    queueList_->clear();
    viewer_->clearView();
    setWindowTitle(QStringLiteral("Femtonics Image Processor"));
    setQueueStatus();
}

void MainWindow::splitCurrentTiff()
{
    const QString path = currentQueuePath();
    if (path.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Split TIFF"), QStringLiteral("Select a TIFF first."));
        return;
    }
    QStringList args;
    args << path << QStringLiteral("--debug");
    args << outputArgsForCurrentTiff(true);
    runPythonModule(
        QStringLiteral("image_processor.split_multi_roi_tiff"),
        args,
        QStringLiteral("Split TIFF"),
        splitResultsDirForCurrentTiff());
}

void MainWindow::exportAveragePng()
{
    const QString path = currentQueuePath();
    if (path.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Average PNG"), QStringLiteral("Select a TIFF first."));
        return;
    }
    QStringList args;
    args << path;
    const QString outputRoot = outputFolderEdit_->text().trimmed();
    if (!outputRoot.isEmpty()) {
        args << QStringLiteral("--output") << outputRoot;
    }
    runPythonModule(
        QStringLiteral("image_processor.export_average_png"),
        args,
        QStringLiteral("Average PNG"),
        averageResultsDirForCurrentTiff());
}

void MainWindow::exportCurrentSignals()
{
    const QString path = currentQueuePath();
    if (path.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Export Signals"), QStringLiteral("Select a TIFF first."));
        return;
    }
    QStringList args;
    args << path;
    args << outputArgsForCurrentTiff(false);
    args << samplingArgs();
    if (!writeCsvCheck_->isChecked()) {
        args << QStringLiteral("--no-csv");
    }
    runPythonModule(
        QStringLiteral("image_processor.export_signals"),
        args,
        QStringLiteral("Export Signals"),
        signalResultsDirForCurrentTiff());
}

void MainWindow::runCurrentPipeline()
{
    const QString path = currentQueuePath();
    if (path.isEmpty()) {
        QMessageBox::information(this, QStringLiteral("Run Pipeline"), QStringLiteral("Select a TIFF first."));
        return;
    }

    QStringList args;
    args << path;
    const QString outputRoot = outputFolderEdit_->text().trimmed();
    if (!outputRoot.isEmpty()) {
        args << QStringLiteral("--output") << outputRoot;
    }
    args << samplingArgs();
    if (!writeCsvCheck_->isChecked()) {
        args << QStringLiteral("--no-csv");
    }
    appendPipelineStepArgs(args);
    appendMaskBackgroundArgs(args);
    appendSelectedHookArgs(args);

    runPythonModule(
        QStringLiteral("image_processor.native_pipeline"),
        args,
        QStringLiteral("Run Pipeline"),
        currentPipelineResultsDir(),
        path);
}

void MainWindow::runQueuePipeline()
{
    if (queueList_->count() <= 0) {
        QMessageBox::information(
            this,
            QStringLiteral("Run Queue Pipeline"),
            QStringLiteral("Add TIFFs to the queue first."));
        return;
    }

    QStringList args;
    for (int row = 0; row < queueList_->count(); ++row) {
        const QString path = queueList_->item(row)->data(Qt::UserRole).toString();
        if (!path.isEmpty()) {
            args << path;
        }
    }
    if (args.isEmpty()) {
        QMessageBox::information(
            this,
            QStringLiteral("Run Queue Pipeline"),
            QStringLiteral("No usable TIFF paths in the queue."));
        return;
    }

    const QString outputRoot = outputFolderEdit_->text().trimmed();
    if (!outputRoot.isEmpty()) {
        args << QStringLiteral("--output") << outputRoot;
    }
    args << samplingArgs();
    if (!writeCsvCheck_->isChecked()) {
        args << QStringLiteral("--no-csv");
    }
    appendPipelineStepArgs(args);
    appendMaskBackgroundArgs(args);
    appendSelectedHookArgs(args);

    runPythonModule(
        QStringLiteral("image_processor.native_pipeline"),
        args,
        QStringLiteral("Run Queue Pipeline"),
        queuePipelineResultsDir(),
        currentQueuePath());
}

void MainWindow::runQueueBenchmark()
{
    if (analysisProcess_->state() != QProcess::NotRunning) {
        QMessageBox::information(
            this,
            QStringLiteral("Benchmark Queue"),
            QStringLiteral("Wait for the Python analysis process to finish first."));
        return;
    }
    if (queueBenchmarkActive_) {
        QMessageBox::information(
            this,
            QStringLiteral("Benchmark Queue"),
            QStringLiteral("A queue benchmark is already running."));
        return;
    }
    if (isFolderScanPending() || isQueuePopulationPending()) {
        QMessageBox::information(
            this,
            QStringLiteral("Benchmark Queue"),
            QStringLiteral("Wait for folder scanning and queue population to finish first."));
        return;
    }
    if (queueList_->count() <= 0) {
        QMessageBox::information(this, QStringLiteral("Benchmark Queue"), QStringLiteral("Add TIFFs to the queue first."));
        return;
    }

    QStringList paths;
    for (int row = 0; row < queueList_->count(); ++row) {
        const QString path = queueList_->item(row)->data(Qt::UserRole).toString();
        if (!path.isEmpty()) {
            paths << path;
        }
    }
    if (paths.isEmpty()) {
        QMessageBox::information(
            this,
            QStringLiteral("Benchmark Queue"),
            QStringLiteral("No usable TIFF paths in the queue."));
        return;
    }

    if (analysisLogFlushTimer_ != nullptr) {
        analysisLogFlushTimer_->stop();
    }
    pendingAnalysisLogDisplayLines_.clear();
    analysisLogMirrorLines_.clear();
    pendingAnalysisStdoutText_.clear();
    pendingAnalysisStderrText_.clear();
    analysisLog_->clear();
    setAnalysisProgressBusy(QStringLiteral("Benchmark Queue"));
    setBackendStatus(QStringLiteral("Native benchmark: running"), QStringLiteral("Measuring queued TIFF read/render timing"));
    appendAnalysisLog(QStringLiteral("Benchmark Queue..."));
    appendAnalysisLog(QStringLiteral("Queued TIFFs: %1").arg(paths.size()));
    appendAnalysisLog(QStringLiteral("Sampling up to 3 frames per TIFF."));

    queueBenchmarkCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    queueBenchmarkActive_ = true;
    ++queueBenchmarkGeneration_;
    setAnalysisBusy(true);
    QThreadPool::globalInstance()->start(new QueueBenchmarkTask(
        this,
        paths,
        queueBenchmarkGeneration_,
        queueBenchmarkCancelFlag_));
}

void MainWindow::cancelQueueBenchmark()
{
    if (queueBenchmarkCancelFlag_ != nullptr) {
        queueBenchmarkCancelFlag_->store(true);
    }
}

void MainWindow::completeQueueBenchmark(
    quint64 generation,
    const QStringList& lines,
    bool cancelled,
    bool ok)
{
    if (generation != queueBenchmarkGeneration_) {
        return;
    }

    queueBenchmarkCancelFlag_.reset();
    queueBenchmarkActive_ = false;
    for (const QString& line : lines) {
        appendAnalysisLog(line);
    }

    if (cancelled) {
        setAnalysisProgressIdle(QStringLiteral("Cancelled"));
        setBackendStatus(QStringLiteral("Native benchmark: cancelled"));
    } else if (ok) {
        setAnalysisProgressIdle(QStringLiteral("Done"));
        setBackendStatus(QStringLiteral("Native benchmark: done"));
    } else {
        setAnalysisProgressIdle(QStringLiteral("Failed"));
        setBackendStatus(QStringLiteral("Native benchmark: failed"));
    }
    setAnalysisBusy(false);
}

void MainWindow::cancelAnalysis()
{
    if (queueBenchmarkActive_) {
        appendAnalysisLog(QStringLiteral("Cancelling queue benchmark..."));
        cancelAnalysisButton_->setEnabled(false);
        cancelQueueBenchmark();
        return;
    }

    if (analysisProcess_->state() == QProcess::NotRunning) {
        return;
    }

    appendAnalysisLog(QStringLiteral("Cancelling analysis process..."));
    cancelAnalysisButton_->setEnabled(false);
    analysisProcess_->terminate();
    QTimer::singleShot(2500, this, [this]() {
        if (analysisProcess_->state() != QProcess::NotRunning) {
            appendAnalysisLog(QStringLiteral("Analysis process did not exit; killing it."));
            analysisProcess_->kill();
        }
    });
}

QStringList MainWindow::outputArgsForCurrentTiff(bool splitOutput) const
{
    const QString path = currentQueuePath();
    const QString outputPath = splitOutput
        ? QString()
        : currentTiffOutputDir(path);

    if (splitOutput) {
        const QString root = outputFolderEdit_->text().trimmed();
        if (root.isEmpty()) {
            return {};
        }
        return QStringList{
            QStringLiteral("--output"),
            QDir(root).absoluteFilePath(QStringLiteral("split_rois")),
        };
    }

    if (outputPath.isEmpty()) {
        return {};
    }
    return QStringList{QStringLiteral("--output"), outputPath};
}

void MainWindow::appendPipelineStepArgs(QStringList& args) const
{
    if (!pipelineAverageCheck_->isChecked()) {
        args << QStringLiteral("--no-average");
    }
    if (!pipelineSignalsCheck_->isChecked()) {
        args << QStringLiteral("--no-signals");
    }
    if (!pipelineSplitCheck_->isChecked()) {
        args << QStringLiteral("--no-split");
    }
}

void MainWindow::appendMaskBackgroundArgs(QStringList& args) const
{
    if (maskBackgroundCombo_ == nullptr) {
        return;
    }
    if (runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked()) {
        return;
    }
    if (runSegmentationCheck_ != nullptr && !runSegmentationCheck_->isChecked()) {
        return;
    }
    const QString mode = maskBackgroundCombo_->currentData().toString();
    if (mode.isEmpty() || mode == QStringLiteral("none")) {
        return;
    }
    args << QStringLiteral("--mask-background") << mode;
}

void MainWindow::appendHookParameterArgs(QStringList& args, const QLineEdit* edit, const QString& option) const
{
    if (edit == nullptr) {
        return;
    }
    appendHookParameterText(args, edit->text(), option);
}

void MainWindow::appendHookParameterText(QStringList& args, const QString& text, const QString& option) const
{
    const QString trimmedText = text.trimmed();
    if (trimmedText.isEmpty()) {
        return;
    }
    const QStringList entries =
        trimmedText.split(QRegularExpression(QStringLiteral("[;\\r\\n]+")), Qt::SkipEmptyParts);
    for (const QString& rawEntry : entries) {
        const QString entry = rawEntry.trimmed();
        if (entry.isEmpty()) {
            continue;
        }
        args << option << entry;
    }
}

void MainWindow::appendSelectedHookArgs(QStringList& args) const
{
    const bool runDynamic =
        runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked();
    if (runDynamic) {
        args << QStringLiteral("--roi-hook") << builtInTrackedRoiHookSpec();
        appendHookParameterArgs(args, roiHookParamsEdit_, QStringLiteral("--roi-hook-param"));
        return;
    }

    const bool runMotion = runMotionCorrectionCheck_ == nullptr || runMotionCorrectionCheck_->isChecked();
    if (runMotion) {
        const QString motionHook = motionHookEdit_ == nullptr ? QString() : motionHookEdit_->text().trimmed();
        const QString hookSpec = motionHook.isEmpty() ? builtInMotionHookSpec() : motionHook;
        args << QStringLiteral("--motion-hook") << hookSpec;

        const QString paramsText = motionHookParamsEdit_ == nullptr ? QString() : motionHookParamsEdit_->text().trimmed();
        appendHookParameterText(
            args,
            paramsText.isEmpty() && hookSpec == builtInMotionHookSpec() ? builtInMotionHookParams() : paramsText,
            QStringLiteral("--motion-hook-param"));
    }

    const bool runSegmentation = runSegmentationCheck_ == nullptr || runSegmentationCheck_->isChecked();
    if (runSegmentation) {
        const QString roiHook = roiHookEdit_ == nullptr ? QString() : roiHookEdit_->text().trimmed();
        args << QStringLiteral("--roi-hook") << (roiHook.isEmpty() ? builtInRoiHookSpec() : roiHook);
        appendHookParameterArgs(args, roiHookParamsEdit_, QStringLiteral("--roi-hook-param"));
    }
}

QStringList MainWindow::samplingArgs() const
{
    if (useMetadataSamplingCheck_->isChecked()) {
        return {};
    }
    return QStringList{
        QStringLiteral("--sampling-rate"),
        QString::number(samplingRateSpin_->value(), 'g', 12),
    };
}

QString MainWindow::currentTiffOutputDir(const QString& path) const
{
    const QString root = outputFolderEdit_->text().trimmed();
    if (root.isEmpty() || path.isEmpty()) {
        return {};
    }
    return QDir(root).absoluteFilePath(QFileInfo(path).completeBaseName());
}

QString MainWindow::defaultExportsDirForTiff(const QString& path) const
{
    if (path.isEmpty()) {
        return {};
    }
    return QFileInfo(path).absoluteDir().absoluteFilePath(QStringLiteral("exports"));
}

QString MainWindow::defaultSplitDirForTiff(const QString& path) const
{
    if (path.isEmpty()) {
        return {};
    }
    return QFileInfo(path).absoluteDir().absoluteFilePath(
        QStringLiteral("split_rois/%1").arg(QFileInfo(path).completeBaseName()));
}

QString MainWindow::averageResultsDirForCurrentTiff() const
{
    const QString path = currentQueuePath();
    const QString root = outputFolderEdit_->text().trimmed();
    if (!root.isEmpty() && !path.isEmpty()) {
        return QDir(root).absoluteFilePath(QFileInfo(path).completeBaseName());
    }
    return defaultExportsDirForTiff(path);
}

QString MainWindow::signalResultsDirForCurrentTiff() const
{
    const QString path = currentQueuePath();
    const QString root = outputFolderEdit_->text().trimmed();
    if (!root.isEmpty() && !path.isEmpty()) {
        return QDir(root).absoluteFilePath(QFileInfo(path).completeBaseName());
    }
    return defaultExportsDirForTiff(path);
}

QString MainWindow::splitResultsDirForCurrentTiff() const
{
    const QString path = currentQueuePath();
    const QString root = outputFolderEdit_->text().trimmed();
    if (!root.isEmpty() && !path.isEmpty()) {
        return QDir(root).absoluteFilePath(
            QStringLiteral("split_rois/%1").arg(QFileInfo(path).completeBaseName()));
    }
    return defaultSplitDirForTiff(path);
}

QString MainWindow::currentPipelineResultsDir() const
{
    const QString root = outputFolderEdit_->text().trimmed();
    if (!root.isEmpty()) {
        return QFileInfo(root).absoluteFilePath();
    }
    return defaultExportsDirForTiff(currentQueuePath());
}

QString MainWindow::queuePipelineResultsDir() const
{
    const QString root = outputFolderEdit_->text().trimmed();
    if (!root.isEmpty()) {
        return QFileInfo(root).absoluteFilePath();
    }
    if (queueList_->count() == 1) {
        return defaultExportsDirForTiff(queueList_->item(0)->data(Qt::UserRole).toString());
    }
    return QDir(repoRootPath()).absoluteFilePath(QStringLiteral("exports"));
}

void MainWindow::runPythonModule(
    const QString& moduleName,
    const QStringList& moduleArgs,
    const QString& label,
    const QString& resultsPath,
    const QString& autoOpenSourcePath)
{
    if (analysisProcess_->state() != QProcess::NotRunning) {
        QMessageBox::information(this, label, QStringLiteral("A Python analysis process is already running."));
        return;
    }
    if (queueBenchmarkActive_) {
        QMessageBox::information(this, label, QStringLiteral("A native queue benchmark is already running."));
        return;
    }

    QStringList pythonDiagnostics;
    QStringList pythonPathEntries;
    const QString python = pythonExecutable(&pythonDiagnostics, &pythonPathEntries);
    QStringList args;
    args << QStringLiteral("-u")
         << QStringLiteral("-c")
         << QStringLiteral(
                "import runpy, sys; "
                "module = sys.argv[1]; "
                "sys.argv = sys.argv[1:]; "
                "runpy.run_module(module, run_name='__main__')")
         << moduleName;
    args << moduleArgs;

    if (analysisLogFlushTimer_ != nullptr) {
        analysisLogFlushTimer_->stop();
    }
    pendingAnalysisLogDisplayLines_.clear();
    analysisLogMirrorLines_.clear();
    pendingAnalysisStdoutText_.clear();
    pendingAnalysisStderrText_.clear();
    analysisLog_->clear();
    lastManifestPath_.clear();
    analysisAutoOpenSourcePath_ = autoOpenSourcePath.isEmpty()
        ? QString()
        : QFileInfo(autoOpenSourcePath).absoluteFilePath();
    lastDisplayResultPath_.clear();
    setLastWorkingTiffPath(QString());
    setResultPaths(QStringList());
    setLastResultsPath(resultsPath);
    setAnalysisProgressBusy(label);
    setBackendStatus(QStringLiteral("Python backend: running %1").arg(moduleName), python);
    appendAnalysisLog(QStringLiteral("%1...").arg(label));
    for (const QString& diagnostic : pythonDiagnostics) {
        appendAnalysisLog(diagnostic);
    }
    appendAnalysisLog(
        QStringLiteral("%1 -u -c <module runner> %2 %3")
            .arg(python, moduleName, moduleArgs.join(QLatin1Char(' '))));
    setAnalysisBusy(true);

    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));
    if (!pythonPathEntries.isEmpty()) {
        const QString existingPythonPath = environment.value(QStringLiteral("PYTHONPATH"));
        QStringList combinedPythonPath = pythonPathEntries;
        if (!existingPythonPath.isEmpty()) {
            combinedPythonPath.append(existingPythonPath.split(QDir::listSeparator(), Qt::SkipEmptyParts));
        }
        environment.insert(QStringLiteral("PYTHONPATH"), combinedPythonPath.join(QDir::listSeparator()));
    }
    analysisProcess_->setWorkingDirectory(repoRootPath());
    analysisProcess_->setProgram(python);
    analysisProcess_->setArguments(args);
    analysisProcess_->setProcessEnvironment(environment);
    analysisProcess_->start();
}

void MainWindow::appendAnalysisLog(const QString& message)
{
    QString normalized = message;
    normalized.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    normalized.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    const QStringList lines = normalized.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    for (const QString& line : lines) {
        const QString trimmed = line.trimmed();
        if (!trimmed.isEmpty()) {
            analysisLogMirrorLines_.append(trimmed);
            pendingAnalysisLogDisplayLines_.append(trimmed);
            updateAnalysisProgressFromLine(trimmed);
            detectAnalysisResultPath(trimmed);
            detectAnalysisOutputPath(trimmed);
        }
    }

    const int excessLines = analysisLogMirrorLines_.size() - AnalysisLogMaxLines;
    if (excessLines > 0) {
        analysisLogMirrorLines_.erase(
            analysisLogMirrorLines_.begin(),
            analysisLogMirrorLines_.begin() + excessLines);
    }

    if (!pendingAnalysisLogDisplayLines_.isEmpty()
        && analysisLogFlushTimer_ != nullptr
        && !analysisLogFlushTimer_->isActive()) {
        analysisLogFlushTimer_->start();
    }
}

void MainWindow::appendAnalysisProcessOutput(QString* pendingText, const QString& message)
{
    if (pendingText == nullptr || message.isEmpty()) {
        return;
    }

    QString normalized = message;
    normalized.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    normalized.replace(QLatin1Char('\r'), QLatin1Char('\n'));
    pendingText->append(normalized);

    int newlineIndex = pendingText->indexOf(QLatin1Char('\n'));
    while (newlineIndex >= 0) {
        const QString line = pendingText->left(newlineIndex);
        pendingText->remove(0, newlineIndex + 1);
        appendAnalysisLog(line);
        newlineIndex = pendingText->indexOf(QLatin1Char('\n'));
    }
}

void MainWindow::flushAnalysisProcessOutput(QString* pendingText)
{
    if (pendingText == nullptr || pendingText->isEmpty()) {
        return;
    }

    const QString line = *pendingText;
    pendingText->clear();
    appendAnalysisLog(line);
}

void MainWindow::flushAnalysisLogBuffer(int maxLines)
{
    if (pendingAnalysisLogDisplayLines_.isEmpty()) {
        if (analysisLogFlushTimer_ != nullptr) {
            analysisLogFlushTimer_->stop();
        }
        return;
    }

    const int lineCount = maxLines <= 0
        ? pendingAnalysisLogDisplayLines_.size()
        : std::min(maxLines, static_cast<int>(pendingAnalysisLogDisplayLines_.size()));
    const QStringList displayLines = pendingAnalysisLogDisplayLines_.mid(0, lineCount);
    pendingAnalysisLogDisplayLines_.erase(
        pendingAnalysisLogDisplayLines_.begin(),
        pendingAnalysisLogDisplayLines_.begin() + lineCount);

    if (analysisLog_ != nullptr && !displayLines.isEmpty()) {
        analysisLog_->appendPlainText(displayLines.join(QLatin1Char('\n')));
    }

    if (pendingAnalysisLogDisplayLines_.isEmpty()) {
        if (analysisLogFlushTimer_ != nullptr) {
            analysisLogFlushTimer_->stop();
        }
    } else if (analysisLogFlushTimer_ != nullptr && !analysisLogFlushTimer_->isActive()) {
        analysisLogFlushTimer_->start();
    }
}

bool MainWindow::parseAnalysisProgressLine(
    const QString& line,
    QString* stageLabel,
    int* done,
    int* total)
{
    static const QRegularExpression progressPattern(
        QStringLiteral("(average frames|signal frames|Streamed frames)\\s+(\\d+)\\s*/\\s*(\\d+)"),
        QRegularExpression::CaseInsensitiveOption);

    const QRegularExpressionMatch match = progressPattern.match(line);
    if (!match.hasMatch()) {
        return false;
    }

    bool doneOk = false;
    bool totalOk = false;
    const int parsedDone = match.captured(2).toInt(&doneOk);
    const int parsedTotal = match.captured(3).toInt(&totalOk);
    if (!doneOk || !totalOk || parsedDone < 0 || parsedTotal <= 0) {
        return false;
    }

    QString label = match.captured(1).trimmed();
    if (label.compare(QStringLiteral("average frames"), Qt::CaseInsensitive) == 0) {
        label = QStringLiteral("Average");
    } else if (label.compare(QStringLiteral("signal frames"), Qt::CaseInsensitive) == 0) {
        label = QStringLiteral("Signals");
    } else {
        label = QStringLiteral("Split");
    }

    if (stageLabel != nullptr) {
        *stageLabel = label;
    }
    if (done != nullptr) {
        *done = parsedDone;
    }
    if (total != nullptr) {
        *total = parsedTotal;
    }
    return true;
}

MainWindow::ResultOpenMode MainWindow::resultOpenModeForPath(const QString& path)
{
    const QFileInfo info(path);
    if (!info.exists()) {
        return ResultOpenMode::Missing;
    }
    if (info.isDir()) {
        const auto isTiffFile = [](const QFileInfo& childInfo) {
            const QString suffix = childInfo.suffix().toLower();
            return suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff");
        };

        QDirIterator directIterator(info.absoluteFilePath(), QDir::Files, QDirIterator::NoIteratorFlags);
        for (int checked = 0; directIterator.hasNext() && checked < ResultFolderDirectScanLimit; ++checked) {
            if (isTiffFile(QFileInfo(directIterator.next()))) {
                return ResultOpenMode::FolderTiffs;
            }
        }

        QElapsedTimer timer;
        timer.start();
        QDirIterator iterator(info.absoluteFilePath(), QDir::Files, QDirIterator::Subdirectories);
        for (int checked = 0;
             iterator.hasNext()
             && checked < ResultFolderRecursiveScanLimit
             && timer.elapsed() < ResultFolderScanBudgetMs;
             ++checked) {
            if (isTiffFile(QFileInfo(iterator.next()))) {
                return ResultOpenMode::FolderTiffs;
            }
        }
        return ResultOpenMode::Folder;
    }

    const QString suffix = info.suffix().toLower();
    if (suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff")) {
        return ResultOpenMode::Viewer;
    }
    if (suffix == QStringLiteral("png")
        || suffix == QStringLiteral("jpg")
        || suffix == QStringLiteral("jpeg")
        || suffix == QStringLiteral("bmp")
        || suffix == QStringLiteral("gif")
        || suffix == QStringLiteral("webp")) {
        return ResultOpenMode::Image;
    }
    if (suffix == QStringLiteral("csv")
        || suffix == QStringLiteral("json")
        || suffix == QStringLiteral("txt")
        || suffix == QStringLiteral("tsv")
        || suffix == QStringLiteral("log")) {
        return ResultOpenMode::Text;
    }
    return ResultOpenMode::External;
}

QString MainWindow::resultPathFromLogLine(const QString& line, const QString& relativeRootPath)
{
    const QString trimmed = line.trimmed();
    if (!trimmed.startsWith(QStringLiteral("[OK]"))) {
        return {};
    }

    QString candidate;
    const int arrowIndex = trimmed.indexOf(QStringLiteral(" -> "));
    if (arrowIndex >= 0) {
        candidate = trimmed.mid(arrowIndex + 4).trimmed();
        const int summaryIndex = candidate.lastIndexOf(QStringLiteral(" ("));
        if (summaryIndex >= 0) {
            candidate = candidate.left(summaryIndex).trimmed();
        }
    } else {
        const int colonIndex = trimmed.indexOf(QStringLiteral(": "));
        if (colonIndex < 0) {
            return {};
        }
        candidate = trimmed.mid(colonIndex + 2).trimmed();
    }

    for (const QString& prefix : {
             QStringLiteral("average PNG "),
             QStringLiteral("signals "),
             QStringLiteral("manifest: "),
         }) {
        if (candidate.startsWith(prefix, Qt::CaseInsensitive)) {
            candidate = candidate.mid(prefix.size()).trimmed();
        }
    }

    if ((candidate.startsWith(QLatin1Char('"')) && candidate.endsWith(QLatin1Char('"')))
        || (candidate.startsWith(QLatin1Char('\'')) && candidate.endsWith(QLatin1Char('\'')))) {
        candidate = candidate.mid(1, candidate.size() - 2).trimmed();
    }
    if (candidate.isEmpty()) {
        return {};
    }

    QFileInfo info(candidate);
    if (info.isRelative() && !relativeRootPath.isEmpty()) {
        info.setFile(QDir(relativeRootPath).absoluteFilePath(candidate));
    }
    if (!info.exists()) {
        return {};
    }
    const QString canonical = info.canonicalFilePath();
    return canonical.isEmpty() ? info.absoluteFilePath() : canonical;
}

QString MainWindow::resultOpenModeName(ResultOpenMode mode)
{
    switch (mode) {
    case ResultOpenMode::Viewer:
        return QStringLiteral("viewer");
    case ResultOpenMode::Image:
        return QStringLiteral("image");
    case ResultOpenMode::Text:
        return QStringLiteral("text");
    case ResultOpenMode::FolderTiffs:
        return QStringLiteral("folder_tiffs");
    case ResultOpenMode::Folder:
        return QStringLiteral("folder");
    case ResultOpenMode::External:
        return QStringLiteral("external");
    case ResultOpenMode::Missing:
        return QStringLiteral("missing");
    }
    return QStringLiteral("unknown");
}

QString MainWindow::resultActionLabel(ResultOpenMode mode)
{
    switch (mode) {
    case ResultOpenMode::Viewer:
        return QStringLiteral("View TIFF");
    case ResultOpenMode::Image:
        return QStringLiteral("Preview Image");
    case ResultOpenMode::Text:
        return QStringLiteral("Preview Text");
    case ResultOpenMode::FolderTiffs:
        return QStringLiteral("Load TIFFs");
    case ResultOpenMode::Folder:
        return QStringLiteral("Open Folder");
    case ResultOpenMode::External:
        return QStringLiteral("Open File");
    case ResultOpenMode::Missing:
        return QStringLiteral("Missing");
    }
    return QStringLiteral("Open Selected");
}

void MainWindow::updateAnalysisProgressFromLine(const QString& line)
{
    if (analysisProgressBar_ == nullptr) {
        return;
    }

    QString stageLabel;
    int done = 0;
    int total = 0;
    if (!parseAnalysisProgressLine(line, &stageLabel, &done, &total)) {
        return;
    }

    analysisProgressBar_->setRange(0, total);
    analysisProgressBar_->setValue(std::min(done, total));
    analysisProgressBar_->setFormat(QStringLiteral("%1 %p% (%v/%m)").arg(stageLabel));
}

void MainWindow::setAnalysisProgressIdle(const QString& text)
{
    if (analysisProgressBar_ == nullptr) {
        return;
    }
    analysisProgressBar_->setRange(0, 1);
    analysisProgressBar_->setValue(text == QStringLiteral("Done") ? 1 : 0);
    analysisProgressBar_->setFormat(text);
}

void MainWindow::setAnalysisProgressBusy(const QString& text)
{
    if (analysisProgressBar_ == nullptr) {
        return;
    }
    analysisProgressBar_->setRange(0, 0);
    analysisProgressBar_->setFormat(text);
}

void MainWindow::setBackendStatus(const QString& text, const QString& toolTip)
{
    if (backendStatusLabel_ == nullptr) {
        return;
    }
    backendStatusLabel_->setText(text);
    backendStatusLabel_->setToolTip(toolTip);
}

void MainWindow::detectAnalysisOutputPath(const QString& line)
{
    const QString manifestPrefix = QStringLiteral("[OK] manifest:");
    if (!line.startsWith(manifestPrefix)) {
        return;
    }

    const QString manifestPath = line.mid(manifestPrefix.size()).trimmed();
    if (manifestPath.isEmpty()) {
        return;
    }

    QFileInfo manifestInfo(manifestPath);
    if (manifestInfo.isRelative()) {
        manifestInfo.setFile(QDir(repoRootPath()).absoluteFilePath(manifestPath));
    }
    lastManifestPath_ = manifestInfo.absoluteFilePath();
    setLastResultsPath(manifestInfo.absolutePath());
    detectWorkingTiffFromManifest();
    detectPreferredDisplayPathFromManifest();
    updateResultListFromManifest();
}

void MainWindow::detectAnalysisResultPath(const QString& line)
{
    const QString path = resultPathFromLogLine(line, repoRootPath());
    if (!path.isEmpty()) {
        addResultPath(path);
    }
}

void MainWindow::setLastResultsPath(const QString& path)
{
    lastResultsPath_ = path.isEmpty() ? QString() : QFileInfo(path).absoluteFilePath();
    if (openResultsButton_ != nullptr) {
        openResultsButton_->setEnabled(
            analysisProcess_ != nullptr
            && analysisProcess_->state() == QProcess::NotRunning
            && !lastResultsPath_.isEmpty());
        openResultsButton_->setToolTip(lastResultsPath_);
    }
}

QString MainWindow::findWorkingTiffFromManifest(
    const QString& manifestPath,
    const QString& currentSourcePath,
    QString* errorMessage,
    const QString& relativeRootPath)
{
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }

    auto fail = [errorMessage](const QString& message) -> QString {
        if (errorMessage != nullptr) {
            *errorMessage = message;
        }
        return {};
    };

    if (manifestPath.isEmpty()) {
        return fail(QStringLiteral("No manifest path was provided."));
    }

    const QFileInfo manifestInfo(manifestPath);
    if (!manifestInfo.isFile()) {
        return fail(QStringLiteral("Manifest does not exist: %1").arg(manifestInfo.absoluteFilePath()));
    }

    QFile file(manifestInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        return fail(
            QStringLiteral("Could not read manifest %1: %2")
                .arg(manifestInfo.absoluteFilePath(), file.errorString()));
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return fail(
            QStringLiteral("Could not parse pipeline manifest for working TIFF: %1")
                .arg(parseError.errorString()));
    }

    auto resolveManifestPath = [&manifestInfo, &relativeRootPath](const QString& path) -> QFileInfo {
        if (path.isEmpty()) {
            return QFileInfo();
        }
        QFileInfo info(path);
        if (info.isRelative()) {
            const QString root = relativeRootPath.isEmpty()
                ? manifestInfo.absolutePath()
                : relativeRootPath;
            info.setFile(QDir(root).absoluteFilePath(path));
        }
        return info;
    };

    auto stablePath = [](const QFileInfo& info) -> QString {
        const QString canonical = info.canonicalFilePath();
        if (!canonical.isEmpty()) {
            return canonical;
        }
        return info.absoluteFilePath();
    };

    const QJsonArray records = document.object().value(QStringLiteral("tiffs")).toArray();
    const QString currentPath = currentSourcePath.isEmpty()
        ? QString()
        : stablePath(QFileInfo(currentSourcePath));
    QString fallbackPath;

    for (const QJsonValue& value : records) {
        const QJsonObject record = value.toObject();
        if (!record.value(QStringLiteral("ok")).toBool(true)) {
            continue;
        }

        const QString workingText = record.value(QStringLiteral("working_tiff")).toString();
        if (workingText.isEmpty()) {
            continue;
        }

        const QFileInfo workingInfo = resolveManifestPath(workingText);
        const QString suffix = workingInfo.suffix().toLower();
        if (!workingInfo.isFile() || (suffix != QStringLiteral("tif") && suffix != QStringLiteral("tiff"))) {
            continue;
        }

        const QString sourceText = record.value(QStringLiteral("tiff")).toString();
        if (sourceText.isEmpty()) {
            continue;
        }
        const QFileInfo sourceInfo = resolveManifestPath(sourceText);
        const QString workingPath = stablePath(workingInfo);
        const QString sourcePath = stablePath(sourceInfo);
        if (!sourcePath.isEmpty() && workingPath.compare(sourcePath, Qt::CaseInsensitive) == 0) {
            continue;
        }

        if (!currentPath.isEmpty() && sourcePath.compare(currentPath, Qt::CaseInsensitive) == 0) {
            return workingPath;
        }
        if (fallbackPath.isEmpty()) {
            fallbackPath = workingPath;
        }
    }

    return fallbackPath;
}

QString MainWindow::findPreferredDisplayPathFromManifest(
    const QString& manifestPath,
    const QString& currentSourcePath,
    QString* errorMessage,
    const QString& relativeRootPath)
{
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }

    auto fail = [errorMessage](const QString& message) -> QString {
        if (errorMessage != nullptr) {
            *errorMessage = message;
        }
        return {};
    };

    if (manifestPath.isEmpty()) {
        return fail(QStringLiteral("No manifest path was provided."));
    }

    const QFileInfo manifestInfo(manifestPath);
    if (!manifestInfo.isFile()) {
        return fail(QStringLiteral("Manifest does not exist: %1").arg(manifestInfo.absoluteFilePath()));
    }

    QFile file(manifestInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        return fail(
            QStringLiteral("Could not read manifest %1: %2")
                .arg(manifestInfo.absoluteFilePath(), file.errorString()));
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return fail(
            QStringLiteral("Could not parse pipeline manifest for display output: %1")
                .arg(parseError.errorString()));
    }

    auto resolveManifestPath = [&manifestInfo, &relativeRootPath](const QString& path) -> QFileInfo {
        if (path.isEmpty()) {
            return QFileInfo();
        }
        QFileInfo info(path);
        if (info.isRelative()) {
            const QString root = relativeRootPath.isEmpty()
                ? manifestInfo.absolutePath()
                : relativeRootPath;
            info.setFile(QDir(root).absoluteFilePath(path));
        }
        return info;
    };

    auto stablePath = [](const QFileInfo& info) -> QString {
        const QString canonical = info.canonicalFilePath();
        return canonical.isEmpty() ? info.absoluteFilePath() : canonical;
    };

    auto firstExistingPathValue = [&](const QJsonValue& value) -> QString {
        if (value.isString()) {
            const QFileInfo info = resolveManifestPath(value.toString());
            if (info.isFile()) {
                return stablePath(info);
            }
            return {};
        }
        if (value.isArray()) {
            const QJsonArray array = value.toArray();
            for (const QJsonValue& child : array) {
                if (!child.isString()) {
                    continue;
                }
                const QFileInfo info = resolveManifestPath(child.toString());
                if (info.isFile()) {
                    return stablePath(info);
                }
            }
        }
        return {};
    };
    auto firstExistingPath = [&](const QJsonObject& object, const QStringList& keys) -> QString {
        for (const QString& key : keys) {
            const QString candidate = firstExistingPathValue(object.value(key));
            if (!candidate.isEmpty()) {
                return candidate;
            }
        }
        return {};
    };

    const QString currentPath = currentSourcePath.isEmpty()
        ? QString()
        : stablePath(QFileInfo(currentSourcePath));
    QString fallbackPath;
    const QJsonArray records = document.object().value(QStringLiteral("tiffs")).toArray();
    for (const QJsonValue& value : records) {
        const QJsonObject record = value.toObject();
        if (!record.value(QStringLiteral("ok")).toBool(true)) {
            continue;
        }

        const QString sourceText = record.value(QStringLiteral("tiff")).toString();
        const QString sourcePath = sourceText.isEmpty() ? QString() : stablePath(resolveManifestPath(sourceText));
        const QJsonObject actions = record.value(QStringLiteral("actions")).toObject();

        QString candidate = firstExistingPath(
            actions.value(QStringLiteral("roi_detection")).toObject(),
            QStringList{
                QStringLiteral("tracked_debug_tiffs"),
                QStringLiteral("tracked_debug_raw_tiffs"),
                QStringLiteral("tracked_debug_mask_tiffs"),
                QStringLiteral("tracked_trace_xlsx"),
                QStringLiteral("tracked_manifest_path"),
                QStringLiteral("initial_mask_tiff"),
                QStringLiteral("initial_mask_motion_qc_json"),
                QStringLiteral("initial_mask_motion_shifts_csv"),
                QStringLiteral("qc_overlay_png"),
                QStringLiteral("segmentation_overlay_png"),
                QStringLiteral("label_tiff"),
            });
        if (candidate.isEmpty()) {
            candidate = firstExistingPath(
                actions.value(QStringLiteral("motion_correction")).toObject(),
                QStringList{
                    QStringLiteral("correlation_plot_png"),
                    QStringLiteral("corrected_tiff"),
                    QStringLiteral("corrected_tiff_path"),
                    QStringLiteral("output_tiff"),
                    QStringLiteral("output_tiff_path"),
                    QStringLiteral("downstream_tiff"),
                });
        }
        if (candidate.isEmpty()) {
            const QString workingText = record.value(QStringLiteral("working_tiff")).toString();
            const QString sourceTiff = record.value(QStringLiteral("tiff")).toString();
            if (!workingText.isEmpty() && workingText != sourceTiff) {
                const QFileInfo workingInfo = resolveManifestPath(workingText);
                if (workingInfo.isFile()) {
                    candidate = stablePath(workingInfo);
                }
            }
        }
        if (candidate.isEmpty()) {
            continue;
        }
        if (!currentPath.isEmpty() && sourcePath.compare(currentPath, Qt::CaseInsensitive) == 0) {
            return candidate;
        }
        if (fallbackPath.isEmpty()) {
            fallbackPath = candidate;
        }
    }

    return fallbackPath;
}

QStringList MainWindow::findResultPathsFromManifest(
    const QString& manifestPath,
    QString* errorMessage,
    const QString& relativeRootPath)
{
    if (errorMessage != nullptr) {
        errorMessage->clear();
    }

    auto fail = [errorMessage](const QString& message) -> QStringList {
        if (errorMessage != nullptr) {
            *errorMessage = message;
        }
        return {};
    };

    if (manifestPath.isEmpty()) {
        return fail(QStringLiteral("No manifest path was provided."));
    }

    const QFileInfo manifestInfo(manifestPath);
    if (!manifestInfo.isFile()) {
        return fail(QStringLiteral("Manifest does not exist: %1").arg(manifestInfo.absoluteFilePath()));
    }

    QFile file(manifestInfo.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        return fail(
            QStringLiteral("Could not read manifest %1: %2")
                .arg(manifestInfo.absoluteFilePath(), file.errorString()));
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return fail(
            QStringLiteral("Could not parse pipeline manifest for results: %1")
                .arg(parseError.errorString()));
    }

    auto resolveManifestPath = [&manifestInfo, &relativeRootPath](const QString& path) -> QFileInfo {
        if (path.isEmpty()) {
            return QFileInfo();
        }
        QFileInfo info(path);
        if (info.isRelative()) {
            const QString root = relativeRootPath.isEmpty()
                ? manifestInfo.absolutePath()
                : relativeRootPath;
            info.setFile(QDir(root).absoluteFilePath(path));
        }
        return info;
    };

    QStringList paths;
    QSet<QString> seen;
    auto appendPath = [](QStringList& paths, QSet<QString>& seen, const QFileInfo& info) {
        if (!info.exists()) {
            return;
        }
        const QString canonical = info.canonicalFilePath();
        const QString absolute = canonical.isEmpty() ? info.absoluteFilePath() : canonical;
        const QString key = absolute.toLower();
        if (seen.contains(key)) {
            return;
        }
        seen.insert(key);
        paths.append(absolute);
    };
    auto appendPathValue = [&](const QJsonValue& pathValue) {
        if (pathValue.isString()) {
            appendPath(paths, seen, resolveManifestPath(pathValue.toString()));
            return;
        }
        if (pathValue.isArray()) {
            const QJsonArray array = pathValue.toArray();
            for (const QJsonValue& child : array) {
                if (child.isString()) {
                    appendPath(paths, seen, resolveManifestPath(child.toString()));
                }
            }
        }
    };
    auto appendKnownActionPaths = [&](const QJsonObject& action, const QStringList& keys) {
        for (const QString& key : keys) {
            appendPathValue(action.value(key));
        }
    };

    const QJsonArray records = document.object().value(QStringLiteral("tiffs")).toArray();
    for (const QJsonValue& value : records) {
        const QJsonObject record = value.toObject();
        const QJsonObject actions = record.value(QStringLiteral("actions")).toObject();

        const QJsonObject average = actions.value(QStringLiteral("average_png")).toObject();
        appendPath(paths, seen, resolveManifestPath(average.value(QStringLiteral("path")).toString()));

        const QJsonObject signalAction = actions.value(QStringLiteral("signals")).toObject();
        appendPath(paths, seen, resolveManifestPath(signalAction.value(QStringLiteral("xlsx")).toString()));
        appendPath(paths, seen, resolveManifestPath(signalAction.value(QStringLiteral("csv")).toString()));
        appendKnownActionPaths(
            signalAction,
            QStringList{
                QStringLiteral("mask_path"),
                QStringLiteral("mask_manifest_path"),
                QStringLiteral("tracked_manifest_path"),
                QStringLiteral("tracked_tracks_csv"),
                QStringLiteral("tracked_debug_tiffs"),
                QStringLiteral("tracked_debug_raw_tiffs"),
                QStringLiteral("tracked_debug_mask_tiffs"),
                QStringLiteral("initial_mask_tiff"),
                QStringLiteral("initial_mask_motion_qc_json"),
                QStringLiteral("initial_mask_motion_shifts_csv"),
            });

        const QJsonObject comparisonAction = actions.value(QStringLiteral("signal_comparison_csvs")).toObject();
        appendPathValue(comparisonAction.value(QStringLiteral("csvs")));
        appendPath(paths, seen, resolveManifestPath(comparisonAction.value(QStringLiteral("xlsx")).toString()));
        const QJsonObject comparisonWorkbook = comparisonAction.value(QStringLiteral("workbook")).toObject();
        appendPath(paths, seen, resolveManifestPath(comparisonWorkbook.value(QStringLiteral("path")).toString()));
        const QJsonObject comparisonVariants = comparisonAction.value(QStringLiteral("variants")).toObject();
        for (auto variant = comparisonVariants.constBegin(); variant != comparisonVariants.constEnd(); ++variant) {
            const QJsonObject variantObject = variant.value().toObject();
            appendPath(paths, seen, resolveManifestPath(variantObject.value(QStringLiteral("csv")).toString()));
            appendKnownActionPaths(
                variantObject,
                QStringList{
                    QStringLiteral("xlsx"),
                    QStringLiteral("tracked_manifest_path"),
                    QStringLiteral("tracked_tracks_csv"),
                    QStringLiteral("tracked_debug_tiffs"),
                    QStringLiteral("tracked_debug_raw_tiffs"),
                    QStringLiteral("tracked_debug_mask_tiffs"),
                });
        }

        const QJsonObject qcImages = actions.value(QStringLiteral("qc_images")).toObject();
        appendKnownActionPaths(
            qcImages,
            QStringList{
                QStringLiteral("raw_average_png"),
                QStringLiteral("motion_vs_raw_qc_png"),
                QStringLiteral("segmentation_overlay_png"),
                QStringLiteral("raw_segmentation_overlay_png"),
            });

        const QJsonObject split = actions.value(QStringLiteral("split")).toObject();
        appendPath(paths, seen, resolveManifestPath(split.value(QStringLiteral("output_dir")).toString()));
        appendPath(paths, seen, resolveManifestPath(split.value(QStringLiteral("roi_label_png")).toString()));

        const QJsonObject motionAction = actions.value(QStringLiteral("motion_correction")).toObject();
        appendKnownActionPaths(
            motionAction,
            QStringList{
                QStringLiteral("corrected_tiff"),
                QStringLiteral("corrected_tiff_path"),
                QStringLiteral("output_tiff"),
                QStringLiteral("output_tiff_path"),
                QStringLiteral("downstream_tiff"),
                QStringLiteral("qc_json"),
                QStringLiteral("shifts_csv"),
                QStringLiteral("correlation_scores_csv"),
                QStringLiteral("correlation_plot_png"),
                QStringLiteral("mean_image_png"),
                QStringLiteral("correlation_qc_json"),
            });

        const QJsonObject roiAction = actions.value(QStringLiteral("roi_detection")).toObject();
        appendKnownActionPaths(
            roiAction,
            QStringList{
                QStringLiteral("qc_overlay_png"),
                QStringLiteral("segmentation_overlay_png"),
                QStringLiteral("label_tiff"),
                QStringLiteral("summary_mean_png"),
                QStringLiteral("summary_std_png"),
                QStringLiteral("summary_max_png"),
                QStringLiteral("mask_path"),
                QStringLiteral("masks_path"),
                QStringLiteral("roi_mask_path"),
                QStringLiteral("output_masks"),
                QStringLiteral("segmentation_mask_path"),
                QStringLiteral("mask_manifest_path"),
                QStringLiteral("manifest_path"),
                QStringLiteral("segmentation_manifest_path"),
                QStringLiteral("roi_manifest_path"),
                QStringLiteral("summary_manifest_path"),
                QStringLiteral("summary_paths"),
                QStringLiteral("tracked_trace_csv"),
                QStringLiteral("tracked_trace_xlsx"),
                QStringLiteral("tracked_manifest_path"),
                QStringLiteral("tracked_tracks_csv"),
                QStringLiteral("tracked_debug_tiffs"),
                QStringLiteral("tracked_debug_raw_tiffs"),
                QStringLiteral("tracked_debug_mask_tiffs"),
                QStringLiteral("initial_mask_tiff"),
                QStringLiteral("initial_mask_motion_qc_json"),
                QStringLiteral("initial_mask_motion_shifts_csv"),
            });

        const QString workingTiff = record.value(QStringLiteral("working_tiff")).toString();
        const QString sourceTiff = record.value(QStringLiteral("tiff")).toString();
        if (!workingTiff.isEmpty() && workingTiff != sourceTiff) {
            appendPath(paths, seen, resolveManifestPath(workingTiff));
        }
    }

    appendPath(paths, seen, manifestInfo);
    return paths;
}

void MainWindow::detectWorkingTiffFromManifest()
{
    if (lastManifestPath_.isEmpty()) {
        setLastWorkingTiffPath(QString());
        return;
    }

    QString errorMessage;
    const QString sourcePath = analysisAutoOpenSourcePath_.isEmpty()
        ? currentQueuePath()
        : analysisAutoOpenSourcePath_;
    const QString workingTiffPath =
        findWorkingTiffFromManifest(lastManifestPath_, sourcePath, &errorMessage, repoRootPath());
    if (!errorMessage.isEmpty()) {
        appendAnalysisLog(QStringLiteral("[WARN] %1").arg(errorMessage));
    }
    setLastWorkingTiffPath(workingTiffPath);
}

void MainWindow::detectPreferredDisplayPathFromManifest()
{
    lastDisplayResultPath_.clear();
    if (lastManifestPath_.isEmpty()) {
        updateProcessedOutputButton();
        return;
    }

    QString errorMessage;
    const QString sourcePath = analysisAutoOpenSourcePath_.isEmpty()
        ? currentQueuePath()
        : analysisAutoOpenSourcePath_;
    lastDisplayResultPath_ =
        findPreferredDisplayPathFromManifest(lastManifestPath_, sourcePath, &errorMessage, repoRootPath());
    if (!errorMessage.isEmpty()) {
        appendAnalysisLog(QStringLiteral("[WARN] %1").arg(errorMessage));
    }
    updateProcessedOutputButton();
}

void MainWindow::setLastWorkingTiffPath(const QString& path)
{
    lastWorkingTiffPath_ = path.isEmpty() ? QString() : QFileInfo(path).absoluteFilePath();
    updateProcessedOutputButton();
}

void MainWindow::updateProcessedOutputButton()
{
    if (openWorkingTiffButton_ == nullptr) {
        return;
    }

    const QString displayPath = lastDisplayResultPath_.isEmpty() ? lastWorkingTiffPath_ : lastDisplayResultPath_;
    const bool available =
        analysisProcess_ != nullptr
        && analysisProcess_->state() == QProcess::NotRunning
        && !displayPath.isEmpty();
    openWorkingTiffButton_->setEnabled(available);
    openWorkingTiffButton_->setToolTip(
        displayPath.isEmpty()
            ? QStringLiteral("Open latest segmentation overlay or corrected/downstream TIFF")
            : displayPath);
}

void MainWindow::updateResultListFromManifest()
{
    if (lastManifestPath_.isEmpty()) {
        setResultPaths(QStringList());
        return;
    }

    QString errorMessage;
    const QStringList paths = findResultPathsFromManifest(lastManifestPath_, &errorMessage, repoRootPath());
    if (!errorMessage.isEmpty()) {
        appendAnalysisLog(QStringLiteral("[WARN] %1").arg(errorMessage));
    }
    setResultPaths(paths);
}

void MainWindow::setResultPaths(const QStringList& paths)
{
    if (resultsList_ == nullptr) {
        return;
    }

    resultsList_->clear();
    for (const QString& path : paths) {
        const QFileInfo info(path);
        QString label = info.fileName();
        if (label.isEmpty()) {
            label = info.absoluteFilePath();
        }
        if (info.isDir()) {
            label += QStringLiteral(" (folder)");
        }

        auto* item = new QListWidgetItem(label);
        item->setData(Qt::UserRole, info.absoluteFilePath());
        item->setToolTip(info.absoluteFilePath());
        resultsList_->addItem(item);
    }

    if (resultsList_->count() > 0) {
        resultsList_->setCurrentRow(0);
    }

    const bool busy = analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning;
    Q_UNUSED(busy);
    updateResultActionUi();
}

void MainWindow::addResultPath(const QString& path)
{
    if (resultsList_ == nullptr || path.isEmpty()) {
        return;
    }

    const QFileInfo info(path);
    if (!info.exists()) {
        return;
    }

    const QString canonical = info.canonicalFilePath();
    const QString absolutePath = canonical.isEmpty() ? info.absoluteFilePath() : canonical;
    for (int row = 0; row < resultsList_->count(); ++row) {
        const QString existing = resultsList_->item(row)->data(Qt::UserRole).toString();
        if (existing.compare(absolutePath, Qt::CaseInsensitive) == 0) {
            return;
        }
    }

    QString label = info.fileName();
    if (label.isEmpty()) {
        label = info.absoluteFilePath();
    }
    if (info.isDir()) {
        label += QStringLiteral(" (folder)");
    }

    auto* item = new QListWidgetItem(label);
    item->setData(Qt::UserRole, absolutePath);
    item->setToolTip(absolutePath);
    resultsList_->addItem(item);
    if (resultsList_->count() == 1) {
        resultsList_->setCurrentRow(0);
    }

    const bool busy = analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning;
    Q_UNUSED(busy);
    updateResultActionUi();
}

void MainWindow::updateResultActionUi()
{
    if (openResultButton_ == nullptr || resultsList_ == nullptr) {
        return;
    }

    const bool busy = analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning;
    const QListWidgetItem* item = resultsList_->currentItem();
    if (item == nullptr) {
        openResultButton_->setText(QStringLiteral("Open Selected"));
        openResultButton_->setToolTip(QString());
        openResultButton_->setEnabled(false);
        return;
    }

    const QString path = item->data(Qt::UserRole).toString();
    const ResultOpenMode mode = resultOpenModeForPath(path);
    openResultButton_->setText(resultActionLabel(mode));
    openResultButton_->setToolTip(path);
    openResultButton_->setEnabled(!busy && mode != ResultOpenMode::Missing);
}

void MainWindow::setAnalysisBusy(bool busy)
{
    const int queueCount = queueList_->count();
    const int currentRow = queueList_->currentRow();
    const bool queueAdding = !pendingQueuePaths_.isEmpty();
    splitButton_->setEnabled(!busy);
    exportSignalsButton_->setEnabled(!busy);
    averagePngButton_->setEnabled(!busy);
    runPipelineButton_->setEnabled(!busy);
    runQueuePipelineButton_->setEnabled(!busy && !queueAdding);
    benchmarkQueueButton_->setEnabled(!busy && !queueAdding && queueCount > 0);
    removeQueueButton_->setEnabled(!busy && !queueAdding && currentRow >= 0 && currentRow < queueCount);
    clearQueueButton_->setEnabled(!busy && (queueCount > 0 || queueAdding));
    cancelAnalysisButton_->setEnabled(busy);
    openResultsButton_->setEnabled(!busy && !lastResultsPath_.isEmpty());
    updateProcessedOutputButton();
    resultsList_->setEnabled(!busy);
    updateResultActionUi();
    useMetadataSamplingCheck_->setEnabled(!busy);
    samplingRateSpin_->setEnabled(!busy && !useMetadataSamplingCheck_->isChecked());
    writeCsvCheck_->setEnabled(!busy);
    pipelineAverageCheck_->setEnabled(!busy);
    pipelineSignalsCheck_->setEnabled(!busy);
    pipelineSplitCheck_->setEnabled(!busy);
    autoOpenWorkingTiffCheck_->setEnabled(!busy);
    pythonBackendEdit_->setEnabled(!busy);
    choosePythonBackendButton_->setEnabled(!busy);
    validateHooksButton_->setEnabled(!busy);
    updateProcessingStageUi();
}

void MainWindow::openQueueRow(int row)
{
    if (row < 0 || row >= queueList_->count()) {
        setQueueStatus();
        return;
    }

    const QString path = queueList_->item(row)->data(Qt::UserRole).toString();
    if (!path.isEmpty()) {
        viewer_->loadFile(path);
    }
    setQueueStatus();
}

void MainWindow::selectQueuePath(const QString& path)
{
    const QString absolutePath = QFileInfo(path).absoluteFilePath();
    const QSignalBlocker blocker(queueList_);
    for (int row = 0; row < queueList_->count(); ++row) {
        if (queueList_->item(row)->data(Qt::UserRole).toString() == absolutePath) {
            queueList_->setCurrentRow(row);
            setQueueStatus();
            return;
        }
    }
    setQueueStatus();
}

void MainWindow::setQueueStatus()
{
    const int count = queueList_->count();
    const int currentRow = queueList_->currentRow();
    const bool busy =
        (analysisProcess_ != nullptr && analysisProcess_->state() != QProcess::NotRunning)
        || queueBenchmarkActive_;
    const bool queueAdding = !pendingQueuePaths_.isEmpty();
    removeQueueButton_->setEnabled(!busy && !queueAdding && currentRow >= 0 && currentRow < count);
    clearQueueButton_->setEnabled(!busy && (count > 0 || queueAdding));
    runQueuePipelineButton_->setEnabled(!busy && !queueAdding);
    benchmarkQueueButton_->setEnabled(!busy && !queueAdding && count > 0);
    if (queueAdding) {
        const int pendingTotal = static_cast<int>(pendingQueuePaths_.size());
        const int pendingDone = std::min(pendingQueueIndex_, pendingTotal);
        queueStatus_->setText(
            QStringLiteral("Adding TIFFs... %1/%2")
                .arg(pendingDone)
                .arg(pendingTotal));
        return;
    }
    if (count <= 0) {
        queueStatus_->setText(QStringLiteral("0 TIFFs"));
        return;
    }
    if (currentRow >= 0 && currentRow < count) {
        queueStatus_->setText(QStringLiteral("%1 TIFFs | current %2").arg(count).arg(currentRow + 1));
        return;
    }
    queueStatus_->setText(QStringLiteral("%1 TIFFs").arg(count));
}

void MainWindow::loadUserSettings()
{
    QSettings settings;
    const QByteArray geometry = settings.value(QStringLiteral("mainWindow/geometry")).toByteArray();
    if (!geometry.isEmpty()) {
        restoreGeometry(geometry);
    }

    outputFolderEdit_->setText(settings.value(QStringLiteral("analysis/outputFolder")).toString());
    recursiveCheck_->setChecked(settings.value(QStringLiteral("queue/recursive"), true).toBool());
    useMetadataSamplingCheck_->setChecked(
        settings.value(QStringLiteral("analysis/useMetadataSampling"), true).toBool());
    samplingRateSpin_->setValue(settings.value(QStringLiteral("analysis/manualSamplingHz"), 1.0).toDouble());
    samplingRateSpin_->setEnabled(!useMetadataSamplingCheck_->isChecked());
    writeCsvCheck_->setChecked(settings.value(QStringLiteral("analysis/writeCsv"), true).toBool());
    const bool dynamicTrackedSegmentation =
        settings.value(QStringLiteral("pipeline/dynamicTrackedSegmentation"), false).toBool();
    {
        QSignalBlocker motionBlocker(runMotionCorrectionCheck_);
        QSignalBlocker segmentationBlocker(runSegmentationCheck_);
        QSignalBlocker dynamicBlocker(runDynamicSegmentationCheck_);
        runDynamicSegmentationCheck_->setChecked(dynamicTrackedSegmentation);
        runMotionCorrectionCheck_->setChecked(
            dynamicTrackedSegmentation
                ? false
                : settings.value(QStringLiteral("pipeline/motionCorrection"), true).toBool());
        runSegmentationCheck_->setChecked(
            dynamicTrackedSegmentation
                ? false
                : settings.value(QStringLiteral("pipeline/segmentation"), true).toBool());
    }
    pipelineAverageCheck_->setChecked(settings.value(QStringLiteral("pipeline/averagePng"), true).toBool());
    pipelineSignalsCheck_->setChecked(settings.value(QStringLiteral("pipeline/signals"), true).toBool());
    const QString maskBackgroundMode =
        settings.value(QStringLiteral("pipeline/maskBackground"), QStringLiteral("none")).toString();
    const int maskBackgroundIndex = maskBackgroundCombo_->findData(maskBackgroundMode);
    maskBackgroundCombo_->setCurrentIndex(maskBackgroundIndex >= 0 ? maskBackgroundIndex : 0);
    maskBackgroundCombo_->setEnabled(pipelineSignalsCheck_->isChecked());
    pipelineSplitCheck_->setChecked(settings.value(QStringLiteral("pipeline/splitRois"), true).toBool());
    autoOpenWorkingTiffCheck_->setChecked(
        settings.value(QStringLiteral("pipeline/autoOpenWorkingTiff"), true).toBool());
    pythonBackendEdit_->setText(settings.value(QStringLiteral("analysis/pythonBackend")).toString());
    const QString savedMotionHook = settings.value(QStringLiteral("hooks/motion")).toString().trimmed();
    const QString savedMotionParams = settings.value(QStringLiteral("hooks/motionParams")).toString().trimmed();
    const QString savedRoiHook = settings.value(QStringLiteral("hooks/roi")).toString().trimmed();
    motionHookEdit_->setText(savedMotionHook.isEmpty() ? builtInMotionHookSpec() : savedMotionHook);
    motionHookParamsEdit_->setText(savedMotionParams.isEmpty() ? builtInMotionHookParams() : savedMotionParams);
    roiHookEdit_->setText(savedRoiHook.isEmpty() ? builtInRoiHookSpec() : savedRoiHook);
    roiHookParamsEdit_->setText(settings.value(QStringLiteral("hooks/roiParams")).toString());
    updateProcessingStageUi();

    const QString outputFolder = outputFolderEdit_->text().trimmed();
    if (!outputFolder.isEmpty()) {
        setLastResultsPath(outputFolder);
    }
}

void MainWindow::saveUserSettings() const
{
    QSettings settings;
    settings.setValue(QStringLiteral("mainWindow/geometry"), saveGeometry());
    settings.setValue(QStringLiteral("analysis/outputFolder"), outputFolderEdit_->text().trimmed());
    settings.setValue(QStringLiteral("queue/recursive"), recursiveCheck_->isChecked());
    settings.setValue(QStringLiteral("analysis/useMetadataSampling"), useMetadataSamplingCheck_->isChecked());
    settings.setValue(QStringLiteral("analysis/manualSamplingHz"), samplingRateSpin_->value());
    settings.setValue(QStringLiteral("analysis/writeCsv"), writeCsvCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/motionCorrection"), runMotionCorrectionCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/segmentation"), runSegmentationCheck_->isChecked());
    settings.setValue(
        QStringLiteral("pipeline/dynamicTrackedSegmentation"),
        runDynamicSegmentationCheck_ != nullptr && runDynamicSegmentationCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/averagePng"), pipelineAverageCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/signals"), pipelineSignalsCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/maskBackground"), maskBackgroundCombo_->currentData().toString());
    settings.setValue(QStringLiteral("pipeline/splitRois"), pipelineSplitCheck_->isChecked());
    settings.setValue(QStringLiteral("pipeline/autoOpenWorkingTiff"), autoOpenWorkingTiffCheck_->isChecked());
    settings.setValue(QStringLiteral("analysis/pythonBackend"), pythonBackendEdit_->text().trimmed());
    settings.setValue(QStringLiteral("hooks/motion"), motionHookEdit_->text().trimmed());
    settings.setValue(QStringLiteral("hooks/motionParams"), motionHookParamsEdit_->text().trimmed());
    settings.setValue(QStringLiteral("hooks/roi"), roiHookEdit_->text().trimmed());
    settings.setValue(QStringLiteral("hooks/roiParams"), roiHookParamsEdit_->text().trimmed());
}

QString MainWindow::currentQueuePath() const
{
    const int row = queueList_->currentRow();
    if (row < 0 || row >= queueList_->count()) {
        return {};
    }
    return queueList_->item(row)->data(Qt::UserRole).toString();
}

QString MainWindow::pythonExecutable(QStringList* diagnostics, QStringList* pythonPathEntries) const
{
    const QString selectedPython = pythonBackendEdit_ == nullptr
        ? QString()
        : pythonBackendEdit_->text().trimmed();
    const QString configuredPython = qEnvironmentVariable("SPIKE_PYTHON");
    const QDir repoRoot(repoRootPath());

    struct Candidate {
        QString path;
        QString label;
        bool required = false;
        QStringList pythonPathEntries;
    };

    QList<Candidate> candidates;
    QSet<QString> seenCandidatePaths;
    const auto appendCandidate = [&candidates, &seenCandidatePaths](
                                     const QString& path,
                                     const QString& label,
                                     bool required,
                                     const QStringList& candidatePythonPathEntries = {}) {
        QString trimmedPath = path.trimmed();
        if (trimmedPath.size() >= 2) {
            const QChar first = trimmedPath.front();
            const QChar last = trimmedPath.back();
            if ((first == QLatin1Char('"') && last == QLatin1Char('"'))
                || (first == QLatin1Char('\'') && last == QLatin1Char('\''))) {
                trimmedPath = trimmedPath.mid(1, trimmedPath.size() - 2).trimmed();
            }
        }
        if (trimmedPath.isEmpty()) {
            return;
        }
        if (QFileInfo::exists(trimmedPath)) {
            trimmedPath = QFileInfo(trimmedPath).absoluteFilePath();
        }
        const QString key = trimmedPath.toCaseFolded();
        if (seenCandidatePaths.contains(key)) {
            return;
        }
        seenCandidatePaths.insert(key);
        candidates.append({trimmedPath, label, required, candidatePythonPathEntries});
    };

    if (!selectedPython.isEmpty()) {
        appendCandidate(selectedPython, QStringLiteral("selected Python backend"), true);
    }
    if (!configuredPython.isEmpty()) {
        appendCandidate(configuredPython, QStringLiteral("SPIKE_PYTHON"), true);
    }

    const auto appendExistingPath =
        [&appendCandidate](
            const QString& path,
            const QString& label,
            const QStringList& candidatePythonPathEntries = {}) {
        if (QFileInfo::exists(path)) {
            appendCandidate(
                QFileInfo(path).absoluteFilePath(),
                label,
                false,
                candidatePythonPathEntries);
        }
    };

    const auto pyvenvExecutable = [](const QString& venvRootPath) {
        QFile config(QDir(venvRootPath).absoluteFilePath(QStringLiteral("pyvenv.cfg")));
        if (!config.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return QString();
        }
        const QStringList lines = QString::fromUtf8(config.readAll()).split(QLatin1Char('\n'));
        for (QString line : lines) {
            line = line.trimmed();
            const int separator = line.indexOf(QLatin1Char('='));
            if (separator < 0) {
                continue;
            }
            const QString key = line.left(separator).trimmed().toLower();
            if (key != QStringLiteral("executable")) {
                continue;
            }
            const QString executable = line.mid(separator + 1).trimmed();
            if (QFileInfo::exists(executable)) {
                return QFileInfo(executable).absoluteFilePath();
            }
        }
        return QString();
    };

    const auto appendVenvCandidates = [&appendExistingPath, &pyvenvExecutable](
                                          const QString& venvRootPath,
                                          const QString& label) {
        const QDir venvRoot(venvRootPath);
        appendExistingPath(
            venvRoot.absoluteFilePath(QStringLiteral("Scripts/python.exe")),
            label);

        const QString sitePackages = venvRoot.absoluteFilePath(QStringLiteral("Lib/site-packages"));
        const QString baseExecutable = pyvenvExecutable(venvRoot.absolutePath());
        if (!baseExecutable.isEmpty()) {
            appendExistingPath(
                baseExecutable,
                QStringLiteral("%1 base interpreter").arg(label),
                QFileInfo::exists(sitePackages) ? QStringList{sitePackages} : QStringList{});
        }
    };

    appendVenvCandidates(
        repoRoot.absoluteFilePath(QStringLiteral("../.venv")),
        QStringLiteral("sibling virtual environment"));
    appendVenvCandidates(
        repoRoot.absoluteFilePath(QStringLiteral(".venv")),
        QStringLiteral("repository virtual environment"));

    const QString codexRuntimePython =
        QStringLiteral(".cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe");
    appendCandidate(
        QDir(QDir::homePath()).absoluteFilePath(codexRuntimePython),
        QStringLiteral("bundled Codex Python fallback"),
        false);
    const QString userProfile = qEnvironmentVariable("USERPROFILE");
    if (!userProfile.isEmpty()) {
        appendCandidate(
            QDir(userProfile).absoluteFilePath(codexRuntimePython),
            QStringLiteral("USERPROFILE bundled Codex Python fallback"),
            false);
    }

    appendCandidate(QStringLiteral("python"), QStringLiteral("PATH python"), false);

    const auto formatProbeOutput = [](const QByteArray& output) {
        QString text = QString::fromLocal8Bit(output).trimmed();
        text.replace(QStringLiteral("\r\n"), QStringLiteral(" "));
        text.replace(QLatin1Char('\n'), QLatin1Char(' '));
        if (text.size() > 240) {
            text = text.left(237) + QStringLiteral("...");
        }
        return text;
    };

    const auto environmentWithPythonPath = [](const QStringList& candidatePythonPathEntries) {
        QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
        if (!candidatePythonPathEntries.isEmpty()) {
            const QString existingPythonPath = environment.value(QStringLiteral("PYTHONPATH"));
            QStringList combinedPythonPath = candidatePythonPathEntries;
            if (!existingPythonPath.isEmpty()) {
                combinedPythonPath.append(existingPythonPath.split(QDir::listSeparator(), Qt::SkipEmptyParts));
            }
            environment.insert(QStringLiteral("PYTHONPATH"), combinedPythonPath.join(QDir::listSeparator()));
        }
        return environment;
    };

    const auto canStartPython = [&formatProbeOutput, &environmentWithPythonPath](
                                    const Candidate& candidate,
                                    QString* detail) {
        QProcess probe;
        probe.setProgram(candidate.path);
        probe.setArguments(
            QStringList() << QStringLiteral("-c")
                          << QStringLiteral("import sys; raise SystemExit(0)"));
        probe.setProcessEnvironment(environmentWithPythonPath(candidate.pythonPathEntries));
        probe.start();

        bool started = false;
        QElapsedTimer startTimer;
        startTimer.start();
        while (startTimer.elapsed() < 1500) {
            if (probe.waitForStarted(25)) {
                started = true;
                break;
            }
            QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
            if (probe.state() == QProcess::NotRunning) {
                break;
            }
        }
        if (!started) {
            if (detail != nullptr) {
                *detail = probe.errorString();
            }
            return false;
        }

        QElapsedTimer finishTimer;
        finishTimer.start();
        while (probe.state() != QProcess::NotRunning && finishTimer.elapsed() < 3000) {
            probe.waitForFinished(25);
            QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
        }
        if (probe.state() != QProcess::NotRunning) {
            probe.kill();
            probe.waitForFinished(1000);
            if (detail != nullptr) {
                *detail = QStringLiteral("probe timed out");
            }
            return false;
        }
        if (probe.exitStatus() != QProcess::NormalExit || probe.exitCode() != 0) {
            if (detail != nullptr) {
                const QString stderrText = formatProbeOutput(probe.readAllStandardError());
                const QString stdoutText = formatProbeOutput(probe.readAllStandardOutput());
                if (!stderrText.isEmpty()) {
                    *detail = stderrText;
                } else if (!stdoutText.isEmpty()) {
                    *detail = stdoutText;
                } else {
                    *detail = QStringLiteral("exit code %1").arg(probe.exitCode());
                }
            }
            return false;
        }
        return true;
    };

    QString firstCandidate;
    for (const Candidate& candidate : candidates) {
        if (firstCandidate.isEmpty()) {
            firstCandidate = candidate.path;
        }

        QString detail;
        if (canStartPython(candidate, &detail)) {
            if (diagnostics != nullptr) {
                diagnostics->append(
                    QStringLiteral("Python backend: using %1: %2").arg(candidate.label, candidate.path));
                if (!candidate.pythonPathEntries.isEmpty()) {
                    diagnostics->append(
                        QStringLiteral("Python backend: adding PYTHONPATH %1")
                            .arg(candidate.pythonPathEntries.join(QDir::listSeparator())));
                }
            }
            if (pythonPathEntries != nullptr) {
                *pythonPathEntries = candidate.pythonPathEntries;
            }
            return candidate.path;
        }

        if (diagnostics != nullptr) {
            diagnostics->append(
                QStringLiteral("[WARN] Python candidate failed (%1): %2 - %3")
                    .arg(candidate.label, candidate.path, detail));
        }

        if (candidate.required && diagnostics != nullptr) {
            diagnostics->append(
                QStringLiteral("[WARN] Falling back because configured SPIKE_PYTHON could not start."));
        }
    }

    if (diagnostics != nullptr) {
        diagnostics->append(
            QStringLiteral("[WARN] No Python candidate passed the startup probe; attempting first candidate anyway."));
    }
    if (!firstCandidate.isEmpty()) {
        if (pythonPathEntries != nullptr) {
            pythonPathEntries->clear();
        }
        return firstCandidate;
    }
    if (pythonPathEntries != nullptr) {
        pythonPathEntries->clear();
    }
    return QStringLiteral("python");
}

QString MainWindow::repoRootPath() const
{
    QDir dir(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 8; ++depth) {
        const QFileInfo mainPy(dir.absoluteFilePath(QStringLiteral("main.py")));
        const QFileInfo imageProcessorDir(dir.absoluteFilePath(QStringLiteral("image_processor")));
        if (mainPy.isFile() && imageProcessorDir.isDir()) {
            return dir.absolutePath();
        }
        if (!dir.cdUp()) {
            break;
        }
    }

    QDir fallback(QCoreApplication::applicationDirPath());
    for (int depth = 0; depth < 3; ++depth) {
        if (!fallback.cdUp()) {
            break;
        }
    }
    return fallback.absolutePath();
}

QStringList MainWindow::scanFolder(const QString& folderPath) const
{
    return scanFolderPaths(folderPath, recursiveCheck_ != nullptr && recursiveCheck_->isChecked());
}

QStringList MainWindow::scanFolderPaths(
    const QString& folderPath,
    bool recursive,
    const std::function<bool()>& shouldCancel)
{
    QStringList paths;
    const QDirIterator::IteratorFlags flags = recursive
        ? QDirIterator::Subdirectories
        : QDirIterator::NoIteratorFlags;
    QDirIterator iterator(folderPath, QDir::Files, flags);
    while (iterator.hasNext()) {
        if (shouldCancel && shouldCancel()) {
            return {};
        }
        const QString path = iterator.next();
        const QString suffix = QFileInfo(path).suffix().toLower();
        if (suffix == QStringLiteral("tif") || suffix == QStringLiteral("tiff")) {
            paths.append(QFileInfo(path).absoluteFilePath());
        }
    }
    if (shouldCancel && shouldCancel()) {
        return {};
    }

    std::sort(paths.begin(), paths.end(), [](const QString& left, const QString& right) {
        return QString::compare(left, right, Qt::CaseInsensitive) < 0;
    });
    return paths;
}
