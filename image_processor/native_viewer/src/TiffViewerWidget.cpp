#include "TiffViewerWidget.h"

#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFont>
#include <QHBoxLayout>
#include <QImageReader>
#include <QLabel>
#include <QMessageBox>
#include <QMetaObject>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPoint>
#include <QPointer>
#include <QPushButton>
#include <QRunnable>
#include <QSlider>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTextCursor>
#include <QVBoxLayout>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

class ImageCanvas final : public QWidget {
public:
    explicit ImageCanvas(QWidget* parent = nullptr)
        : QWidget(parent)
    {
        setMinimumSize(640, 460);
        setAutoFillBackground(false);
    }

    void setImage(const QImage& image)
    {
        image_ = image;
        update();
    }

    void clear()
    {
        image_ = QImage();
        update();
    }

    void setRoiOverlay(const NativeRoiOverlay& overlay)
    {
        overlay_ = overlay;
        update();
    }

    void clearRoiOverlay()
    {
        overlay_ = NativeRoiOverlay();
        update();
    }

    void setRoiOverlayVisible(bool visible)
    {
        roiOverlayVisible_ = visible;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter painter(this);
        painter.fillRect(rect(), QColor(QStringLiteral("#15171a")));

        if (image_.isNull()) {
            painter.setPen(QColor(QStringLiteral("#aab2bd")));
            painter.drawText(rect(), Qt::AlignCenter, QStringLiteral("No image loaded"));
            return;
        }

        const QRect available = rect().adjusted(16, 16, -16, -16);
        const double scale = std::min(
            static_cast<double>(available.width()) / static_cast<double>(image_.width()),
            static_cast<double>(available.height()) / static_cast<double>(image_.height()));
        const int drawWidth = static_cast<int>(static_cast<double>(image_.width()) * scale);
        const int drawHeight = static_cast<int>(static_cast<double>(image_.height()) * scale);
        const int left = available.left() + (available.width() - drawWidth) / 2;
        const int top = available.top() + (available.height() - drawHeight) / 2;

        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
        const QRect target(left, top, drawWidth, drawHeight);
        painter.drawImage(target, image_);

        if (roiOverlayVisible_ && overlay_.isValid()) {
            const double scaleX = static_cast<double>(target.width()) / static_cast<double>(overlay_.width);
            const double scaleY = static_cast<double>(target.height()) / static_cast<double>(overlay_.height);

            QPen boxPen(QColor(QStringLiteral("#ffc857")));
            boxPen.setWidth(2);
            painter.setPen(boxPen);
            QFont labelFont = painter.font();
            labelFont.setBold(true);
            labelFont.setPointSize(9);
            painter.setFont(labelFont);

            for (const NativeRoiBox& box : overlay_.boxes) {
                const QRect roiRect(
                    target.left() + static_cast<int>(std::lround(static_cast<double>(box.left) * scaleX)),
                    target.top() + static_cast<int>(std::lround(static_cast<double>(box.upper) * scaleY)),
                    static_cast<int>(std::lround(static_cast<double>(box.right - box.left) * scaleX)),
                    static_cast<int>(std::lround(static_cast<double>(box.lower - box.upper) * scaleY)));
                painter.drawRect(roiRect);

                const QString label = QStringLiteral("%1").arg(box.roiIndex);
                QRect textRect = painter.fontMetrics().boundingRect(label).adjusted(-4, -2, 4, 2);
                textRect.moveTopLeft(roiRect.topLeft() + QPoint(4, 4));
                painter.fillRect(textRect, QColor(QStringLiteral("#15171a")));
                painter.drawText(textRect, Qt::AlignCenter, label);
            }
        }
    }

private:
    QImage image_;
    NativeRoiOverlay overlay_;
    bool roiOverlayVisible_ = true;
};

namespace {

enum class InfoLoadMode {
    Preview,
    Full,
};

class InfoLoadTask final : public QRunnable {
public:
    InfoLoadTask(
        TiffViewerWidget* target,
        QString path,
        quint64 generation,
        InfoLoadMode mode,
        std::shared_ptr<std::atomic_bool> cancelFlag)
        : target_(target)
        , path_(std::move(path))
        , generation_(generation)
        , mode_(mode)
        , cancelFlag_(std::move(cancelFlag))
    {
    }

    void run() override
    {
        TiffStackInfo info;
        QString error;
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };
        const bool previewOnly = mode_ == InfoLoadMode::Preview;
        const bool ok = previewOnly
            ? TiffStack::readPreviewInfo(path_, &info, &error, shouldCancel)
            : TiffStack::readInfo(path_, &info, &error, shouldCancel);
        const QPointer<TiffViewerWidget> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target, generation = generation_, path = path_, previewOnly, ok, info, error]() {
                if (target != nullptr) {
                    target->completeInfoLoad(generation, path, previewOnly, ok, info, error);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<TiffViewerWidget> target_;
    QString path_;
    quint64 generation_;
    InfoLoadMode mode_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
};

class FrameLoadTask final : public QRunnable {
public:
    FrameLoadTask(
        TiffViewerWidget* target,
        std::shared_ptr<const TiffStackInfo> stackInfo,
        std::shared_ptr<std::atomic_bool> cancelFlag,
        QString path,
        quint64 generation,
        quint64 requestId,
        int frameIndex)
        : target_(target)
        , stackInfo_(std::move(stackInfo))
        , cancelFlag_(std::move(cancelFlag))
        , path_(std::move(path))
        , generation_(generation)
        , requestId_(requestId)
        , frameIndex_(frameIndex)
    {
    }

    void run() override
    {
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };
        auto result = std::make_shared<TiffFrameResult>(
            stackInfo_ != nullptr
                ? TiffStack::readFrame(*stackInfo_, frameIndex_, shouldCancel)
                : TiffStack::readFrame(path_, frameIndex_, shouldCancel));
        const QPointer<TiffViewerWidget> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target,
             generation = generation_,
             requestId = requestId_,
             path = path_,
             frameIndex = frameIndex_,
             result = std::move(result)]() {
                if (target != nullptr) {
                    target->completeFrameLoad(generation, requestId, path, frameIndex, result);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<TiffViewerWidget> target_;
    std::shared_ptr<const TiffStackInfo> stackInfo_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
    QString path_;
    quint64 generation_;
    quint64 requestId_;
    int frameIndex_;
};

class PrefetchLoadTask final : public QRunnable {
public:
    PrefetchLoadTask(
        TiffViewerWidget* target,
        std::shared_ptr<const TiffStackInfo> stackInfo,
        std::shared_ptr<std::atomic_bool> cancelFlag,
        QString path,
        quint64 generation,
        quint64 prefetchGeneration,
        int frameIndex)
        : target_(target)
        , stackInfo_(std::move(stackInfo))
        , cancelFlag_(std::move(cancelFlag))
        , path_(std::move(path))
        , generation_(generation)
        , prefetchGeneration_(prefetchGeneration)
        , frameIndex_(frameIndex)
    {
    }

    void run() override
    {
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };
        auto result = std::make_shared<TiffFrameResult>(
            stackInfo_ != nullptr
                ? TiffStack::readFrame(*stackInfo_, frameIndex_, shouldCancel)
                : TiffStack::readFrame(path_, frameIndex_, shouldCancel));
        const QPointer<TiffViewerWidget> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target,
             generation = generation_,
             prefetchGeneration = prefetchGeneration_,
             path = path_,
             frameIndex = frameIndex_,
             result = std::move(result)]() {
                if (target != nullptr) {
                    target->completePrefetchLoad(generation, prefetchGeneration, path, frameIndex, result);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<TiffViewerWidget> target_;
    std::shared_ptr<const TiffStackInfo> stackInfo_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
    QString path_;
    quint64 generation_;
    quint64 prefetchGeneration_;
    int frameIndex_;
};

uint8_t scaleToByte(double value, double black, double scale)
{
    const int scaled = static_cast<int>((value - black) * scale + 0.5);
    return static_cast<uint8_t>(std::clamp(scaled, 0, 255));
}

std::array<uint8_t, 256> buildUint8Lookup(double black, double white)
{
    std::array<uint8_t, 256> lookup {};
    if (white <= black) {
        lookup.fill(0);
        return lookup;
    }

    const double scale = 255.0 / (white - black);
    for (int value = 0; value < static_cast<int>(lookup.size()); ++value) {
        lookup[static_cast<size_t>(value)] = scaleToByte(static_cast<double>(value), black, scale);
    }
    return lookup;
}

std::vector<uint8_t> buildUint16Lookup(double black, double white)
{
    std::vector<uint8_t> lookup(65536);
    if (white <= black) {
        return lookup;
    }

    const double scale = 255.0 / (white - black);
    for (int value = 0; value <= 65535; ++value) {
        lookup[static_cast<size_t>(value)] = scaleToByte(static_cast<double>(value), black, scale);
    }
    return lookup;
}

QImage renderCachedFrameImage(
    const CachedFrame& frame,
    double black,
    double white,
    const std::function<bool()>& shouldCancel = {})
{
    QImage image(frame.width, frame.height, QImage::Format_Grayscale8);
    if (image.isNull() || !frame.hasSamples()) {
        return image;
    }

    if (frame.samples8 != nullptr) {
        const auto lookup = buildUint8Lookup(black, white);
        const auto& samples = *frame.samples8;
        for (int row = 0; row < frame.height; ++row) {
            if (shouldCancel && shouldCancel()) {
                return QImage();
            }
            const auto* source =
                samples.data() + static_cast<size_t>(row) * static_cast<size_t>(frame.width);
            uint8_t* destination = image.scanLine(row);
            for (int column = 0; column < frame.width; ++column) {
                destination[column] = lookup[source[column]];
            }
        }
        return image;
    }

    if (frame.samples16 != nullptr) {
        const auto lookup = buildUint16Lookup(black, white);
        const auto& samples = *frame.samples16;
        for (int row = 0; row < frame.height; ++row) {
            if (shouldCancel && shouldCancel()) {
                return QImage();
            }
            const auto* source =
                samples.data() + static_cast<size_t>(row) * static_cast<size_t>(frame.width);
            uint8_t* destination = image.scanLine(row);
            for (int column = 0; column < frame.width; ++column) {
                destination[column] = lookup[source[column]];
            }
        }
        return image;
    }

    const double scale = white > black ? 255.0 / (white - black) : 0.0;
    for (int row = 0; row < frame.height; ++row) {
        if (shouldCancel && shouldCancel()) {
            return QImage();
        }
        uint8_t* destination = image.scanLine(row);
        for (int column = 0; column < frame.width; ++column) {
            const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(frame.width)
                + static_cast<size_t>(column);
            double value = black;
            if (frame.samplesFloat != nullptr) {
                value = static_cast<double>((*frame.samplesFloat)[offset]);
                if (!std::isfinite(value)) {
                    value = black;
                }
            }
            const int scaled = white > black
                ? static_cast<int>((value - black) * scale + 0.5)
                : 0;
            destination[column] = static_cast<uint8_t>(std::clamp(scaled, 0, 255));
        }
    }
    return image;
}

class FrameRenderTask final : public QRunnable {
public:
    FrameRenderTask(
        TiffViewerWidget* target,
        CachedFrame frame,
        QString path,
        quint64 generation,
        quint64 renderRequestId,
        int frameIndex,
        double black,
        double white,
        std::shared_ptr<std::atomic_bool> cancelFlag)
        : target_(target)
        , frame_(std::move(frame))
        , path_(std::move(path))
        , generation_(generation)
        , renderRequestId_(renderRequestId)
        , frameIndex_(frameIndex)
        , black_(black)
        , white_(white)
        , cancelFlag_(std::move(cancelFlag))
    {
    }

    void run() override
    {
        const auto shouldCancel = [cancelFlag = cancelFlag_]() {
            return cancelFlag != nullptr && cancelFlag->load();
        };
        QElapsedTimer timer;
        timer.start();
        const QImage image = renderCachedFrameImage(frame_, black_, white_, shouldCancel);
        const qint64 renderElapsedMs = timer.elapsed();
        const bool cancelled = shouldCancel() || image.isNull();
        const QPointer<TiffViewerWidget> target = target_;
        if (target == nullptr) {
            return;
        }

        QMetaObject::invokeMethod(
            target.data(),
            [target,
             generation = generation_,
             renderRequestId = renderRequestId_,
             path = path_,
             frameIndex = frameIndex_,
             image,
             renderElapsedMs,
             cancelled]() {
                if (target != nullptr) {
                    target->completeFrameRender(
                        generation,
                        renderRequestId,
                        path,
                        frameIndex,
                        image,
                        renderElapsedMs,
                        cancelled);
                }
            },
            Qt::QueuedConnection);
    }

private:
    QPointer<TiffViewerWidget> target_;
    CachedFrame frame_;
    QString path_;
    quint64 generation_;
    quint64 renderRequestId_;
    int frameIndex_;
    double black_;
    double white_;
    std::shared_ptr<std::atomic_bool> cancelFlag_;
};

CachedFrame cachedFrameFromResult(TiffFrameResult& result, const QString& source)
{
    CachedFrame frame;
    frame.width = result.width;
    frame.height = result.height;
    frame.bitsPerSample = result.bitsPerSample;
    frame.sampleFormat = result.sampleFormat;
    frame.observedMin = result.observedMin;
    frame.observedMax = result.observedMax;
    if (!result.samples8.empty()) {
        frame.samples8 = std::make_shared<const std::vector<uint8_t>>(std::move(result.samples8));
    }
    if (!result.samples16.empty()) {
        frame.samples16 = std::make_shared<const std::vector<uint16_t>>(std::move(result.samples16));
    }
    if (!result.samplesFloat.empty()) {
        frame.samplesFloat = std::make_shared<const std::vector<float>>(std::move(result.samplesFloat));
    }
    frame.elapsedMs = result.elapsedMs;
    frame.source = source;
    return frame;
}

constexpr qint64 TextPreviewMaxBytes = 2 * 1024 * 1024;
constexpr int SampleFormatUint = 1;
constexpr int SampleFormatInt = 2;
constexpr int SampleFormatIeeeFloat = 3;

struct DisplayLevels {
    double minimum = 0.0;
    double maximum = 65535.0;
    double black = 0.0;
    double white = 65535.0;
    double step = 100.0;
    int decimals = 0;
};

double finiteLevel(double value, double fallback)
{
    return std::isfinite(value) ? value : fallback;
}

void keepLevelsOrdered(DisplayLevels* levels)
{
    if (levels == nullptr) {
        return;
    }
    if (!(levels->maximum > levels->minimum)) {
        levels->minimum -= 0.5;
        levels->maximum += 0.5;
    }
    levels->black = std::clamp(levels->black, levels->minimum, levels->maximum);
    levels->white = std::clamp(levels->white, levels->minimum, levels->maximum);
    if (levels->white <= levels->black) {
        const double delta = std::max(levels->step, 1e-6);
        levels->white = std::min(levels->maximum, levels->black + delta);
        if (levels->white <= levels->black) {
            levels->black = std::max(levels->minimum, levels->white - delta);
        }
    }
}

DisplayLevels displayLevelsForFrame(const CachedFrame& frame, bool fullRange)
{
    DisplayLevels levels;

    if (frame.sampleFormat == SampleFormatUint && frame.bitsPerSample == 8) {
        levels.maximum = 255.0;
        levels.step = 1.0;
        levels.black = fullRange ? levels.minimum : std::clamp(finiteLevel(frame.observedMin, 0.0), 0.0, 255.0);
        levels.white = fullRange ? levels.maximum : std::clamp(finiteLevel(frame.observedMax, 255.0), 0.0, 255.0);
    } else if (frame.sampleFormat == SampleFormatUint && frame.bitsPerSample == 16) {
        levels.maximum = 65535.0;
        levels.step = 100.0;
        levels.black = fullRange ? levels.minimum : std::clamp(finiteLevel(frame.observedMin, 0.0), 0.0, 65535.0);
        levels.white =
            fullRange ? levels.maximum : std::clamp(finiteLevel(frame.observedMax, 65535.0), 0.0, 65535.0);
    } else if (frame.sampleFormat == SampleFormatInt && frame.bitsPerSample == 16) {
        levels.minimum = -32768.0;
        levels.maximum = 32767.0;
        levels.step = 100.0;
        levels.black =
            fullRange ? levels.minimum : std::clamp(finiteLevel(frame.observedMin, -32768.0), levels.minimum, levels.maximum);
        levels.white =
            fullRange ? levels.maximum : std::clamp(finiteLevel(frame.observedMax, 32767.0), levels.minimum, levels.maximum);
    } else if (frame.sampleFormat == SampleFormatIeeeFloat && frame.bitsPerSample == 32) {
        double observedMin = finiteLevel(frame.observedMin, 0.0);
        double observedMax = finiteLevel(frame.observedMax, observedMin);
        if (!(observedMax > observedMin)) {
            observedMin -= 0.5;
            observedMax += 0.5;
        }
        const double span = observedMax - observedMin;
        levels.minimum = observedMin - span;
        levels.maximum = observedMax + span;
        levels.black = observedMin;
        levels.white = observedMax;
        levels.step = std::max(span / 100.0, 0.001);
        levels.decimals = 6;
    }

    keepLevelsOrdered(&levels);
    return levels;
}

} // namespace

TiffViewerWidget::TiffViewerWidget(QWidget* parent)
    : QWidget(parent)
    , frameCache_(8)
{
    previewInfoPool_.setMaxThreadCount(1);
    fullInfoPool_.setMaxThreadCount(1);
    framePool_.setMaxThreadCount(2);
    prefetchPool_.setMaxThreadCount(1);
    renderPool_.setMaxThreadCount(1);

    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(10, 10, 10, 10);
    rootLayout->setSpacing(8);

    auto* topLayout = new QHBoxLayout();
    openButton_ = new QPushButton(QStringLiteral("Open TIFF"));
    roiOverlayCheck_ = new QCheckBox(QStringLiteral("ROI"));
    roiOverlayCheck_->setChecked(true);
    roiOverlayCheck_->setEnabled(false);
    pathLabel_ = new QLabel(QStringLiteral("No file loaded"));
    pathLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    topLayout->addWidget(openButton_);
    topLayout->addWidget(roiOverlayCheck_);
    topLayout->addWidget(pathLabel_, 1);
    rootLayout->addLayout(topLayout);

    metadataLabel_ = new QLabel(QStringLiteral("Metadata: none"));
    metadataLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    rootLayout->addWidget(metadataLabel_);

    contentStack_ = new QStackedWidget(this);
    canvas_ = new ImageCanvas(contentStack_);
    textPreview_ = new QPlainTextEdit(contentStack_);
    textPreview_->setReadOnly(true);
    textPreview_->setLineWrapMode(QPlainTextEdit::NoWrap);
    textPreview_->setFont(QFont(QStringLiteral("Consolas"), 10));
    contentStack_->addWidget(canvas_);
    contentStack_->addWidget(textPreview_);
    rootLayout->addWidget(contentStack_, 1);

    auto* frameLayout = new QHBoxLayout();
    frameSlider_ = new QSlider(Qt::Horizontal);
    frameSlider_->setTracking(true);
    frameSlider_->setEnabled(false);
    frameSpin_ = new QSpinBox();
    frameSpin_->setRange(1, 1);
    frameSpin_->setEnabled(false);
    frameTotalLabel_ = new QLabel(QStringLiteral("of 0"));
    frameLayout->addWidget(new QLabel(QStringLiteral("Frame")));
    frameLayout->addWidget(frameSlider_, 1);
    frameLayout->addWidget(frameSpin_);
    frameLayout->addWidget(frameTotalLabel_);
    rootLayout->addLayout(frameLayout);

    auto* displayLayout = new QHBoxLayout();
    displayBlackSpin_ = new QDoubleSpinBox();
    displayWhiteSpin_ = new QDoubleSpinBox();
    autoDisplayButton_ = new QPushButton(QStringLiteral("Auto"));
    fullDisplayButton_ = new QPushButton(QStringLiteral("Full"));
    for (QDoubleSpinBox* spin : {displayBlackSpin_, displayWhiteSpin_}) {
        spin->setDecimals(0);
        spin->setRange(0, 65535);
        spin->setSingleStep(100);
        spin->setEnabled(false);
    }
    autoDisplayButton_->setEnabled(false);
    fullDisplayButton_->setEnabled(false);
    displayLayout->addWidget(new QLabel(QStringLiteral("Black")));
    displayLayout->addWidget(displayBlackSpin_);
    displayLayout->addWidget(new QLabel(QStringLiteral("White")));
    displayLayout->addWidget(displayWhiteSpin_);
    displayLayout->addWidget(autoDisplayButton_);
    displayLayout->addWidget(fullDisplayButton_);
    rootLayout->addLayout(displayLayout);

    statusLabel_ = new QLabel(QStringLiteral("Ready"));
    statusLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    rootLayout->addWidget(statusLabel_);

    connect(openButton_, &QPushButton::clicked, this, &TiffViewerWidget::openFileDialog);
    connect(roiOverlayCheck_, &QCheckBox::toggled, this, [this](bool checked) {
        canvas_->setRoiOverlayVisible(checked);
    });
    connect(frameSlider_, &QSlider::valueChanged, this, &TiffViewerWidget::requestFrameFromSlider);
    connect(frameSpin_, &QSpinBox::valueChanged, this, &TiffViewerWidget::requestFrameFromSpin);
    connect(displayBlackSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        if (!updatingDisplayControls_) {
            refreshDisplayedFrame();
        }
    });
    connect(displayWhiteSpin_, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        if (!updatingDisplayControls_) {
            refreshDisplayedFrame();
        }
    });
    connect(autoDisplayButton_, &QPushButton::clicked, this, [this]() {
        if (hasDisplayedFrame_) {
            setDisplayLevelsFromFrame(displayedFrame_);
            refreshDisplayedFrame();
        }
    });
    connect(fullDisplayButton_, &QPushButton::clicked, this, [this]() {
        setDisplayFullRange();
        refreshDisplayedFrame();
    });
}

void TiffViewerWidget::loadFile(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }

    cancelPendingInfoLoads();
    cancelActiveFrameLoad();
    cancelPendingPrefetchLoads();
    cancelActiveRender();
    resetForOpen(path);
    contentStack_->setCurrentWidget(canvas_);
    infoCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    previewInfoPool_.start(new InfoLoadTask(this, path, generation_, InfoLoadMode::Preview, infoCancelFlag_));
}

void TiffViewerWidget::loadImageFile(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }

    cancelPendingInfoLoads();
    cancelActiveFrameLoad();
    cancelPendingPrefetchLoads();
    cancelActiveRender();
    ++generation_;
    ++requestId_;
    filePath_ = path;
    hasInfo_ = false;
    info_ = TiffStackInfo();
    stackInfo_.reset();
    roiOverlay_ = NativeRoiOverlay();
    displayedFrame_ = CachedFrame();
    frameCache_.clear();
    prefetchInFlight_.clear();
    hasDisplayedFrame_ = false;
    displayLevelsInitialized_ = false;
    currentFrameIndex_ = 0;
    pendingFrameIndex_ = -1;
    lastFrameStep_ = 1;
    pendingFrameRequestId_ = 0;
    frameWorkerActive_ = false;
    frameWorkerRequestId_ = 0;

    pathLabel_->setText(QFileInfo(path).fileName());
    pathLabel_->setToolTip(path);
    contentStack_->setCurrentWidget(canvas_);
    canvas_->clearRoiOverlay();
    roiOverlayCheck_->setEnabled(false);
    metadataLabel_->setToolTip(QString());

    updatingDisplayControls_ = true;
    displayBlackSpin_->setEnabled(false);
    displayWhiteSpin_->setEnabled(false);
    autoDisplayButton_->setEnabled(false);
    fullDisplayButton_->setEnabled(false);
    displayBlackSpin_->setRange(0, 65535);
    displayWhiteSpin_->setRange(0, 65535);
    displayBlackSpin_->setValue(0);
    displayWhiteSpin_->setValue(65535);
    updatingDisplayControls_ = false;
    configureFrameControls(0, 0);

    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage image = reader.read();
    if (image.isNull()) {
        canvas_->clear();
        metadataLabel_->setText(QStringLiteral("Image: unavailable"));
        setStatus(QStringLiteral("Open image failed: %1").arg(reader.errorString()));
        QMessageBox::warning(this, QStringLiteral("Open image"), reader.errorString());
        return;
    }

    canvas_->setImage(image);
    metadataLabel_->setText(QStringLiteral("Image: %1 x %2").arg(image.width()).arg(image.height()));
    setStatus(QStringLiteral("Image result loaded: %1 x %2").arg(image.width()).arg(image.height()));
}

void TiffViewerWidget::loadTextFile(const QString& path)
{
    if (path.isEmpty()) {
        return;
    }

    cancelPendingInfoLoads();
    cancelActiveFrameLoad();
    cancelPendingPrefetchLoads();
    cancelActiveRender();
    ++generation_;
    ++requestId_;
    filePath_ = path;
    hasInfo_ = false;
    info_ = TiffStackInfo();
    stackInfo_.reset();
    roiOverlay_ = NativeRoiOverlay();
    displayedFrame_ = CachedFrame();
    frameCache_.clear();
    prefetchInFlight_.clear();
    hasDisplayedFrame_ = false;
    displayLevelsInitialized_ = false;
    currentFrameIndex_ = 0;
    pendingFrameIndex_ = -1;
    lastFrameStep_ = 1;
    pendingFrameRequestId_ = 0;
    frameWorkerActive_ = false;
    frameWorkerRequestId_ = 0;

    pathLabel_->setText(QFileInfo(path).fileName());
    pathLabel_->setToolTip(path);
    contentStack_->setCurrentWidget(textPreview_);
    canvas_->clear();
    canvas_->clearRoiOverlay();
    roiOverlayCheck_->setEnabled(false);

    updatingDisplayControls_ = true;
    displayBlackSpin_->setEnabled(false);
    displayWhiteSpin_->setEnabled(false);
    autoDisplayButton_->setEnabled(false);
    fullDisplayButton_->setEnabled(false);
    displayBlackSpin_->setRange(0, 65535);
    displayWhiteSpin_->setRange(0, 65535);
    displayBlackSpin_->setValue(0);
    displayWhiteSpin_->setValue(65535);
    updatingDisplayControls_ = false;
    configureFrameControls(0, 0);

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        textPreview_->clear();
        metadataLabel_->setText(QStringLiteral("Text: unavailable"));
        setStatus(QStringLiteral("Open text failed: %1").arg(file.errorString()));
        QMessageBox::warning(this, QStringLiteral("Open text"), file.errorString());
        return;
    }

    const QByteArray bytes = file.read(TextPreviewMaxBytes + 1);
    const bool truncated = bytes.size() > TextPreviewMaxBytes;
    QByteArray visibleBytes = bytes;
    if (truncated) {
        visibleBytes.truncate(TextPreviewMaxBytes);
    }

    QString text = QString::fromUtf8(visibleBytes);
    if (truncated) {
        text.append(QStringLiteral("\n\n[Preview truncated at %1 bytes]").arg(TextPreviewMaxBytes));
    }
    textPreview_->setPlainText(text);
    textPreview_->moveCursor(QTextCursor::Start);
    metadataLabel_->setText(
        QStringLiteral("Text: %1 bytes%2")
            .arg(QFileInfo(path).size())
            .arg(truncated ? QStringLiteral(" (preview truncated)") : QString()));
    setStatus(QStringLiteral("Text result loaded"));
}

void TiffViewerWidget::clearView()
{
    cancelPendingInfoLoads();
    cancelActiveFrameLoad();
    cancelPendingPrefetchLoads();
    cancelActiveRender();
    ++generation_;
    ++requestId_;
    filePath_.clear();
    hasInfo_ = false;
    info_ = TiffStackInfo();
    stackInfo_.reset();
    roiOverlay_ = NativeRoiOverlay();
    displayedFrame_ = CachedFrame();
    frameCache_.clear();
    prefetchInFlight_.clear();
    hasDisplayedFrame_ = false;
    displayLevelsInitialized_ = false;
    currentFrameIndex_ = 0;
    pendingFrameIndex_ = -1;
    lastFrameStep_ = 1;
    pendingFrameRequestId_ = 0;
    frameWorkerActive_ = false;
    frameWorkerRequestId_ = 0;

    pathLabel_->setText(QStringLiteral("No file loaded"));
    pathLabel_->setToolTip(QString());
    contentStack_->setCurrentWidget(canvas_);
    canvas_->clear();
    textPreview_->clear();
    canvas_->clearRoiOverlay();
    roiOverlayCheck_->setEnabled(false);
    metadataLabel_->setText(QStringLiteral("Metadata: none"));
    metadataLabel_->setToolTip(QString());

    updatingDisplayControls_ = true;
    displayBlackSpin_->setEnabled(false);
    displayWhiteSpin_->setEnabled(false);
    autoDisplayButton_->setEnabled(false);
    fullDisplayButton_->setEnabled(false);
    displayBlackSpin_->setRange(0, 65535);
    displayWhiteSpin_->setRange(0, 65535);
    displayBlackSpin_->setValue(0);
    displayWhiteSpin_->setValue(65535);
    updatingDisplayControls_ = false;
    configureFrameControls(0, 0);
    setStatus(QStringLiteral("Ready"));
}

void TiffViewerWidget::setCurrentFrameIndex(int frameIndex)
{
    requestFrame(frameIndex);
}

QString TiffViewerWidget::currentFilePath() const
{
    return filePath_;
}

int TiffViewerWidget::currentFrameIndex() const
{
    return currentFrameIndex_;
}

QString TiffViewerWidget::statusText() const
{
    return statusLabel_ == nullptr ? QString() : statusLabel_->text();
}

bool TiffViewerWidget::hasDisplayedFrame() const
{
    return hasDisplayedFrame_;
}

void TiffViewerWidget::cancelPendingInfoLoads()
{
    if (infoCancelFlag_ != nullptr) {
        infoCancelFlag_->store(true);
        infoCancelFlag_.reset();
    }
    previewInfoPool_.clear();
    fullInfoPool_.clear();
}

void TiffViewerWidget::cancelActiveFrameLoad()
{
    if (frameCancelFlag_ != nullptr) {
        frameCancelFlag_->store(true);
        frameCancelFlag_.reset();
    }
    frameWorkerActive_ = false;
    frameWorkerRequestId_ = 0;
    framePool_.clear();
}

void TiffViewerWidget::cancelPendingPrefetchLoads()
{
    ++prefetchGeneration_;
    if (prefetchCancelFlag_ != nullptr) {
        prefetchCancelFlag_->store(true);
        prefetchCancelFlag_.reset();
    }
    prefetchPool_.clear();
    prefetchInFlight_.clear();
}

void TiffViewerWidget::cancelActiveRender()
{
    if (renderCancelFlag_ != nullptr) {
        renderCancelFlag_->store(true);
        renderCancelFlag_.reset();
    }
    renderPool_.clear();
}

void TiffViewerWidget::completeInfoLoad(
    quint64 generation,
    const QString& path,
    bool previewOnly,
    bool ok,
    const TiffStackInfo& info,
    const QString& error)
{
    if (generation != generation_ || path != filePath_) {
        return;
    }

    if (previewOnly) {
        if (!ok) {
            return;
        }
        if (hasInfo_ && info_.indexComplete) {
            return;
        }

        hasInfo_ = true;
        info_ = info;
        stackInfo_ = std::make_shared<TiffStackInfo>(info_);
        configureFrameControls(info_.frameCount, 0);
        setStatus(QStringLiteral("Preview ready: %1 x %2, %3; indexing stack...")
                      .arg(info_.width)
                      .arg(info_.height)
                      .arg(info_.pixelType()));
        if (!hasDisplayedFrame_) {
            requestFrame(0);
        }
        startFullInfoLoad();
        return;
    }

    infoCancelFlag_.reset();

    if (!ok) {
        hasInfo_ = false;
        configureFrameControls(0, 0);
        canvas_->clear();
        setStatus(QStringLiteral("Open failed: %1").arg(error));
        QMessageBox::critical(this, QStringLiteral("Open TIFF"), error);
        return;
    }

    const bool hadDisplayedFrame = hasDisplayedFrame_;
    const int frameIndex = currentFrameIndex_;
    hasInfo_ = true;
    info_ = info;
    stackInfo_ = std::make_shared<TiffStackInfo>(info_);
    loadRoiOverlay(path);
    configureFrameControls(info_.frameCount, std::clamp(frameIndex, 0, std::max(0, info_.frameCount - 1)));
    emit fileLoaded(path);

    setStatus(QStringLiteral("%1 frames, %2 x %3, %4, %5, %6, %7 %8 IFDs in %9 ms")
                  .arg(info_.frameCount)
                  .arg(info_.width)
                  .arg(info_.height)
                  .arg(info_.pixelType())
                  .arg(info_.bigTiff ? QStringLiteral("BigTIFF") : QStringLiteral("classic TIFF"))
                  .arg(info_.tiled ? QStringLiteral("tiled") : QStringLiteral("strips"))
                  .arg(info_.fromCache ? QStringLiteral("cached") : QStringLiteral("indexed"))
                  .arg(static_cast<int>(info_.directoryOffsets.size()))
                  .arg(info_.elapsedMs));
    if (hadDisplayedFrame) {
        prefetchNeighbors(currentFrameIndex_);
    } else {
        requestFrame(0);
    }
}

void TiffViewerWidget::startFullInfoLoad()
{
    if (infoCancelFlag_ == nullptr || filePath_.isEmpty()) {
        return;
    }
    fullInfoPool_.clear();
    fullInfoPool_.start(new InfoLoadTask(this, filePath_, generation_, InfoLoadMode::Full, infoCancelFlag_));
}

void TiffViewerWidget::completeFrameLoad(
    quint64 generation,
    quint64 requestId,
    const QString& path,
    int frameIndex,
    std::shared_ptr<TiffFrameResult> result)
{
    if (requestId == frameWorkerRequestId_) {
        frameWorkerActive_ = false;
        frameWorkerRequestId_ = 0;
        frameCancelFlag_.reset();
    }

    if (generation == generation_ && requestId == requestId_ && path == filePath_ && result != nullptr) {
        if (result->ok) {
            CachedFrame frame = cachedFrameFromResult(
                *result,
                result->usedDirectoryOffset ? QStringLiteral("libtiff-indexed") : QStringLiteral("libtiff"));
            frameCache_.put(frameIndex, frame);
            displayFrame(frameIndex, frame);
            prefetchNeighbors(frameIndex);
        } else if (!result->cancelled) {
            setStatus(QStringLiteral("Frame load failed: %1").arg(result->error));
            QMessageBox::warning(this, QStringLiteral("Load frame"), result->error);
        }
    }

    if (pendingFrameIndex_ >= 0) {
        startPendingFrameWorker();
    }
}

void TiffViewerWidget::completeFrameRender(
    quint64 generation,
    quint64 renderRequestId,
    const QString& path,
    int frameIndex,
    const QImage& image,
    qint64 renderElapsedMs,
    bool cancelled)
{
    if (renderRequestId == renderRequestId_) {
        renderCancelFlag_.reset();
    }
    if (cancelled || generation != generation_ || renderRequestId != renderRequestId_ || path != filePath_) {
        return;
    }
    if (frameIndex != currentFrameIndex_ || image.isNull()) {
        return;
    }

    canvas_->setImage(image);
    setStatus(QStringLiteral("Frame %1/%2: %3 x %4, %5, read %6 ms, render %7 ms")
                  .arg(frameIndex + 1)
                  .arg(info_.frameCount)
                  .arg(displayedFrame_.width)
                  .arg(displayedFrame_.height)
                  .arg(displayedFrame_.source)
                  .arg(displayedFrame_.elapsedMs)
                  .arg(renderElapsedMs));
    emit frameRendered(path, frameIndex);
}

void TiffViewerWidget::completePrefetchLoad(
    quint64 generation,
    quint64 prefetchGeneration,
    const QString& path,
    int frameIndex,
    std::shared_ptr<TiffFrameResult> result)
{
    if (generation != generation_ || prefetchGeneration != prefetchGeneration_ || path != filePath_) {
        return;
    }
    prefetchInFlight_.remove(frameIndex);
    if (result == nullptr || !result->ok || !result->hasSamples() || frameCache_.contains(frameIndex)) {
        return;
    }

    CachedFrame frame = cachedFrameFromResult(
        *result,
        result->usedDirectoryOffset ? QStringLiteral("libtiff-indexed-prefetch") : QStringLiteral("libtiff-prefetch"));
    frameCache_.put(frameIndex, frame);
}

void TiffViewerWidget::openFileDialog()
{
    const QString path = QFileDialog::getOpenFileName(
        this,
        QStringLiteral("Open TIFF stack"),
        QString(),
        QStringLiteral("TIFF files (*.tif *.tiff);;All files (*.*)"));
    if (!path.isEmpty()) {
        loadFile(path);
    }
}

void TiffViewerWidget::requestFrameFromSlider(int value)
{
    if (updatingControls_) {
        return;
    }
    requestFrame(value);
}

void TiffViewerWidget::requestFrameFromSpin(int value)
{
    if (updatingControls_) {
        return;
    }
    requestFrame(value - 1);
}

void TiffViewerWidget::resetForOpen(const QString& path)
{
    ++generation_;
    ++requestId_;
    filePath_ = path;
    hasInfo_ = false;
    info_ = TiffStackInfo();
    stackInfo_.reset();
    roiOverlay_ = NativeRoiOverlay();
    displayedFrame_ = CachedFrame();
    frameCache_.clear();
    prefetchInFlight_.clear();
    hasDisplayedFrame_ = false;
    displayLevelsInitialized_ = false;
    currentFrameIndex_ = 0;
    pendingFrameIndex_ = -1;
    lastFrameStep_ = 1;
    pendingFrameRequestId_ = 0;
    frameWorkerActive_ = false;
    frameWorkerRequestId_ = 0;
    pathLabel_->setText(QFileInfo(path).fileName());
    pathLabel_->setToolTip(path);
    contentStack_->setCurrentWidget(canvas_);
    canvas_->clear();
    textPreview_->clear();
    canvas_->clearRoiOverlay();
    roiOverlayCheck_->setEnabled(false);
    metadataLabel_->setText(QStringLiteral("Metadata: loading..."));
    metadataLabel_->setToolTip(QString());
    updatingDisplayControls_ = true;
    displayBlackSpin_->setEnabled(false);
    displayWhiteSpin_->setEnabled(false);
    autoDisplayButton_->setEnabled(false);
    fullDisplayButton_->setEnabled(false);
    displayBlackSpin_->setRange(0, 65535);
    displayWhiteSpin_->setRange(0, 65535);
    displayBlackSpin_->setValue(0);
    displayWhiteSpin_->setValue(65535);
    updatingDisplayControls_ = false;
    configureFrameControls(0, 0);
    setStatus(QStringLiteral("Opening %1...").arg(QFileInfo(path).fileName()));
}

void TiffViewerWidget::configureFrameControls(int frameCount, int frameIndex)
{
    frameCount = std::max(0, frameCount);
    frameIndex = std::clamp(frameIndex, 0, std::max(0, frameCount - 1));
    const bool enabled = frameCount > 0;

    updatingControls_ = true;
    frameSlider_->setEnabled(enabled);
    frameSpin_->setEnabled(enabled);
    frameSlider_->setRange(0, std::max(0, frameCount - 1));
    frameSlider_->setPageStep(enabled ? std::max(1, frameCount / 20) : 1);
    frameSlider_->setValue(frameIndex);
    frameSpin_->setRange(1, std::max(1, frameCount));
    frameSpin_->setValue(enabled ? frameIndex + 1 : 1);
    frameTotalLabel_->setText(QStringLiteral("of %1").arg(frameCount));
    updatingControls_ = false;
}

void TiffViewerWidget::syncFrameControls()
{
    updatingControls_ = true;
    frameSlider_->setValue(currentFrameIndex_);
    frameSpin_->setValue(currentFrameIndex_ + 1);
    updatingControls_ = false;
}

void TiffViewerWidget::requestFrame(int frameIndex)
{
    if (!hasInfo_ || info_.frameCount <= 0) {
        return;
    }

    frameIndex = std::clamp(frameIndex, 0, info_.frameCount - 1);
    if (frameIndex != currentFrameIndex_) {
        lastFrameStep_ = frameIndex > currentFrameIndex_ ? 1 : -1;
        cancelPendingPrefetchLoads();
    }
    currentFrameIndex_ = frameIndex;
    syncFrameControls();
    ++requestId_;
    const quint64 requestId = requestId_;

    setStatus(QStringLiteral("Loading frame %1/%2...").arg(frameIndex + 1).arg(info_.frameCount));

    if (frameWorkerActive_) {
        cancelActiveFrameLoad();
    }

    const auto cached = frameCache_.get(frameIndex);
    if (cached.has_value()) {
        pendingFrameIndex_ = -1;
        pendingFrameRequestId_ = 0;
        displayFrame(frameIndex, cached.value());
        prefetchNeighbors(frameIndex);
        return;
    }

    pendingFrameIndex_ = frameIndex;
    pendingFrameRequestId_ = requestId;
    if (!frameWorkerActive_) {
        startPendingFrameWorker();
    }
}

void TiffViewerWidget::startPendingFrameWorker()
{
    if (!hasInfo_ || pendingFrameIndex_ < 0) {
        return;
    }
    if (frameWorkerActive_) {
        return;
    }

    const int frameIndex = pendingFrameIndex_;
    const quint64 requestId = pendingFrameRequestId_;
    pendingFrameIndex_ = -1;
    pendingFrameRequestId_ = 0;
    frameWorkerActive_ = true;
    frameWorkerRequestId_ = requestId;
    frameCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    framePool_.start(new FrameLoadTask(this, stackInfo_, frameCancelFlag_, filePath_, generation_, requestId, frameIndex));
}

void TiffViewerWidget::prefetchNeighbors(int frameIndex)
{
    if (!hasInfo_ || info_.frameCount <= 1) {
        return;
    }

    const int direction = lastFrameStep_ < 0 ? -1 : 1;
    startPrefetch(frameIndex + direction);
    startPrefetch(frameIndex + (2 * direction));
    startPrefetch(frameIndex - direction);
}

void TiffViewerWidget::startPrefetch(int frameIndex)
{
    if (!hasInfo_ || frameIndex < 0 || frameIndex >= info_.frameCount) {
        return;
    }
    if (frameIndex == currentFrameIndex_) {
        return;
    }
    if (frameCache_.contains(frameIndex) || prefetchInFlight_.contains(frameIndex)) {
        return;
    }

    prefetchInFlight_.insert(frameIndex);
    if (prefetchCancelFlag_ == nullptr) {
        prefetchCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    }
    prefetchPool_.start(new PrefetchLoadTask(
        this,
        stackInfo_,
        prefetchCancelFlag_,
        filePath_,
        generation_,
        prefetchGeneration_,
        frameIndex));
}

void TiffViewerWidget::displayFrame(int frameIndex, const CachedFrame& frame)
{
    displayedFrame_ = frame;
    hasDisplayedFrame_ = true;
    if (!displayLevelsInitialized_) {
        setDisplayLevelsFromFrame(frame);
    }
    startFrameRender(frameIndex, displayedFrame_);
    setStatus(QStringLiteral("Rendering frame %1/%2...").arg(frameIndex + 1).arg(info_.frameCount));
}

void TiffViewerWidget::refreshDisplayedFrame()
{
    if (!hasDisplayedFrame_) {
        return;
    }

    if (displayWhiteSpin_->value() <= displayBlackSpin_->value()) {
        updatingDisplayControls_ = true;
        const double delta = std::max(displayBlackSpin_->singleStep(), 1e-6);
        double white = std::min(displayWhiteSpin_->maximum(), displayBlackSpin_->value() + delta);
        if (white <= displayBlackSpin_->value()) {
            displayBlackSpin_->setValue(std::max(displayBlackSpin_->minimum(), white - delta));
        }
        displayWhiteSpin_->setValue(white);
        updatingDisplayControls_ = false;
    }

    startFrameRender(currentFrameIndex_, displayedFrame_);
    setStatus(QStringLiteral("Rendering frame %1/%2...").arg(currentFrameIndex_ + 1).arg(info_.frameCount));
}

void TiffViewerWidget::startFrameRender(int frameIndex, const CachedFrame& frame)
{
    cancelActiveRender();
    if (!frame.hasSamples()) {
        canvas_->setImage(QImage());
        return;
    }

    ++renderRequestId_;
    renderCancelFlag_ = std::make_shared<std::atomic_bool>(false);
    renderPool_.start(new FrameRenderTask(
        this,
        frame,
        filePath_,
        generation_,
        renderRequestId_,
        frameIndex,
        displayBlackSpin_->value(),
        displayWhiteSpin_->value(),
        renderCancelFlag_));
}

void TiffViewerWidget::setDisplayLevelsFromFrame(const CachedFrame& frame)
{
    const DisplayLevels levels = displayLevelsForFrame(frame, false);

    updatingDisplayControls_ = true;
    for (QDoubleSpinBox* spin : {displayBlackSpin_, displayWhiteSpin_}) {
        spin->setDecimals(levels.decimals);
        spin->setRange(levels.minimum, levels.maximum);
        spin->setSingleStep(levels.step);
        spin->setEnabled(true);
    }
    displayBlackSpin_->setValue(levels.black);
    displayWhiteSpin_->setValue(levels.white);
    autoDisplayButton_->setEnabled(true);
    fullDisplayButton_->setEnabled(true);
    updatingDisplayControls_ = false;
    displayLevelsInitialized_ = true;
}

void TiffViewerWidget::setDisplayFullRange()
{
    if (!hasDisplayedFrame_) {
        return;
    }
    const DisplayLevels levels = displayLevelsForFrame(displayedFrame_, true);
    updatingDisplayControls_ = true;
    for (QDoubleSpinBox* spin : {displayBlackSpin_, displayWhiteSpin_}) {
        spin->setDecimals(levels.decimals);
        spin->setRange(levels.minimum, levels.maximum);
        spin->setSingleStep(levels.step);
        spin->setEnabled(true);
    }
    displayBlackSpin_->setValue(levels.black);
    displayWhiteSpin_->setValue(levels.white);
    autoDisplayButton_->setEnabled(true);
    fullDisplayButton_->setEnabled(true);
    updatingDisplayControls_ = false;
    displayLevelsInitialized_ = true;
}

QImage TiffViewerWidget::renderFrameImage(const CachedFrame& frame) const
{
    return renderCachedFrameImage(frame, displayBlackSpin_->value(), displayWhiteSpin_->value());
}

void TiffViewerWidget::loadRoiOverlay(const QString& path)
{
    const NativeMetadataResult metadata = NativeMetadataReader::readForTiff(path);
    if (!metadata.ok) {
        clearRoiOverlay(QStringLiteral("Metadata: not available"));
        metadataLabel_->setToolTip(metadata.error);
        return;
    }

    if (metadata.overlay.width != info_.width || metadata.overlay.height != info_.height) {
        clearRoiOverlay(
            QStringLiteral("Metadata: size mismatch (%1 x %2)")
                .arg(metadata.overlay.width)
                .arg(metadata.overlay.height));
        metadataLabel_->setToolTip(
            QStringLiteral("TIFF is %1 x %2, metadata is %3 x %4.")
                .arg(info_.width)
                .arg(info_.height)
                .arg(metadata.overlay.width)
                .arg(metadata.overlay.height));
        return;
    }

    roiOverlay_ = metadata.overlay;
    canvas_->setRoiOverlay(roiOverlay_);
    roiOverlayCheck_->setEnabled(true);
    metadataLabel_->setText(roiOverlay_.summary());
    metadataLabel_->setToolTip(roiOverlay_.metadataPath);
}

void TiffViewerWidget::clearRoiOverlay(const QString& message)
{
    roiOverlay_ = NativeRoiOverlay();
    canvas_->clearRoiOverlay();
    roiOverlayCheck_->setEnabled(false);
    metadataLabel_->setText(message);
}

void TiffViewerWidget::setStatus(const QString& message)
{
    statusLabel_->setText(message);
}
