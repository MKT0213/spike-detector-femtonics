#pragma once

#include "FrameCache.h"
#include "NativeMetadata.h"
#include "TiffStack.h"

#include <QImage>
#include <QSet>
#include <QThreadPool>
#include <QWidget>

#include <atomic>
#include <memory>

class ImageCanvas;
class QCheckBox;
class QDoubleSpinBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;
class QSlider;
class QSpinBox;
class QStackedWidget;

class TiffViewerWidget : public QWidget {
    Q_OBJECT

public:
    explicit TiffViewerWidget(QWidget* parent = nullptr);

    void loadFile(const QString& path);
    void loadImageFile(const QString& path);
    void loadTextFile(const QString& path);
    void clearView();
    void setCurrentFrameIndex(int frameIndex);
    QString currentFilePath() const;
    int currentFrameIndex() const;
    QString statusText() const;
    bool hasDisplayedFrame() const;
    void completeInfoLoad(
        quint64 generation,
        const QString& path,
        bool previewOnly,
        bool ok,
        const TiffStackInfo& info,
        const QString& error);
    void completeFrameLoad(
        quint64 generation,
        quint64 requestId,
        const QString& path,
        int frameIndex,
        std::shared_ptr<TiffFrameResult> result);
    void completeFrameRender(
        quint64 generation,
        quint64 renderRequestId,
        const QString& path,
        int frameIndex,
        const QImage& image,
        qint64 renderElapsedMs,
        bool cancelled);
    void completePrefetchLoad(
        quint64 generation,
        quint64 prefetchGeneration,
        const QString& path,
        int frameIndex,
        std::shared_ptr<TiffFrameResult> result);

signals:
    void fileLoaded(const QString& path);
    void frameRendered(const QString& path, int frameIndex);

private slots:
    void openFileDialog();
    void requestFrameFromSlider(int value);
    void requestFrameFromSpin(int value);

private:
    void cancelPendingInfoLoads();
    void cancelActiveFrameLoad();
    void cancelPendingPrefetchLoads();
    void cancelActiveRender();
    void resetForOpen(const QString& path);
    void startFullInfoLoad();
    void configureFrameControls(int frameCount, int frameIndex);
    void syncFrameControls();
    void requestFrame(int frameIndex);
    void startPendingFrameWorker();
    void prefetchNeighbors(int frameIndex);
    void startPrefetch(int frameIndex);
    void displayFrame(int frameIndex, const CachedFrame& frame);
    void refreshDisplayedFrame();
    void startFrameRender(int frameIndex, const CachedFrame& frame);
    void setDisplayLevelsFromFrame(const CachedFrame& frame);
    void setDisplayFullRange();
    QImage renderFrameImage(const CachedFrame& frame) const;
    void loadRoiOverlay(const QString& path);
    void clearRoiOverlay(const QString& message);
    void setStatus(const QString& message);

    QPushButton* openButton_ = nullptr;
    QCheckBox* roiOverlayCheck_ = nullptr;
    QLabel* pathLabel_ = nullptr;
    QLabel* metadataLabel_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QLabel* frameTotalLabel_ = nullptr;
    QSlider* frameSlider_ = nullptr;
    QSpinBox* frameSpin_ = nullptr;
    QDoubleSpinBox* displayBlackSpin_ = nullptr;
    QDoubleSpinBox* displayWhiteSpin_ = nullptr;
    QPushButton* autoDisplayButton_ = nullptr;
    QPushButton* fullDisplayButton_ = nullptr;
    QStackedWidget* contentStack_ = nullptr;
    ImageCanvas* canvas_ = nullptr;
    QPlainTextEdit* textPreview_ = nullptr;

    QThreadPool previewInfoPool_;
    QThreadPool fullInfoPool_;
    QThreadPool framePool_;
    QThreadPool prefetchPool_;
    QThreadPool renderPool_;
    FrameCache frameCache_;

    QString filePath_;
    TiffStackInfo info_;
    std::shared_ptr<const TiffStackInfo> stackInfo_;
    std::shared_ptr<std::atomic_bool> infoCancelFlag_;
    std::shared_ptr<std::atomic_bool> frameCancelFlag_;
    std::shared_ptr<std::atomic_bool> prefetchCancelFlag_;
    std::shared_ptr<std::atomic_bool> renderCancelFlag_;
    NativeRoiOverlay roiOverlay_;
    CachedFrame displayedFrame_;
    bool hasInfo_ = false;
    bool hasDisplayedFrame_ = false;
    bool displayLevelsInitialized_ = false;
    bool updatingDisplayControls_ = false;
    bool updatingControls_ = false;
    bool frameWorkerActive_ = false;
    int currentFrameIndex_ = 0;
    int pendingFrameIndex_ = -1;
    int lastFrameStep_ = 1;
    QSet<int> prefetchInFlight_;
    quint64 generation_ = 0;
    quint64 prefetchGeneration_ = 0;
    quint64 requestId_ = 0;
    quint64 pendingFrameRequestId_ = 0;
    quint64 frameWorkerRequestId_ = 0;
    quint64 renderRequestId_ = 0;
};
