#pragma once

#include <QMainWindow>
#include <QSet>
#include <QString>
#include <QStringList>

#include <atomic>
#include <functional>
#include <memory>

class QCloseEvent;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProcess;
class QProgressBar;
class QPushButton;
class QTimer;
class TiffViewerWidget;

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    enum class ResultOpenMode {
        Viewer,
        Image,
        Text,
        FolderTiffs,
        Folder,
        External,
        Missing,
    };

    explicit MainWindow(QWidget* parent = nullptr);

    void openPath(const QString& path);
    void openPaths(const QStringList& paths);
    void openFolder(const QString& folderPath);
    void queuePathForAnalysis(const QString& path);
    void queuePathsForAnalysis(const QStringList& paths);
    int queuedTiffCount() const;
    bool isFolderScanPending() const;
    bool isQueuePopulationPending() const;
    QString queueStatusText() const;
    void setRecursiveFolderScan(bool recursive);
    bool isQueueBenchmarkActive() const;
    bool isAnalysisProcessRunning() const;
    QString processLogText() const;
    QString currentViewerPath() const;
    bool viewerHasDisplayedFrame() const;
    void setOutputFolder(const QString& folderPath);
    void setPythonBackendPath(const QString& pythonPath);
    void setMotionHookSpec(const QString& hookSpec);
    void setRoiHookSpec(const QString& hookSpec);
    void setMotionHookParams(const QString& params);
    void setRoiHookParams(const QString& params);
    void setMaskBackgroundMode(const QString& mode);
    void setPipelineStepSelection(bool averagePng, bool exportSignals, bool splitRois);
    void runAveragePngForCurrent();
    void runPipelineForCurrent();
    void runPipelineForQueue();
    void runBackendValidation();
    void runQueueBenchmark();
    void completeFolderScan(
        quint64 generation,
        const QString& folderPath,
        const QStringList& paths,
        bool cancelled);
    void completeQueueBenchmark(
        quint64 generation,
        const QStringList& lines,
        bool cancelled,
        bool ok);
    static QStringList scanFolderPaths(
        const QString& folderPath,
        bool recursive,
        const std::function<bool()>& shouldCancel = {});
    static QString findWorkingTiffFromManifest(
        const QString& manifestPath,
        const QString& currentSourcePath = QString(),
        QString* errorMessage = nullptr,
        const QString& relativeRootPath = QString());
    static QString findPreferredDisplayPathFromManifest(
        const QString& manifestPath,
        const QString& currentSourcePath = QString(),
        QString* errorMessage = nullptr,
        const QString& relativeRootPath = QString());
    static QStringList findResultPathsFromManifest(
        const QString& manifestPath,
        QString* errorMessage = nullptr,
        const QString& relativeRootPath = QString());
    static QString resultPathFromLogLine(
        const QString& line,
        const QString& relativeRootPath = QString());
    static bool parseAnalysisProgressLine(
        const QString& line,
        QString* stageLabel,
        int* done,
        int* total);
    static ResultOpenMode resultOpenModeForPath(const QString& path);
    static QString resultOpenModeName(ResultOpenMode mode);
    static QString resultActionLabel(ResultOpenMode mode);

private:
    void closeEvent(QCloseEvent* event) override;
    void openFileDialog();
    void openFolderDialog();
    void startFolderScan(const QString& folderPath);
    void cancelFolderScan();
    void startQueuedPathAdd(const QStringList& paths, bool openFirst);
    void cancelQueuedPathAdd();
    void addNextQueuedPathBatch(quint64 generation);
    void openPendingQueueFirstUsable();
    void chooseOutputFolder();
    void openResultsFolder();
    void openWorkingTiff();
    void openWorkingTiffIfConfigured();
    void openSelectedResult();
    void choosePythonBackendFile();
    void chooseMotionHookFile();
    void chooseRoiHookFile();
    void chooseHookFileFor(QLineEdit* edit, const QString& title);
    void validateHooks();
    void setBuiltInMotionHookDefaults();
    void setBuiltInRoiHookDefaults();
    void updateProcessingStageUi();
    void addPaths(const QStringList& paths, bool openFirst);
    void removeSelectedQueueItem();
    void clearQueue();
    void exportAveragePng();
    void exportCurrentSignals();
    void runCurrentPipeline();
    void runQueuePipeline();
    void cancelQueueBenchmark();
    void cancelAnalysis();
    void splitCurrentTiff();
    QStringList outputArgsForCurrentTiff(bool splitOutput) const;
    void appendPipelineStepArgs(QStringList& args) const;
    void appendMaskBackgroundArgs(QStringList& args) const;
    void appendHookParameterArgs(QStringList& args, const QLineEdit* edit, const QString& option) const;
    void appendHookParameterText(QStringList& args, const QString& text, const QString& option) const;
    void appendSelectedHookArgs(QStringList& args) const;
    QStringList samplingArgs() const;
    QString currentTiffOutputDir(const QString& path) const;
    QString defaultExportsDirForTiff(const QString& path) const;
    QString defaultSplitDirForTiff(const QString& path) const;
    QString averageResultsDirForCurrentTiff() const;
    QString signalResultsDirForCurrentTiff() const;
    QString splitResultsDirForCurrentTiff() const;
    QString currentPipelineResultsDir() const;
    QString queuePipelineResultsDir() const;
    void runPythonModule(
        const QString& moduleName,
        const QStringList& moduleArgs,
        const QString& label,
        const QString& resultsPath = QString(),
        const QString& autoOpenSourcePath = QString());
    void appendAnalysisLog(const QString& message);
    void appendAnalysisProcessOutput(QString* pendingText, const QString& message);
    void flushAnalysisProcessOutput(QString* pendingText);
    void flushAnalysisLogBuffer(int maxLines = 0);
    void updateAnalysisProgressFromLine(const QString& line);
    void setAnalysisProgressIdle(const QString& text);
    void setAnalysisProgressBusy(const QString& text);
    void setBackendStatus(const QString& text, const QString& toolTip = QString());
    void detectAnalysisOutputPath(const QString& line);
    void detectAnalysisResultPath(const QString& line);
    void setLastResultsPath(const QString& path);
    void detectWorkingTiffFromManifest();
    void detectPreferredDisplayPathFromManifest();
    void setLastWorkingTiffPath(const QString& path);
    void updateProcessedOutputButton();
    void updateResultListFromManifest();
    void setResultPaths(const QStringList& paths);
    void addResultPath(const QString& path);
    void updateResultActionUi();
    void setAnalysisBusy(bool busy);
    void openQueueRow(int row);
    void selectQueuePath(const QString& path);
    void setQueueStatus();
    void loadUserSettings();
    void saveUserSettings() const;
    QString currentQueuePath() const;
    QString pythonExecutable(
        QStringList* diagnostics = nullptr,
        QStringList* pythonPathEntries = nullptr) const;
    QString repoRootPath() const;
    QStringList scanFolder(const QString& folderPath) const;

    QListWidget* queueList_ = nullptr;
    QLabel* queueStatus_ = nullptr;
    QPushButton* removeQueueButton_ = nullptr;
    QPushButton* clearQueueButton_ = nullptr;
    QCheckBox* recursiveCheck_ = nullptr;
    QLineEdit* outputFolderEdit_ = nullptr;
    QPushButton* openResultsButton_ = nullptr;
    QPushButton* openWorkingTiffButton_ = nullptr;
    QListWidget* resultsList_ = nullptr;
    QPushButton* openResultButton_ = nullptr;
    QLabel* backendStatusLabel_ = nullptr;
    QLineEdit* pythonBackendEdit_ = nullptr;
    QPushButton* choosePythonBackendButton_ = nullptr;
    QLineEdit* motionHookEdit_ = nullptr;
    QLineEdit* roiHookEdit_ = nullptr;
    QLineEdit* motionHookParamsEdit_ = nullptr;
    QLineEdit* roiHookParamsEdit_ = nullptr;
    QPushButton* builtInMotionHookButton_ = nullptr;
    QPushButton* builtInRoiHookButton_ = nullptr;
    QPushButton* chooseMotionHookButton_ = nullptr;
    QPushButton* chooseRoiHookButton_ = nullptr;
    QPushButton* validateHooksButton_ = nullptr;
    QCheckBox* useMetadataSamplingCheck_ = nullptr;
    QDoubleSpinBox* samplingRateSpin_ = nullptr;
    QCheckBox* writeCsvCheck_ = nullptr;
    QCheckBox* runMotionCorrectionCheck_ = nullptr;
    QCheckBox* runSegmentationCheck_ = nullptr;
    QCheckBox* runDynamicSegmentationCheck_ = nullptr;
    QCheckBox* pipelineAverageCheck_ = nullptr;
    QCheckBox* pipelineSignalsCheck_ = nullptr;
    QCheckBox* pipelineSplitCheck_ = nullptr;
    QComboBox* maskBackgroundCombo_ = nullptr;
    QCheckBox* autoOpenWorkingTiffCheck_ = nullptr;
    QProgressBar* analysisProgressBar_ = nullptr;
    QPushButton* splitButton_ = nullptr;
    QPushButton* exportSignalsButton_ = nullptr;
    QPushButton* averagePngButton_ = nullptr;
    QPushButton* runPipelineButton_ = nullptr;
    QPushButton* runQueuePipelineButton_ = nullptr;
    QPushButton* benchmarkQueueButton_ = nullptr;
    QPushButton* cancelAnalysisButton_ = nullptr;
    QPlainTextEdit* analysisLog_ = nullptr;
    QTimer* analysisLogFlushTimer_ = nullptr;
    QProcess* analysisProcess_ = nullptr;
    TiffViewerWidget* viewer_ = nullptr;
    QStringList pendingAnalysisLogDisplayLines_;
    QStringList analysisLogMirrorLines_;
    QString pendingAnalysisStdoutText_;
    QString pendingAnalysisStderrText_;
    std::shared_ptr<std::atomic_bool> folderScanCancelFlag_;
    quint64 folderScanGeneration_ = 0;
    std::shared_ptr<std::atomic_bool> queueBenchmarkCancelFlag_;
    quint64 queueBenchmarkGeneration_ = 0;
    bool queueBenchmarkActive_ = false;
    QStringList pendingQueuePaths_;
    QSet<QString> pendingQueueExistingPaths_;
    bool pendingQueueOpenFirst_ = false;
    bool pendingQueueOpenedFirst_ = false;
    int pendingQueueIndex_ = 0;
    int pendingQueueFirstAddedRow_ = -1;
    QString pendingQueueFirstUsablePath_;
    quint64 queueAddGeneration_ = 0;
    QString lastResultsPath_;
    QString lastManifestPath_;
    QString lastWorkingTiffPath_;
    QString lastDisplayResultPath_;
    QString analysisAutoOpenSourcePath_;
};
