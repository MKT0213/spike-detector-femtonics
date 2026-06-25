#pragma once

#include <QHash>
#include <QList>
#include <QString>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

struct CachedFrame {
    int width = 0;
    int height = 0;
    int bitsPerSample = 0;
    int sampleFormat = 1;
    double observedMin = 0.0;
    double observedMax = 0.0;
    std::shared_ptr<const std::vector<uint8_t>> samples8;
    std::shared_ptr<const std::vector<uint16_t>> samples16;
    std::shared_ptr<const std::vector<float>> samplesFloat;
    qint64 elapsedMs = 0;
    QString source;

    bool hasSamples() const;
    size_t sampleBytes() const;
};

class FrameCache {
public:
    explicit FrameCache(int maxFrames = 8, size_t maxBytes = 256ULL * 1024ULL * 1024ULL);

    void clear();
    bool contains(int frameIndex) const;
    int size() const;
    size_t totalBytes() const;
    size_t maxBytes() const;
    std::optional<CachedFrame> get(int frameIndex);
    void put(int frameIndex, const CachedFrame& frame);

private:
    void touch(int frameIndex);
    void remove(int frameIndex);
    void enforceLimits();

    int maxFrames_;
    size_t maxBytes_;
    size_t totalBytes_ = 0;
    QHash<int, CachedFrame> frames_;
    QList<int> lruOrder_;
};
