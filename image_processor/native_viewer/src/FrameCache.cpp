#include "FrameCache.h"

#include <algorithm>

namespace {

constexpr int SampleFormatUint = 1;
constexpr int SampleFormatInt = 2;
constexpr int SampleFormatIeeeFloat = 3;

} // namespace

bool CachedFrame::hasSamples() const
{
    if (width <= 0 || height <= 0) {
        return false;
    }
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (sampleFormat == SampleFormatUint && bitsPerSample == 8) {
        return samples8 != nullptr
            && samples8->size() == pixelCount;
    }
    if (sampleFormat == SampleFormatUint && bitsPerSample == 16) {
        return samples16 != nullptr
            && samples16->size() == pixelCount;
    }
    if ((sampleFormat == SampleFormatInt && bitsPerSample == 16)
        || (sampleFormat == SampleFormatIeeeFloat && bitsPerSample == 32)) {
        return samplesFloat != nullptr
            && samplesFloat->size() == pixelCount;
    }
    return false;
}

size_t CachedFrame::sampleBytes() const
{
    if (samples8 != nullptr) {
        return samples8->size() * sizeof(uint8_t);
    }
    if (samples16 != nullptr) {
        return samples16->size() * sizeof(uint16_t);
    }
    if (samplesFloat != nullptr) {
        return samplesFloat->size() * sizeof(float);
    }
    return 0;
}

FrameCache::FrameCache(int maxFrames, size_t maxBytes)
    : maxFrames_(maxFrames)
    , maxBytes_(maxBytes)
{
}

void FrameCache::clear()
{
    frames_.clear();
    lruOrder_.clear();
    totalBytes_ = 0;
}

bool FrameCache::contains(int frameIndex) const
{
    return frames_.contains(frameIndex);
}

int FrameCache::size() const
{
    return frames_.size();
}

size_t FrameCache::totalBytes() const
{
    return totalBytes_;
}

size_t FrameCache::maxBytes() const
{
    return maxBytes_;
}

std::optional<CachedFrame> FrameCache::get(int frameIndex)
{
    if (!frames_.contains(frameIndex)) {
        return std::nullopt;
    }
    touch(frameIndex);
    return frames_.value(frameIndex);
}

void FrameCache::put(int frameIndex, const CachedFrame& frame)
{
    const size_t frameBytes = frame.sampleBytes();
    if (maxFrames_ <= 0 || maxBytes_ == 0 || !frame.hasSamples() || frameBytes == 0 || frameBytes > maxBytes_) {
        remove(frameIndex);
        return;
    }

    if (frames_.contains(frameIndex)) {
        remove(frameIndex);
    }
    frames_.insert(frameIndex, frame);
    totalBytes_ += frameBytes;
    touch(frameIndex);
    enforceLimits();
}

void FrameCache::remove(int frameIndex)
{
    if (frames_.contains(frameIndex)) {
        totalBytes_ -= std::min(totalBytes_, frames_.value(frameIndex).sampleBytes());
        frames_.remove(frameIndex);
    }
    lruOrder_.removeAll(frameIndex);
}

void FrameCache::enforceLimits()
{
    while (!lruOrder_.isEmpty()
           && (lruOrder_.size() > maxFrames_ || totalBytes_ > maxBytes_)) {
        remove(lruOrder_.last());
    }
}

void FrameCache::touch(int frameIndex)
{
    lruOrder_.removeAll(frameIndex);
    lruOrder_.prepend(frameIndex);
}
