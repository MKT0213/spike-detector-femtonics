#include "NativeMetadata.h"

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QRegularExpression>
#include <QStringList>
#include <QTextStream>

#include <algorithm>

namespace {

bool parseIntField(const QHash<QString, QString>& fields, const QString& key, int* value, QString* error)
{
    if (!fields.contains(key)) {
        if (error != nullptr) {
            *error = QStringLiteral("Metadata is missing required field: %1.").arg(key);
        }
        return false;
    }

    bool ok = false;
    const int parsed = fields.value(key).trimmed().toInt(&ok);
    if (!ok) {
        if (error != nullptr) {
            *error = QStringLiteral("Invalid integer for %1: %2").arg(key, fields.value(key));
        }
        return false;
    }

    if (value != nullptr) {
        *value = parsed;
    }
    return true;
}

QList<int> parseIntegerList(const QString& value)
{
    QList<int> values;
    static const QRegularExpression numberPattern(QStringLiteral("-?\\d+"));
    QRegularExpressionMatchIterator iterator = numberPattern.globalMatch(value);
    while (iterator.hasNext()) {
        const QRegularExpressionMatch match = iterator.next();
        values.append(match.captured(0).toInt());
    }
    return values;
}

bool parseRoiSizes(
    const QString& rawValue,
    int roiCount,
    QList<QPair<int, int>>* sizes,
    QString* error)
{
    const QList<int> values = parseIntegerList(rawValue);
    if (values.size() == 2) {
        for (int index = 0; index < roiCount; ++index) {
            sizes->append(qMakePair(values.at(0), values.at(1)));
        }
        return true;
    }

    if (values.size() != roiCount * 2) {
        if (error != nullptr) {
            *error = QStringLiteral("RoiSize has %1 numeric values but expected 2 or %2.")
                         .arg(values.size())
                         .arg(roiCount * 2);
        }
        return false;
    }

    for (int index = 0; index < values.size(); index += 2) {
        sizes->append(qMakePair(values.at(index), values.at(index + 1)));
    }
    return true;
}

QList<int> parseRoiIndices(const QString& rawValue, int roiCount)
{
    QList<int> indices = parseIntegerList(rawValue);
    if (indices.size() == roiCount) {
        return indices;
    }

    indices.clear();
    for (int index = 1; index <= roiCount; ++index) {
        indices.append(index);
    }
    return indices;
}

double extractSamplingRate(const QHash<QString, QString>& fields, bool* ok)
{
    QStringList searchValues;
    for (const QString& key : {QStringLiteral("Comment"), QStringLiteral("SamplingRate"), QStringLiteral("SamplingRateHz"), QStringLiteral("FrameRate"), QStringLiteral("FrameRateHz")}) {
        if (fields.contains(key)) {
            searchValues.append(fields.value(key));
        }
    }
    for (auto iterator = fields.constBegin(); iterator != fields.constEnd(); ++iterator) {
        if (!searchValues.contains(iterator.value())) {
            searchValues.append(iterator.value());
        }
    }

    static const QRegularExpression ratePattern(QStringLiteral("(\\d+(?:\\.\\d+)?)\\s*(?:hz|hertz)\\b"), QRegularExpression::CaseInsensitiveOption);
    for (const QString& value : searchValues) {
        const QRegularExpressionMatch match = ratePattern.match(value);
        if (match.hasMatch()) {
            bool parsedOk = false;
            const double rate = match.captured(1).toDouble(&parsedOk);
            if (parsedOk && rate > 0.0) {
                if (ok != nullptr) {
                    *ok = true;
                }
                return rate;
            }
        }
    }

    if (ok != nullptr) {
        *ok = false;
    }
    return 0.0;
}

} // namespace

bool NativeRoiOverlay::isValid() const
{
    return width > 0 && height > 0 && roiCount > 0 && boxes.size() == roiCount;
}

QString NativeRoiOverlay::summary() const
{
    if (!isValid()) {
        return QStringLiteral("Metadata: no ROI overlay");
    }

    QString text = QStringLiteral("Metadata: %1 ROI, %2 x %3 grid, ROI %4 x %5")
                       .arg(roiCount)
                       .arg(columns)
                       .arg(rows)
                       .arg(roiWidth)
                       .arg(roiHeight);
    if (hasSamplingRate) {
        text += QStringLiteral(", %1 Hz").arg(samplingRateHz, 0, 'g', 6);
    }
    if (!channel.isEmpty()) {
        text += QStringLiteral(", %1").arg(channel);
    }
    return text;
}

QString NativeMetadataReader::metadataPathForTiff(const QString& tiffPath)
{
    return tiffPath + QStringLiteral(".metadata.txt");
}

NativeMetadataResult NativeMetadataReader::readForTiff(const QString& tiffPath)
{
    return read(metadataPathForTiff(tiffPath));
}

NativeMetadataResult NativeMetadataReader::read(const QString& metadataPath)
{
    NativeMetadataResult result;
    result.overlay.metadataPath = metadataPath;

    QFile file(metadataPath);
    if (!file.exists()) {
        result.error = QStringLiteral("Metadata file was not found: %1").arg(metadataPath);
        return result;
    }
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        result.error = QStringLiteral("Could not open metadata file: %1").arg(metadataPath);
        return result;
    }

    QTextStream stream(&file);
    QHash<QString, QString> fields;
    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (line.isEmpty()
            || line.startsWith(QLatin1Char('%'))
            || line.startsWith(QLatin1Char('>'))
            || line.startsWith(QLatin1Char('<'))) {
            continue;
        }

        const int separator = line.indexOf(QLatin1Char(':'));
        if (separator < 0) {
            continue;
        }

        QString key = line.left(separator).trimmed();
        const QString value = line.mid(separator + 1).trimmed();
        if (key.startsWith(QStringLiteral("RoiSize"))) {
            key = QStringLiteral("RoiSize");
        }
        fields.insert(key, value);
    }

    int width = 0;
    int height = 0;
    int roiCount = 0;
    if (!parseIntField(fields, QStringLiteral("Width"), &width, &result.error)
        || !parseIntField(fields, QStringLiteral("Height"), &height, &result.error)
        || !parseIntField(fields, QStringLiteral("RoiCount"), &roiCount, &result.error)) {
        return result;
    }

    if (width <= 0 || height <= 0 || roiCount <= 0) {
        result.error = QStringLiteral("Metadata width, height, and RoiCount must be positive.");
        return result;
    }

    if (!fields.contains(QStringLiteral("RoiSize"))) {
        result.error = QStringLiteral("Metadata is missing required field: RoiSize.");
        return result;
    }

    QList<QPair<int, int>> roiSizes;
    if (!parseRoiSizes(fields.value(QStringLiteral("RoiSize")), roiCount, &roiSizes, &result.error)) {
        return result;
    }

    const QPair<int, int> firstSize = roiSizes.first();
    if (firstSize.first <= 0 || firstSize.second <= 0) {
        result.error = QStringLiteral("ROI size must be positive.");
        return result;
    }
    const bool mixedSizes = std::any_of(roiSizes.constBegin(), roiSizes.constEnd(), [firstSize](const QPair<int, int>& size) {
        return size != firstSize;
    });
    if (mixedSizes) {
        result.error = QStringLiteral("Mixed ROI sizes are not supported by native overlay v1.");
        return result;
    }

    const int roiWidth = firstSize.first;
    const int roiHeight = firstSize.second;
    if (width % roiWidth != 0 || height % roiHeight != 0) {
        result.error = QStringLiteral("ROI size %1x%2 does not evenly tile metadata size %3x%4.")
                           .arg(roiWidth)
                           .arg(roiHeight)
                           .arg(width)
                           .arg(height);
        return result;
    }

    const int columns = width / roiWidth;
    const int rows = height / roiHeight;
    if (columns * rows != roiCount) {
        result.error = QStringLiteral("Computed grid has %1 tiles (%2 x %3), but RoiCount is %4.")
                           .arg(columns * rows)
                           .arg(columns)
                           .arg(rows)
                           .arg(roiCount);
        return result;
    }

    QList<int> roiIndices;
    if (fields.contains(QStringLiteral("RoiIndex"))) {
        roiIndices = parseRoiIndices(fields.value(QStringLiteral("RoiIndex")), roiCount);
    } else {
        roiIndices = parseRoiIndices(QString(), roiCount);
    }

    NativeRoiOverlay overlay;
    overlay.metadataPath = QFileInfo(metadataPath).absoluteFilePath();
    overlay.width = width;
    overlay.height = height;
    overlay.roiCount = roiCount;
    overlay.columns = columns;
    overlay.rows = rows;
    overlay.roiWidth = roiWidth;
    overlay.roiHeight = roiHeight;
    overlay.channel = fields.value(QStringLiteral("Channel"));
    overlay.dataType = fields.value(QStringLiteral("DataType"));
    overlay.comment = fields.value(QStringLiteral("Comment"));
    overlay.samplingRateHz = extractSamplingRate(fields, &overlay.hasSamplingRate);

    for (int ordinal = 1; ordinal <= roiCount; ++ordinal) {
        const int zeroBased = ordinal - 1;
        const int row = zeroBased / columns;
        const int column = zeroBased % columns;
        NativeRoiBox box;
        box.ordinal = ordinal;
        box.roiIndex = roiIndices.at(zeroBased);
        box.row = row;
        box.column = column;
        box.left = column * roiWidth;
        box.upper = row * roiHeight;
        box.right = box.left + roiWidth;
        box.lower = box.upper + roiHeight;
        overlay.boxes.append(box);
    }

    result.ok = true;
    result.overlay = overlay;
    return result;
}
