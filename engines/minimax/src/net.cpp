#include "poe2/minimax/net.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pattern_gain_layout.hpp"
#include "poe2/minimax/evaluation.hpp"
#include "poe2/minimax/model_header.hpp"
#include "poe2/minimax/sha256.hpp"

namespace poe2::minimax {

namespace {

constexpr std::array<std::uint8_t, 8> kMagic{{'P', 'O', 'E', '2', 'N', 'E', 'T', 0}};
constexpr std::uint32_t kContainerVersion = 1;
constexpr std::size_t kDesignBytes = 32;
constexpr std::size_t kTrailerBytes = 32;
constexpr std::size_t kFixedHeaderBytes = 56;
constexpr std::size_t kMaximumTables = 16;
constexpr std::size_t kMaximumMetadataBytes = 16384;
constexpr std::array<std::string_view, 1> kSupportedDesigns{{kPatternGainDesign}};

// pattern-gain-v1 tables, in file order: line knots (u8), gain knots (u8), intercepts (i32),
// line weights (i16), gain weights (i16).
constexpr std::array<std::uint32_t, 5> kPatternGainTableSizes{{
    kPatternGainLineKnotCount,
    kPatternGainKnotCount,
    kPatternGainPlyCount,
    kPatternGainLineKnotCount* kPatternGainOrbitCount,
    kPatternGainKnotCount* kPatternGainFeatureCount,
}};

// Bounds every evaluation well inside the search's int32 window: the line and gain residuals of
// int16 weights stay below 2^29 in magnitude, so intercepts get the remaining headroom.
constexpr std::int32_t kMaximumIntercept = std::int32_t{1} << 24;

class Reader final {
 public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  [[nodiscard]] std::span<const std::uint8_t> take(std::size_t count) {
    if (count > bytes_.size() - offset_) {
      throw NetError{"net file is truncated"};
    }
    const std::span<const std::uint8_t> result = bytes_.subspan(offset_, count);
    offset_ += count;
    return result;
  }

  [[nodiscard]] std::uint32_t u32() {
    const std::span<const std::uint8_t> bytes = take(4);
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
  }

  [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }

  [[nodiscard]] std::int16_t i16() {
    const std::span<const std::uint8_t> bytes = take(2);
    return static_cast<std::int16_t>(static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[0]) | (static_cast<std::uint16_t>(bytes[1]) << 8)));
  }

  [[nodiscard]] std::uint8_t u8() { return take(1)[0]; }

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

 private:
  std::span<const std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

class Writer final {
 public:
  void bytes(std::span<const std::uint8_t> values) {
    output_.insert(output_.end(), values.begin(), values.end());
  }
  void u32(std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
      output_.push_back(static_cast<std::uint8_t>(value >> shift));
    }
  }
  void i32(std::int32_t value) { u32(static_cast<std::uint32_t>(value)); }
  void i16(std::int16_t value) {
    const auto bits = static_cast<std::uint16_t>(value);
    output_.push_back(static_cast<std::uint8_t>(bits));
    output_.push_back(static_cast<std::uint8_t>(bits >> 8));
  }
  void u8(std::uint8_t value) { output_.push_back(value); }

  [[nodiscard]] std::vector<std::uint8_t> take() { return std::move(output_); }
  [[nodiscard]] const std::vector<std::uint8_t>& view() const noexcept { return output_; }

 private:
  std::vector<std::uint8_t> output_;
};

[[nodiscard]] std::string supported_design_list() {
  std::string text;
  for (const std::string_view design : kSupportedDesigns) {
    if (!text.empty()) {
      text += ", ";
    }
    text += design;
  }
  return text;
}

[[nodiscard]] std::string design_name(std::span<const std::uint8_t> field) {
  const auto end = std::find(field.begin(), field.end(), std::uint8_t{0});
  if (std::any_of(end, field.end(), [](std::uint8_t byte) { return byte != 0; })) {
    throw NetError{"net design identifier is malformed"};
  }
  std::string name(end - field.begin(), '\0');
  std::copy(field.begin(), end, name.begin());
  if (name.empty() || std::any_of(name.begin(), name.end(), [](char character) {
        return character < '!' || character > '~';
      })) {
    throw NetError{"net design identifier is malformed"};
  }
  return name;
}

template <std::size_t Size>
void require_knots(const std::array<std::uint8_t, Size>& knots,
                   const std::array<std::uint8_t, Size>& expected, std::string_view label) {
  if (knots != expected) {
    std::string text;
    for (const std::uint8_t knot : expected) {
      text += text.empty() ? "" : ", ";
      text += std::to_string(knot);
    }
    throw NetError{"net " + std::string{label} + " knots must be " + text + " in design " +
                   std::string{kPatternGainDesign}};
  }
}

[[nodiscard]] bool valid_key(std::string_view key) noexcept {
  return !key.empty() && key.front() >= 'a' && key.front() <= 'z' &&
         std::all_of(key.begin(), key.end(), [](char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= '0' && character <= '9') || character == '_';
         });
}

[[nodiscard]] bool valid_value(std::string_view value) noexcept {
  return std::none_of(value.begin(), value.end(), [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return byte < 0x20 || byte == 0x7f;
  });
}

// Metadata is UTF-8 text of "key=value" lines, each ending in a newline, with keys strictly
// ascending so that one set of entries has exactly one encoding.
[[nodiscard]] NetMetadata parse_metadata(std::span<const std::uint8_t> bytes) {
  const std::string_view text{reinterpret_cast<const char*>(bytes.data()), bytes.size()};
  NetMetadata entries;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t end = text.find('\n', begin);
    if (end == std::string_view::npos) {
      throw NetError{"net metadata must end each line with a newline"};
    }
    const std::string_view line = text.substr(begin, end - begin);
    const std::size_t separator = line.find('=');
    if (separator == std::string_view::npos) {
      throw NetError{"net metadata line has no '=': " + std::string{line}};
    }
    const std::string_view key = line.substr(0, separator);
    const std::string_view value = line.substr(separator + 1);
    if (!valid_key(key) || !valid_value(value)) {
      throw NetError{"net metadata line is malformed: " + std::string{line}};
    }
    if (!entries.empty() && entries.back().first >= key) {
      throw NetError{"net metadata keys must be unique and in ascending order"};
    }
    entries.emplace_back(std::string{key}, std::string{value});
    begin = end + 1;
  }
  const auto name = std::find_if(entries.begin(), entries.end(),
                                 [](const auto& entry) { return entry.first == "name"; });
  if (name == entries.end() || name->second.empty()) {
    throw NetError{"net metadata has no name"};
  }
  return entries;
}

[[nodiscard]] std::string serialize_metadata(const NetMetadata& metadata) {
  std::string text;
  for (const auto& [key, value] : metadata) {
    text += key;
    text += '=';
    text += value;
    text += '\n';
  }
  return text;
}

[[nodiscard]] PatternGainParameters read_pattern_gain(Reader& reader) {
  PatternGainParameters parameters;
  for (std::uint8_t& knot : parameters.line_knots) {
    knot = reader.u8();
  }
  for (std::uint8_t& knot : parameters.gain_knots) {
    knot = reader.u8();
  }
  for (std::int32_t& value : parameters.intercept) {
    value = reader.i32();
  }
  for (std::int16_t& value : parameters.line_weights) {
    value = reader.i16();
  }
  for (std::int16_t& value : parameters.gain_weights) {
    value = reader.i16();
  }
  return parameters;
}

void validate_pattern_gain(const PatternGainParameters& parameters) {
  require_knots(parameters.line_knots, kPatternGainLineKnots, "line");
  require_knots(parameters.gain_knots, kPatternGainKnots, "gain");
  if (std::any_of(parameters.intercept.begin(), parameters.intercept.end(), [](std::int32_t value) {
        return value < -kMaximumIntercept || value > kMaximumIntercept;
      })) {
    throw NetError{"net intercepts must lie within +/-2^24"};
  }
}

[[nodiscard]] PatternGainTables make_tables(const PatternGainParameters& parameters) {
  validate_pattern_gain(parameters);
  PatternGainTables tables;
  for (std::size_t pattern = 0; pattern < tables.raw_line_weights.size(); ++pattern) {
    const std::size_t orbit = pattern_gain::kRawToReversalOrbit[pattern];
    for (std::size_t knot = 0; knot < kPatternGainLineKnotCount; ++knot) {
      tables.raw_line_weights[pattern][knot] =
          parameters.line_weights[knot * kPatternGainOrbitCount + orbit];
    }
  }
  tables.intercept = parameters.intercept;
  tables.gain_weights = parameters.gain_weights;
  return tables;
}

[[nodiscard]] std::string metadata_lookup(const NetMetadata& metadata, std::string_view key) {
  const auto entry = std::find_if(metadata.begin(), metadata.end(),
                                  [key](const auto& item) { return item.first == key; });
  return entry == metadata.end() ? std::string{} : entry->second;
}

[[nodiscard]] bool is_sha256_hex(std::string_view text) noexcept {
  return text.size() == 64 && std::all_of(text.begin(), text.end(), [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] PatternGainParameters compiled_header_parameters() {
  namespace header = frozen_pattern_gain_model;
  static_assert(header::kFractionalBits == kPatternGainFractionalBits);
  static_assert(header::kReversalOrbitCount == kPatternGainOrbitCount);
  static_assert(header::kGainFeatureCount == kPatternGainFeatureCount);
  PatternGainParameters parameters;
  std::copy(header::kLineKnots.begin(), header::kLineKnots.end(), parameters.line_knots.begin());
  std::copy(header::kGainKnots.begin(), header::kGainKnots.end(), parameters.gain_knots.begin());
  std::copy(header::kIntercept.begin(), header::kIntercept.end(), parameters.intercept.begin());
  std::copy(header::kLineWeights.begin(), header::kLineWeights.end(),
            parameters.line_weights.begin());
  std::copy(header::kGainWeights.begin(), header::kGainWeights.end(),
            parameters.gain_weights.begin());
  return parameters;
}

}  // namespace

Net::Net(std::string sha256, NetMetadata metadata, const PatternGainParameters& parameters)
    : sha256_(std::move(sha256)),
      metadata_(std::move(metadata)),
      parameters_(parameters),
      tables_(make_tables(parameters)) {
  const std::string header = metadata_lookup(metadata_, "header_sha256");
  if (!header.empty() && !is_sha256_hex(header)) {
    throw NetError{"net metadata header_sha256 must be a SHA-256 hex digest"};
  }
  id_ = (header.empty() ? sha256_ : header).substr(0, 12);
}

std::string_view Net::name() const noexcept { return metadata_value("name"); }

std::string_view Net::metadata_value(std::string_view key) const noexcept {
  const auto entry = std::find_if(metadata_.begin(), metadata_.end(),
                                  [key](const auto& item) { return item.first == key; });
  return entry == metadata_.end() ? std::string_view{} : std::string_view{entry->second};
}

std::span<const std::string_view> supported_net_designs() noexcept { return kSupportedDesigns; }

std::shared_ptr<const Net> parse_net(std::span<const std::uint8_t> bytes) {
  if (bytes.size() < kMagic.size() || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
    throw NetError{"not a PoE2 net file"};
  }
  Reader reader{bytes};
  static_cast<void>(reader.take(kMagic.size()));
  const std::uint32_t version = reader.u32();
  if (version != kContainerVersion) {
    throw NetError{"net file uses container format " + std::to_string(version) +
                   "; this engine reads format " + std::to_string(kContainerVersion)};
  }
  const std::uint32_t header_bytes = reader.u32();
  if (bytes.size() < header_bytes + kTrailerBytes || header_bytes < kFixedHeaderBytes + 4) {
    throw NetError{"net file is truncated"};
  }
  const std::span<const std::uint8_t> body = bytes.first(bytes.size() - kTrailerBytes);
  const labeling::Sha256Digest digest = labeling::sha256(body);
  if (!std::equal(digest.begin(), digest.end(), bytes.end() - kTrailerBytes)) {
    throw NetError{"net file is damaged: its integrity check does not match its contents"};
  }

  const std::string design = design_name(reader.take(kDesignBytes));
  if (std::find(kSupportedDesigns.begin(), kSupportedDesigns.end(), design) ==
      kSupportedDesigns.end()) {
    throw NetError{"net design \"" + design + "\" is not supported by this engine (supported: " +
                   supported_design_list() + ")"};
  }
  const std::uint32_t fractional_bits = reader.u32();
  const std::uint32_t table_count = reader.u32();
  if (table_count > kMaximumTables ||
      header_bytes != kFixedHeaderBytes + (std::size_t{4} * table_count) + 4) {
    throw NetError{"net file header is malformed"};
  }
  std::vector<std::uint32_t> table_sizes(table_count);
  for (std::uint32_t& size : table_sizes) {
    size = reader.u32();
  }
  const std::uint32_t metadata_bytes = reader.u32();
  if (fractional_bits != static_cast<std::uint32_t>(kPatternGainFractionalBits)) {
    throw NetError{"net design " + design + " requires " +
                   std::to_string(kPatternGainFractionalBits) + " fractional bits; the file has " +
                   std::to_string(fractional_bits)};
  }
  if (!std::ranges::equal(table_sizes, kPatternGainTableSizes)) {
    throw NetError{"net table sizes do not match design " + design};
  }
  if (metadata_bytes > kMaximumMetadataBytes) {
    throw NetError{"net metadata is larger than " + std::to_string(kMaximumMetadataBytes) +
                   " bytes"};
  }

  Reader body_reader{body};
  static_cast<void>(body_reader.take(header_bytes));
  const PatternGainParameters parameters = read_pattern_gain(body_reader);
  NetMetadata metadata = parse_metadata(body_reader.take(metadata_bytes));
  if (body_reader.offset() != body.size()) {
    throw NetError{"net file has unexpected bytes after its metadata"};
  }
  return std::make_shared<const Net>(labeling::sha256_text(labeling::sha256(bytes)),
                                     std::move(metadata), parameters);
}

std::vector<std::uint8_t> serialize_net(std::string_view design, const NetMetadata& metadata,
                                        const PatternGainParameters& parameters) {
  if (design.empty() || design.size() > kDesignBytes) {
    throw NetError{"net design identifier must be 1 to 32 bytes"};
  }
  const std::string metadata_text = serialize_metadata(metadata);
  Writer writer;
  writer.bytes(kMagic);
  writer.u32(kContainerVersion);
  writer.u32(static_cast<std::uint32_t>(kFixedHeaderBytes + 4 * kPatternGainTableSizes.size() + 4));
  std::array<std::uint8_t, kDesignBytes> design_field{};
  std::copy(design.begin(), design.end(), design_field.begin());
  writer.bytes(design_field);
  writer.u32(static_cast<std::uint32_t>(kPatternGainFractionalBits));
  writer.u32(static_cast<std::uint32_t>(kPatternGainTableSizes.size()));
  for (const std::uint32_t size : kPatternGainTableSizes) {
    writer.u32(size);
  }
  writer.u32(static_cast<std::uint32_t>(metadata_text.size()));
  for (const std::uint8_t knot : parameters.line_knots) {
    writer.u8(knot);
  }
  for (const std::uint8_t knot : parameters.gain_knots) {
    writer.u8(knot);
  }
  for (const std::int32_t value : parameters.intercept) {
    writer.i32(value);
  }
  for (const std::int16_t value : parameters.line_weights) {
    writer.i16(value);
  }
  for (const std::int16_t value : parameters.gain_weights) {
    writer.i16(value);
  }
  writer.bytes(
      std::span{reinterpret_cast<const std::uint8_t*>(metadata_text.data()), metadata_text.size()});
  const labeling::Sha256Digest digest = labeling::sha256(writer.view());
  writer.bytes(digest);
  return writer.take();
}

const std::shared_ptr<const Net>& default_net() {
  static const std::shared_ptr<const Net> net = std::make_shared<const Net>(
      std::string{kModelHeaderSha256}, NetMetadata{{"name", "compiled-header"}},
      compiled_header_parameters());
  return net;
}

}  // namespace poe2::minimax
