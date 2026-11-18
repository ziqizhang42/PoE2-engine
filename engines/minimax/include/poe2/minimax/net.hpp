#ifndef POE2_MINIMAX_NET_HPP
#define POE2_MINIMAX_NET_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace poe2::minimax {

class NetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline constexpr std::string_view kPatternGainDesign = "pattern-gain-v1";

inline constexpr std::size_t kPatternGainPlyCount = 50;
inline constexpr std::size_t kPatternGainLineKnotCount = 3;
inline constexpr std::size_t kPatternGainOrbitCount = 1716;
inline constexpr std::size_t kPatternGainKnotCount = 5;
inline constexpr std::size_t kPatternGainFeatureCount = 19;
inline constexpr std::size_t kPatternGainRawPatternCount = 3276;
// The design fixes its phase knots, so interpolation divides by constants. Files still state
// them, and a net with other knots is a different design.
inline constexpr std::array<std::uint8_t, kPatternGainLineKnotCount> kPatternGainLineKnots{
    {0, 28, 49}};
inline constexpr std::array<std::uint8_t, kPatternGainKnotCount> kPatternGainKnots{
    {0, 12, 24, 36, 49}};

// The evaluator's lookup layout. A net fills it once when loaded; search reads it unchanged.
// A weight here is the interpolation numerator at one ply: the weights of the two knots around
// the ply, each scaled by the ply's distance to the other knot. An evaluation therefore sums
// one row per residual and divides the sum by the knot spacing, which gives exactly the value
// of interpolating the two knots' sums.
struct PatternGainTables {
  std::array<std::array<std::int32_t, kPatternGainRawPatternCount>, kPatternGainPlyCount>
      line_weights{};
  std::array<std::int32_t, kPatternGainPlyCount> intercept{};
  std::array<std::array<std::int32_t, kPatternGainFeatureCount>, kPatternGainPlyCount>
      gain_weights{};
};

struct PatternGainParameters {
  std::array<std::uint8_t, kPatternGainLineKnotCount> line_knots{};
  std::array<std::uint8_t, kPatternGainKnotCount> gain_knots{};
  std::array<std::int32_t, kPatternGainPlyCount> intercept{};
  // Knot-major: weight [knot * kPatternGainOrbitCount + orbit].
  std::array<std::int16_t, kPatternGainLineKnotCount * kPatternGainOrbitCount> line_weights{};
  // Knot-major: weight [knot * kPatternGainFeatureCount + feature].
  std::array<std::int16_t, kPatternGainKnotCount * kPatternGainFeatureCount> gain_weights{};
};

using NetMetadata = std::vector<std::pair<std::string, std::string>>;

class Net final {
 public:
  Net(std::string sha256, NetMetadata metadata, const PatternGainParameters& parameters);

  [[nodiscard]] std::string_view design() const noexcept { return kPatternGainDesign; }
  [[nodiscard]] std::string_view sha256() const noexcept { return sha256_; }
  // The first 12 hex digits of the identity hash: the file hash, or for a net converted from a
  // C++ header the header's hash recorded as `header_sha256`, which earlier records used.
  [[nodiscard]] std::string_view id() const noexcept { return id_; }
  [[nodiscard]] std::string_view name() const noexcept;
  [[nodiscard]] std::string_view metadata_value(std::string_view key) const noexcept;
  [[nodiscard]] const NetMetadata& metadata() const noexcept { return metadata_; }
  [[nodiscard]] const PatternGainParameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] const PatternGainTables& pattern_gain() const noexcept { return tables_; }

 private:
  std::string sha256_;
  std::string id_;
  NetMetadata metadata_;
  PatternGainParameters parameters_;
  PatternGainTables tables_;
};

[[nodiscard]] std::span<const std::string_view> supported_net_designs() noexcept;

// Parses and validates a complete net file, or returns a reason the user can act on. It does not
// throw for a bad file, so builds that cannot catch exceptions (the browser package) can use it.
[[nodiscard]] std::expected<std::shared_ptr<const Net>, std::string> try_parse_net(
    std::span<const std::uint8_t> bytes);
// The same, throwing NetError with the reason.
[[nodiscard]] std::shared_ptr<const Net> parse_net(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::shared_ptr<const Net> load_net_file(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::uint8_t> serialize_net(std::string_view design,
                                                      const NetMetadata& metadata,
                                                      const PatternGainParameters& parameters);

// The net built into this engine, used whenever no file is given.
[[nodiscard]] const std::shared_ptr<const Net>& default_net();

}  // namespace poe2::minimax

#endif  // POE2_MINIMAX_NET_HPP
