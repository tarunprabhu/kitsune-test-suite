#ifndef KITSUNE_TEST_SUITE_TIMING_H
#define KITSUNE_TEST_SUITE_TIMING_H

// This contains some common code to setup timers and report kernel execution
// statistics in the Kitsune/Benchmarks directory of the kitsune test suite.

#ifdef __cplusplus

#include <chrono>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#else // ! __cplusplus

#include <stdint.h>

#endif // __cplusplus

/// Execution time statistics for some region of code.
typedef struct Stats {
  /// The number of times the region of code was entered.
  uint64_t entries;

  /// The total execution time, in seconds, of the region. The mean execution
  /// time will simply be this divided by @ref entries.
  uint64_t total;

  /// The minimum execution time, in microseconds, of any one invocation.
  uint64_t min;

  /// The maximum execution time, in microseconds, of any one invocation.
  uint64_t max;
} Stats;

#ifdef __cplusplus

using Clock = std::chrono::steady_clock;
using Microseconds = std::chrono::microseconds;

/// A simple timer class that is intended for timing and to keep track of
/// statistics. Each instance of this class is intended to record the execution
/// time of some  named region. The start() method should be called just before
/// the region of interest in the code is entered (which, in the case of the
/// benchmarks in the kitsune test suite is typically a forall loop). The stop()
/// method should be called at the end of the region, typically immediately
/// after a forall loop. Nested invocations are not allowed i.e. If start() is
/// called after another call to start() but before any call to stop(), it will
/// be ignored.
class Timer {
private:
  /// The name of the region with which this timer is associated (typically the
  /// name of a kernel).
  std::string name;

  /// The time point registered when the timer was started. This will be
  /// std::nullopt if the timer has not been started.
  std::optional<std::chrono::time_point<Clock>> tick = std::nullopt;

  /// Execution time statistics
  Stats stats;

public:
  Timer(const std::string &name) : name(name) {
    stats.entries = 0;
    stats.total = 0;
    stats.min = std::numeric_limits<decltype(stats.min)>::max();
    stats.max = 0;
  }

  /// Start the timer. If the timer has already been started, this has no
  /// o effect.
  void start() {
    if (tick)
      return;
    tick = Clock::now();
  }

  /// Stop the timer. If the timer has not been started, this will have no
  /// effect. Otherwise, statistics will be recorded, and the number of entries
  /// will be incremented by one. This will return the duration, in
  /// microseconds, since the last call to start, or 0 if the timer has not
  /// been started.
  uint64_t stop() {
    // Stop the clock so we don't capture anything that this method specifically
    // does.
    std::chrono::time_point<Clock> tock = Clock::now();
    if (!tick)
      return 0;

    uint64_t us =
        std::chrono::duration_cast<Microseconds>(tock - *tick).count();

    stats.entries += 1;
    stats.total += us;
    if (us < stats.min)
      stats.min = us;
    if (us > stats.max)
      stats.max = us;
    tick = std::nullopt;

    return us;
  }

  /// Get the total time recorded by this timer.
  uint64_t total() const { return stats.total; }

  /// Get the total number of entries recorded by this timer.
  uint64_t entries() const { return stats.entries; }

  /// Get the minimum time, in microseconds, recorded by this timer.
  uint64_t min() const { return stats.min; }

  /// Get the maximum time, in microseconds, recorded by this timer.
  uint64_t max() const { return stats.max; }

  /// Get the mean time, in microseconds (rounded down to the nearest
  /// microsecond), recorded by this timer
  uint64_t mean() const { return stats.total / stats.entries; }

  /// Print the statistics for this region in JSON format to the given output
  /// stream.
  std::ostream &json(std::ostream &os) const {
    os << "  \"" << name << "\": {" << std::endl;
    os << "    \"entries\": " << stats.entries << "," << std::endl;
    os << "    \"min\": " << stats.min << "," << std::endl;
    os << "    \"max\": " << stats.max << "," << std::endl;
    os << "    \"mean\": " << (stats.total / stats.entries) << "," << std::endl;
    os << "    \"total\": " << stats.total << std::endl;
    os << "  }";
    return os;
  }
};

/// Print statistics for a number of timers to the given output stream. This
/// will print the statistics in JSON format with some sentinels on either
/// side. A post-processing script can scrape the results by looking for the
/// sentinels. The name is usually the name of the benchmark.
static std::ostream &json(std::ostream &os, const std::vector<Timer> &timers) {
  os << "<json>" << std::endl;
  os << "{" << std::endl;
  if (timers.size()) {
    timers.front().json(os);
    for (size_t i = 1; i < timers.size(); ++i) {
      os << "," << std::endl;
      timers.at(i).json(os);
    }
  }
  os << std::endl << "}" << std::endl;
  os << "</json>" << std::endl;
  return os;
}

#else // !__cplusplus

/// A simple timer class that is intended for timing and to keep track of
/// statistics. Each instance of this class is intended to record the execution
/// time of some  named region. The start() method should be called just before
/// the region of interest in the code is entered (which, in the case of the
/// benchmarks in the kitsune test suite is typically a forall loop). The stop()
/// method should be called at the end of the region, typically immediately
/// after a forall loop. Nested invocations are not allowed i.e. If start() is
/// called after another call to start() but before any call to stop(), it will
/// be ignored.
typedef struct Timer {
  /// The name of the region with which this timer is associated (typically the
  /// name of a kernel).
  const char *name;

  /// Execution time statistics
  Stats stats;
} Timer;

// TODO: Implement timing for C

#endif // __cplusplus

#endif // KITSUNE_TEST_SUITE_TIMING_H
