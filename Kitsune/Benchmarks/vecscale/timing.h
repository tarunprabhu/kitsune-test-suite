#ifndef KITSUNE_TEST_SUITE_TIMING_H
#define KITSUNE_TEST_SUITE_TIMING_H

// This contains some common code to setup timers and report kernel execution
// statistics in the Kitsune/Benchmarks directory of the kitsune test suite.

#ifdef __cplusplus

#include <chrono>
#include <cstdint>
#include <memory>
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
  uint64_t count;

  /// The total execution time, in seconds, of the region. The mean execution
  /// time will simply be this divided by @ref count.
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
  /// name of a kernel). The name is used when printing the timer statistics in
  /// JSON format. Ideally, this should not have any spaces in it, but that is
  /// not required.
  std::string mName;

  /// A long name for the timer. This is used when pretty-printing the timer
  /// group in which this timer is contained. This is generally intended for
  /// human consumption, so it may be more descriptive than the name. If a label
  /// is not provided when constructing the timer, this will be the same as the
  /// name.
  std::string mLabel;

  /// The time point registered when the timer was started. This will be
  /// std::nullopt if the timer has not been started.
  std::optional<std::chrono::time_point<Clock>> tick = std::nullopt;

  /// Execution time statistics
  Stats stats;

public:
  /// Create a timer with the given name and an optional label. If a label is
  /// not provided, the label of the timer will be the same as the name.
  Timer(const std::string &name, const std::string &label = "")
      : mName(name), mLabel(label) {
    if (not label.size())
      mLabel = name;
    stats.count = 0;
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

    stats.count += 1;
    stats.total += us;
    if (us < stats.min)
      stats.min = us;
    if (us > stats.max)
      stats.max = us;
    tick = std::nullopt;

    return us;
  }

  /// Get the name of the timer.
  const std::string &name() const { return mName; }

  /// Get the timer label.
  const std::string &label() const { return mLabel; }

  /// Get the total time recorded by this timer.
  uint64_t total() const { return stats.total; }

  /// Get the total number of entries recorded by this timer.
  uint64_t count() const { return stats.count; }

  /// Get the minimum time, in microseconds, recorded by this timer.
  uint64_t min() const { return stats.min; }

  /// Get the maximum time, in microseconds, recorded by this timer.
  uint64_t max() const { return stats.max; }

  /// Get the mean time, in microseconds (rounded down to the nearest
  /// microsecond), recorded by this timer
  uint64_t mean() const { return stats.total / stats.count; }

  /// Print the statistics for this region in JSON format to the given output
  /// stream.
  std::ostream &json(std::ostream &os) const {
    os << "  \"" << name() << "\": {" << std::endl;
    os << "    \"count\": " << stats.count << "," << std::endl;
    os << "    \"min\": " << stats.min << "," << std::endl;
    os << "    \"max\": " << stats.max << "," << std::endl;
    os << "    \"mean\": " << (stats.total / stats.count) << "," << std::endl;
    os << "    \"total\": " << stats.total << std::endl;
    os << "  }";
    return os;
  }

  // Convert the given value in microseconds to seconds.
  static std::string secs(uint64_t us) {
    // Ideally, we should use a stringstream here, but the kitsune.h header
    // redefines sync which causes a conflict with streambuf in GCC. If the
    // issue in kitsune.h is fixed, should be able to switch to doing this in
    // the C++ way.
    char buf[16];
    snprintf(buf, 16, "%8.4f secs", float(us) / 1000000.0);
    return buf;
  }
};

/// A timer group that manages an ordered set of timers.
class TimerGroup {
private:
  /// The name of the timer group.
  std::string name;

  /// The timers in the group.
  std::vector<std::unique_ptr<Timer>> timers;

public:
  TimerGroup(const std::string &name) : name(name) {}
  TimerGroup(const TimerGroup &) = delete;
  TimerGroup(TimerGroup &&) = delete;
  TimerGroup &operator=(const TimerGroup &) = delete;
  TimerGroup &operator=(TimerGroup &&) = delete;

  /// Create a new timer with the given name and label. This will create a timer
  /// with the name even if one already exists in the group.
  Timer &add(const std::string &name, const std::string &label = "") {
    timers.emplace_back(new Timer(name, label));
    return *timers.back();
  }

  /// Get the timer with the given name. If more than one timer with the name
  /// was added to the group, the first timer will always be returned. It is an
  /// error to call this function with a name that does not exist in the group.
  Timer &get(const std::string &name) {
    for (const std::unique_ptr<Timer> &timer : timers)
      if (timer->name() == name)
        return *timer;
    std::cerr << "No such timer: '" << name << "'\n";
    std::abort();
  }

  /// Print statistics for the timers in the group to the given output stream.
  /// This will print the statistics in JSON format with some sentinels on
  /// either side. A post-processing script can scrape the results by looking
  /// for the sentinels. The name is usually the name of the benchmark.
  std::ostream &json(std::ostream &os) const {
    os << "<json>" << std::endl;
    os << "{" << std::endl;
    if (timers.size()) {
      timers.front()->json(os);
      for (size_t i = 1; i < timers.size(); ++i) {
        os << "," << std::endl;
        timers.at(i)->json(os);
      }
    }
    os << std::endl << "}" << std::endl;
    os << "</json>" << std::endl;
    return os;
  }

  /// Print the times for each timer in this region in a human-friendly format
  /// to the given output stream.
  std::ostream &prettyTimes(std::ostream &os, unsigned indent = 0) const {
    unsigned width = 0;
    for (const std::unique_ptr<Timer> &pTimer : timers)
      if (pTimer->label().size() > width)
        width = pTimer->label().size();

    char buf[80];
    for (const std::unique_ptr<Timer> &pTimer : timers) {
      snprintf(buf, 80, "%*s%-*s :  %s", indent, "", width,
               pTimer->label().c_str(), Timer::secs(pTimer->total()).c_str());
      os << buf << "\n";
    }
    return os;
  }
};

static std::ostream &json(std::ostream &os, std::vector<Timer> timers) {
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
/// statistics. Each instance of this class is intended to record the
/// execution time of some  named region. The start() method should be
/// called just before the region of interest in the code is entered (which,
/// in the case of the benchmarks in the kitsune test suite is typically a
/// forall loop). The stop() method should be called at the end of the
/// region, typically immediately after a forall loop. Nested invocations
/// are not allowed i.e. If start() is called after another call to start()
/// but before any call to stop(), it will be ignored.
typedef struct Timer {
  /// The name of the region with which this timer is associated (typically
  /// the name of a kernel).
  const char *name;

  /// Execution time statistics
  Stats stats;
} Timer;

// TODO: Implement timing for C

#endif // __cplusplus

#endif // KITSUNE_TEST_SUITE_TIMING_H
