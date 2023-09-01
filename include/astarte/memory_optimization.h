#ifndef _ASTARTE_MEMORY_OPTIMIZATION_H_
#define _ASTARTE_MEMORY_OPTIMIZATION_H_

#include <cassert>
#include <string>

namespace astarte {

enum class MemoryUsageType {
    GLOBAL,
    PER_DEVICE_MAX,
};

enum class MemorySearchAlgo {
    MULTI_OBJECTIVE,
};

class MemoryOptimConfig {
public:
    MemoryUsageType mem_usage_type;
    MemorySearchAlgo mem_search_algo;
    float run_time_cost_factor;

    MemoryOptimConfig()
        : mem_usage_type{MemoryUsageType::GLOBAL},
          mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
          run_time_cost_factor{0.5} {}
    MemoryOptimConfig(float factor)
        : mem_usage_type{MemoryUsageType::GLOBAL},
          mem_search_algo{MemorySearchAlgo::MULTI_OBJECTIVE},
          run_time_cost_factor{factor} {}
};

class MemorySearchResult {
public:
    float run_time_cost{};
    float memory_cost{};
    float search_time{};
    float max_per_device_mem_all_devices = 0.0;
};

namespace PCG {

class MemoryUsage {
public:
    MemoryUsageType usage_type;
    float num;

    MemoryUsage() : usage_type{MemoryUsageType::GLOBAL}, num{0.0} {}
    MemoryUsage(MemoryUsageType _usage_type, float _num)
        : usage_type{_usage_type}, num{_num} {}

    std::string to_string() const;

    MemoryUsage &operator+=(MemoryUsage const &rhs);

    friend MemoryUsage operator+(MemoryUsage lhs, MemoryUsage const &rhs);

    friend std::ostream &operator<<(std::ostream &s, MemoryUsage const &usage);
};


} // namespace PCG

} // namespace astarte
#endif // _ASTARTE_MEMORY_OPTIMIZATION_H_