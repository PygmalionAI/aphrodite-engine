#pragma once
#include "astarte/batch_config.h"
#include <string>
#include <vector>

namespace astarte {

struct GenerationConfig {
    bool do_sample = false;
    float temperature = 0.75,
    float topp = 0.6,
    GenerationConfig(bool _do_sample, float _temperature, float _topp) {
        temperature = _temperature > 0 ? _temperature : temperature;
        topp = _topp > 0 ? _topp : topp;
        do_sample = _do_sample;
    }
    GenerationConfig() {}
};

struct GenerationResult {
    using RequestGuid = BatchConfig::RequestGuid;
    using TokenID = BatchConfig::TokenId;
    RequestGuid guid;
    std::string input_text;
    std::string output_text;
    std::vector<TokenId> input_tokens;
    std::vector<TokenId> output_tokens;
};

#include <string>
#include <vector>

std::string join_path(std::vector<std::string> const &paths);

} // namespace astarte