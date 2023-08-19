#pragma once

#include "astarte/caconst.h"
#include "legion.h"
#include <cstddef>
#include <cstdlib>

namespace astarte {

class InferenceResult;
class beamInferenceResult;

using BatchConfigFuture = Legion::Future;
using InferenceResultFuture = Legion::Future;
using BeamSearchBatchConfigFuture = Legion::Future;
using TreeVerifyBatchConfigFuture = Legion::Future;
using beamInferenceResultFuture = Legion::Future;

class BatchConfig {
public:
    using RequestGuid = size_t;
    using TokenId = int;
    BatchConfig();
    int num_active_requests() const;
    int num_active_tokens() const;
    void print() const;
    virtual InferenceMode get_mode() const;
    static BatchConfig const *from_future(BatchConfigFuture const &future);
    static int const MAX_NUM_REQUESTS = 1;
    static int const MAX_NUM_TOKENS = 64;
    static int const MAX_PROMPT_LENGTH = 62;
    static int const MAX_SEQ_LENGTH = 256;

    // these are set by update
    int num_tokens;

    struct PerRequestInfo {
        int token_offset_offset;
        int num_tokens_in_batch;
        int max_sequence_length;
        RequestGuid request_guid;
    };
    struct PerTokenInfo {
        int abs_depth_in_request;
        int request_index;
        TokenId token_id;
    };
    PerRequestInfo requestInfo[MAX_NUM_REQUESTS];
    PerTokenInfo tokensInfo[MAX_NUM_REQUESTS];

    bool request_completed[MAX_NUM_REQUESTS];
};

class TreeVerifyBatchConfig : public BatchConfig {
public:
    TreeVerifyBatchConfig();
    ~TreeVerifyBatchConfig();
    InferenceMode get_mode() const;
    void print() const;
    struct CommittedTokensInfo {
        int token_index;
        int request_index;
        int token_depth;
    };

    int num_tokens_to_commit;
    CommittedTokensInfo committed_tokens[MAX_NUM_TOKENS];
};

struct InferenceResult {
    static int const MAX_NUM_TOKENS = BatchConfig::MAX_NUM_TOKENS;
    BatchConfig::TokenId token_ids[MAX_NUM_TOKENS];
};

class BeamSearchBatchConfig : public BatchConfig {
public:    
    BeamSearchBatchConfig();
    BeamSearchBatchConfig(int model_id);
    BeamSearchBatchConfig(size_t beam_width, size_t target_iterations);
    BeamSearchBatchConfig(BeamSearchBatchConfig const &other, int model_id);
    InferenceMode get_mode() const;

    ~BeamSearchBatchConfig();

    void print() const;
    bool done() const;
    int max_beam_depth_all_requests() const;
    int current_depth_all_requests() const;
    
    size_t beam_width;
    size_t target_iterations;
    inline static int const MAX_BEAM_WIDTH = 1;
    inline static int const MAX_BEAM_DEPTH = 8;

    int model_id;
    int max_init_length = 0;

    struct BeamSearchPerRequestInfo {
        int beam_size;
        int current_depth = -1;
        int max_depth = MAX_BEAM_DEPTH;

        BatchConfig::TokenId tokens[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
        float probs[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
        int parent_id[BeamSearchBatchConfig::MAX_BEAM_WIDTH];
    };

    struct BeamSearchPerTokenInfo {
        int sub_request_index;
    };

    BeamSearchPerRequestInfo beamRequestsInfo[MAX_NUM_REQUESTS];
    BeamSearchPerTokenInfo beamTokenInfo[MAX_NUM_TOKENS * MAX_BEAM_WIDTH];
    int sub_requests[MAX_NUM_REQUESTS * MAX_BEAM_WIDTH];

private:
    size_t current_iteration;
};

struct BeamInferenceResult {
    static int const MAX_NUM_TOKENS = BatchConfig::MAX_NUM_TOKENS;
    BatchConfig::TokenId
        token_ids[MAX_NUM_TOKENS * BeamSearchBatchConfig::MAX_BEAM_WIDTH];
    float probs[MAX_NUM_TOKENS * BeamSearchBatchConfig::MAX_BEAM_WIDTH];
    int parent_id[MAX_NUM_TOKENS * BeamSearchBatchConfig::MAX_BEAM_WIDTH];
};

}; // namespace astarte