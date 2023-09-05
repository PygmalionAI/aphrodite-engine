#ifndef _ASTARTE_PARALLEL_OPS_PARALLEL_OP_INFO_H
#define _ASTARTE_PARALLEL_OPS_PARALLEL_OP_INFO_H

#include "astarte/caconst.h"

namespace astarte {

struct ParallelOpInfo {
    friend void swap(ParallelOpInfo &, ParallelOpInfo &);

    OperatorType op_type;
    int parallel_dim;
    int parallel_degree;
};
bool operator==(ParallelOpInfo const &, ParallelOpInfo const &);
} // namespace astarte


#endif // _ASTARTE_PARALLEL_OPS_PARALLEL_OP_INFO_H