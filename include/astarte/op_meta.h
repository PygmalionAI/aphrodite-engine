#ifndef _OP_META_H
#define _OP_META_H

#include "astarte/config.h"

namespace astarte {

class Op;

class OpMeta {
public:
    OpMeta(CAHandler _handle);
    OpMeta(CAHandler _handle, Op const *op);

public:
    CAHandler handle;
    bool profiling;
    bool trainableInputs[MAX_NUM_INPUTS];
    DataType input_type[MAX_NUM_INPUTS];
    DataType weight_type[MAX_NUM_WEIGHTS];
    DataType output_type[MAX_NUM_WEIGHTS];
};

}; // namespace astarte
#endif // _OP_META_H