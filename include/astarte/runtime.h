#ifndef _ASTARTE_RUNTIME_H_
#define _ASTARTE_RUNTIME_H_

#include "config.h"

namespace astarte {

class CARuntime {
public:
    CARuntime(CAConfig &config);
    CAHandler handlers[MAX_NUM_WORKERS];
};


} // namespace astarte

#endif // _ASTARTE_RUNTIME_H_