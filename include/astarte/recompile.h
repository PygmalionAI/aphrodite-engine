#ifndef _ASTARTE_RECOMPILE_H_
#define _ASTARTE_RECOMPILE_H_

#include "legion.h"
#include <functional>

namespace astarte {

class CAModel;

class RecompileState {
public:
    RecompileState(std::function<bool(CAModel *)> _trigger_func,
                   std::function<void(CAModel *)> _alter_func,
                   CAModel *_ff);
    bool trigger();
    void alter();

public:
    int recompilations;

private:
    std::function<bool(CAModel *)> trigger_func;
    std::function<void(CAModel *)> alter_func;
    CAModel *ff;
};


}; // namespace astarte

#endif // _ASTARTE_RECOMPILE_H_