#ifndef _CA_TYPE_H
#define _CA_TYPE_H

#include "astarte/caconst.h"
#include <cstddef>

namespace astarte {

class LayerID {
public:
    static const LayerID NO_ID;
    LayerID();
    LayerID(size_t id, size_t transformer_layer_id);
    bool is_valid_id() const;
    friend bool operator==(LayerID const &lhs, LayerID const &rhs);

public:
    size_t id, transformer_layer_id;
};

}; // namespace astarte

#endif // _CA_TYPE_H