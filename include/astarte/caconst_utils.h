#ifndef _ASTARTE_CACONST_UTILS_H
#define _ASTARTE_CACONST_UTILS_H

#include "astarte/caconst.h"
#include <string>

namespace astarte {

std::string get_operator_type_name(OperatorType type);

size_t data_type_size(DataType type);

#define INT4_NUM_OF_ELEMENTS_PER_GROUP 32

size_t get_quantization_to_bte_size(DataType type,
                                    DataType quantization_type,
                                    size_t num_elements);

std::ostream &operator<<(std::ostream &, OperatorType);

}; // namespace astarte

#endif // _ASTARTE_CACONST_UTILS_H