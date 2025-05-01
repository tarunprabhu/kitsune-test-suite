#include <kitsune.h>

#include "../../Benchmarks/euler3d/common.h"

void initialize_variables(int nelr, kitsune::mobile_ptr<float> variables,
                          kitsune::mobile_ptr<float> ff_variable) {
  forall(int i = 0; i < nelr; i++) {
    for (int j = 0; j < NVAR; j++)
      variables[i + j * nelr] = ff_variable[j];
  }
}
