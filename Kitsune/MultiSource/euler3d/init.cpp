#include <kitsune.h>

#include "../../SingleSource/euler3d/euler3d.h"

void initialize_variables(int nelr, float *[[kitsune::mobile]] variables,
                          float *[[kitsune::mobile]] ff_variable) {
  forall(int i = 0; i < nelr; i++) {
    forall (int j = 0; j < NVAR; j++)
      variables[i + j * nelr] = ff_variable[j];
  }
}
