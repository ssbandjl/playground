#include <pybind11/pybind11.h>
#include "cache.h"
#include <torch/torch.h>
#include <iostream>

PYBIND11_MODULE(gpu_cpu_benchmark, m) {
  m.def("copy_dma", &copy_dma);
  m.def("copy_custom_kernel", &copy_custom_kernel);
}
