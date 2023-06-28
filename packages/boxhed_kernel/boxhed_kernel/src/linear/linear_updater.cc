/*!
 * Copyright 2018
 */
#include <boxhed_kernel/linear_updater.h>
#include <dmlc/registry.h>
#include "./param.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::boxhed_kernel::LinearUpdaterReg);
}  // namespace dmlc

namespace boxhed_kernel {

LinearUpdater* LinearUpdater::Create(const std::string& name, GenericParameter const* lparam) {
  auto *e = ::dmlc::Registry< ::boxhed_kernel::LinearUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown linear updater " << name;
  }
  auto p_linear = (e->body)();
  p_linear->learner_param_ = lparam;
  return p_linear;
}

}  // namespace boxhed_kernel

namespace boxhed_kernel {
namespace linear {
DMLC_REGISTER_PARAMETER(LinearTrainParam);

// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_shotgun);
DMLC_REGISTRY_LINK_TAG(updater_coordinate);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_coordinate);
#endif  // XGBOOST_USE_CUDA
}  // namespace linear
}  // namespace boxhed_kernel
