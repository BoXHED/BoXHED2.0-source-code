/*!
 * Copyright 2015 by Contributors
 * \file tree_updater.cc
 * \brief Registry of tree updaters.
 */
#include <dmlc/registry.h>

#include "boxhed_kernel/tree_updater.h"
#include "boxhed_kernel/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::boxhed_kernel::TreeUpdaterReg);
}  // namespace dmlc

namespace boxhed_kernel {

TreeUpdater* TreeUpdater::Create(const std::string& name, GenericParameter const* tparam) {
  auto *e = ::dmlc::Registry< ::boxhed_kernel::TreeUpdaterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown tree updater " << name;
  }
  auto p_updater = (e->body)();
  p_updater->tparam_ = tparam;
  return p_updater;
}

}  // namespace boxhed_kernel

namespace boxhed_kernel {
namespace tree {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(updater_colmaker);
DMLC_REGISTRY_LINK_TAG(updater_refresh);
DMLC_REGISTRY_LINK_TAG(updater_prune);
DMLC_REGISTRY_LINK_TAG(updater_quantile_hist);
DMLC_REGISTRY_LINK_TAG(updater_histmaker);
DMLC_REGISTRY_LINK_TAG(updater_sync);
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(updater_gpu_hist);
#endif  // XGBOOST_USE_CUDA
}  // namespace tree
}  // namespace boxhed_kernel
