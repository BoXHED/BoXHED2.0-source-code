/*!
 * Copyright 2015-2020 by Contributors
 * \file gbm.cc
 * \brief Registry of gradient boosters.
 */
#include <dmlc/registry.h>
#include <string>
#include <vector>
#include <memory>

#include "boxhed_kernel/gbm.h"
#include "boxhed_kernel/learner.h"
#include "boxhed_kernel/generic_parameters.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::boxhed_kernel::GradientBoosterReg);
}  // namespace dmlc

namespace boxhed_kernel {
GradientBooster* GradientBooster::Create(
    const std::string& name,
    GenericParameter const* generic_param,
    LearnerModelParam const* learner_model_param) {
  auto *e = ::dmlc::Registry< ::boxhed_kernel::GradientBoosterReg>::Get()->Find(name);
  if (e == nullptr) {
    LOG(FATAL) << "Unknown gbm type " << name;
  }
  auto p_bst =  (e->body)(learner_model_param);
  p_bst->generic_param_ = generic_param;
  return p_bst;
}
}  // namespace boxhed_kernel

namespace boxhed_kernel {
namespace gbm {
// List of files that will be force linked in static links.
DMLC_REGISTRY_LINK_TAG(gblinear);
DMLC_REGISTRY_LINK_TAG(gbtree);
}  // namespace gbm
}  // namespace boxhed_kernel
