/*!
 * Copyright 2015 by Contributors
 * \file objective.cc
 * \brief Registry of all objective functions.
 */
#include <boxhed_kernel/objective.h>
#include <dmlc/registry.h>

#include <sstream>

#include "boxhed_kernel/host_device_vector.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::boxhed_kernel::ObjFunctionReg);
}  // namespace dmlc

namespace boxhed_kernel {
// implement factory functions
ObjFunction* ObjFunction::Create(const std::string& name, GenericParameter const* tparam) {
  auto *e = ::dmlc::Registry< ::boxhed_kernel::ObjFunctionReg>::Get()->Find(name);
  if (e == nullptr) {
    std::stringstream ss;
    for (const auto& entry : ::dmlc::Registry< ::boxhed_kernel::ObjFunctionReg>::List()) {
      ss << "Objective candidate: " << entry->name << "\n";
    }
    LOG(FATAL) << "Unknown objective function: `" << name << "`\n"
               << ss.str();
  }
  auto pobj = (e->body)();
  pobj->tparam_ = tparam;
  return pobj;
}

}  // namespace boxhed_kernel

namespace boxhed_kernel {
namespace obj {
// List of files that will be force linked in static links.
#ifdef XGBOOST_USE_CUDA
DMLC_REGISTRY_LINK_TAG(regression_obj_gpu);
DMLC_REGISTRY_LINK_TAG(hinge_obj_gpu);
DMLC_REGISTRY_LINK_TAG(multiclass_obj_gpu);
DMLC_REGISTRY_LINK_TAG(rank_obj_gpu);
#else
DMLC_REGISTRY_LINK_TAG(regression_obj);
DMLC_REGISTRY_LINK_TAG(hinge_obj);
DMLC_REGISTRY_LINK_TAG(multiclass_obj);
DMLC_REGISTRY_LINK_TAG(rank_obj);
#endif  // XGBOOST_USE_CUDA
}  // namespace obj
}  // namespace boxhed_kernel
