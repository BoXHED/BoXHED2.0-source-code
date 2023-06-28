/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA
#include <dmlc/base.h>
#if DMLC_ENABLE_STD_THREAD

#include "ellpack_page_source.h"
#include <boxhed_kernel/data.h>
namespace boxhed_kernel {
namespace data {

EllpackPageSource::EllpackPageSource(DMatrix* dmat,
                                     const std::string& cache_info,
                                     const BatchParam& param) noexcept(false) {
  LOG(FATAL)
      << "Internal Error: "
         "XGBoost is not compiled with CUDA but EllpackPageSource is required";
}

}  // namespace data
}  // namespace boxhed_kernel
#endif  // DMLC_ENABLE_STD_THREAD
#endif  // XGBOOST_USE_CUDA
