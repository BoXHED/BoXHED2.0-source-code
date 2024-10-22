/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_USE_CUDA

#include <boxhed_kernel/data.h>

// dummy implementation of EllpackPage in case CUDA is not used
namespace boxhed_kernel {

class EllpackPageImpl {};

EllpackPage::EllpackPage() = default;

EllpackPage::EllpackPage(DMatrix* dmat, const BatchParam& param) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}

EllpackPage::~EllpackPage() {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}

void EllpackPage::SetBaseRowId(size_t row_id) {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
}
size_t EllpackPage::Size() const {
  LOG(FATAL) << "Internal Error: XGBoost is not compiled with CUDA but "
                "EllpackPage is required";
  return 0;
}

}  // namespace boxhed_kernel

#endif  // XGBOOST_USE_CUDA
