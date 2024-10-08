/*!
 * Copyright 2019 XGBoost contributors
 */
#ifndef XGBOOST_COMMON_VERSION_H_
#define XGBOOST_COMMON_VERSION_H_

#include <dmlc/io.h>
#include <string>
#include <tuple>

#include "boxhed_kernel/base.h"

namespace boxhed_kernel {
class Json;
// a static class for handling version info
struct Version {
  using TripletT = std::tuple<XGBoostVersionT, XGBoostVersionT, XGBoostVersionT>;
  static const TripletT kInvalid;

  // Save/Load version info to Json document
  static TripletT Load(Json const& in);
  static void Save(Json* out);

  // Save/Load version info to dmlc::Stream
  static Version::TripletT Load(dmlc::Stream* fi);
  static void Save(dmlc::Stream* fo);

  static std::string String(TripletT const& version);
  static TripletT Self();

  static bool Same(TripletT const& triplet);
};

}      // namespace boxhed_kernel
#endif  // XGBOOST_COMMON_VERSION_H_
