XGBoost Plugin Example
======================
This folder provides an example of implementing boxhed_kernel plugin.

There are three steps you need to do to add a plugin to boxhed_kernel
- Create your source .cc file, implement a new extension
  - In this example [custom_obj.cc](custom_obj.cc)
- Register this extension to boxhed_kernel via a registration macro
  - In this example ```XGBOOST_REGISTER_OBJECTIVE``` in [this line](custom_obj.cc#L78)
- Add a line to `boxhed_kernel/plugin/CMakeLists.txt`:
```
target_sources(objboxhed_kernel PRIVATE ${boxhed_kernel_SOURCE_DIR}/plugin/example/custom_obj.cc)
```

Then you can test this plugin by using ```objective=mylogistic``` parameter.

<!--  LocalWords:  XGBoost
 -->
