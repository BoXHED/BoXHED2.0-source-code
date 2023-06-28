function (write_version)
  message(STATUS "boxhed_kernel VERSION: ${boxhed_kernel_VERSION}")
  configure_file(
    ${boxhed_kernel_SOURCE_DIR}/cmake/version_config.h.in
    ${boxhed_kernel_SOURCE_DIR}/include/boxhed_kernel/version_config.h @ONLY)
  configure_file(
    ${boxhed_kernel_SOURCE_DIR}/cmake/Python_version.in
    ${boxhed_kernel_SOURCE_DIR}/python-package/boxhed_kernel/VERSION @ONLY)
endfunction (write_version)
