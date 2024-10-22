#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
setup_log="${DIR}/setup_log.txt"
cmake_args=(
    ..
    )

check_success () {
    if [ $? -ne 0 ]; then
        echo " -- unsuccessful, check ${setup_log}"
        exit 1
    fi
}

print () {
    to_echo="~ ~ ~ > $1 ..."
    echo $to_echo >> ${setup_log}
    echo $to_echo
}

# is windows
[[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] ; is_windows=$?
is_windows=$((1-$is_windows))

# is mac
[[ "$OSTYPE" == "darwin"* ]] ; is_mac=$?
is_mac=$((1-$is_mac))

build_dirs=(
    "${DIR}/packages/boxhed/boxhed.egg-info/"
    "${DIR}/packages/boxhed/build/"
    #"${DIR}/packages/boxhed_kernel/boxhed_kernel/build/"
    "${DIR}/packages/boxhed_kernel/boxhed_kernel/lib/"
    "${DIR}/packages/boxhed_kernel/boxhed_kernel/python-package/build/"
    "${DIR}/packages/boxhed_kernel/boxhed_kernel/python-package/xgboost.egg-info/"
    "${DIR}/packages/boxhed_prep/build/"
    "${DIR}/packages/boxhed_prep/boxhed_prep.egg-info/") 

for dir in ${build_dirs[@]}; 
do
    rm -rf $dir
done

            
rm -f setup_log

echo "This installation relies heavily on https://xgboost.readthedocs.io/en/latest/build.html"

print "creating build directory for boxhed2.0 in ${DIR}/packages/boxhed_kernel/"
cd "${DIR}/packages/boxhed_kernel/boxhed_kernel/"
mkdir -p build &> ${setup_log}
check_success

if [[ $is_mac == 1 ]]; then
    print "installing libomp for MacOS"
    brew install libomp >> ${setup_log} 2>&1
    check_success
fi

print "running cmake for boxhed in ${DIR}/packages/boxhed_kernel/boxhed_kernel/build/"
cd "${DIR}/packages/boxhed_kernel/boxhed_kernel/build/"

cmake_args+=(-DUSE_CUDA=ON)
# CMake arguments may be passed like the following. It has the general
# form of "-D _ARGUMENT_=VALUE". A common problem that may arise is
# CMake not being able to find CUDA compiler. In case this happens you
# may uncomment following line and replace the address place holder 
# with your true CUDA compiler address. (The compiler address should look
# like this example: /home/grads/j/j.doe/cuda-11.1/bin/nvcc)
#cmake_args+=(-D CMAKE_CUDA_COMPILER=__ADDRESS_TO_CUDA_COMPILER__)
cmake ${cmake_args[@]} >> ${setup_log} 2>&1
check_success

print "running make for boxhed in ${DIR}/packages/boxhed_kernel/boxhed_kernel/build/"
cd "${DIR}/packages/boxhed_kernel/boxhed_kernel/build/"
build_args=(
    --build .
)
if [[ $is_windows == 1 ]]; then
    build_args+=(--config Release)
fi
cmake ${build_args[@]}  >> ${setup_log} 2>&1
check_success

print "setting up boxhed for python in ${DIR}/packages/boxhed_kernel/boxhed_kernel/python-package/"
cd "${DIR}/packages/boxhed_kernel/boxhed_kernel/python-package/"
python setup.py install >> ${setup_log} 2>&1
check_success


print "boxhed installed successfully"

####### setting up preprocessing #######

print "installing preprocessor"
cd "${DIR}/packages/"
pip install ./boxhed_prep


####### setting up boxhed #######

print "installing BoXHED"
cd "${DIR}/packages/"
pip install ./boxhed

