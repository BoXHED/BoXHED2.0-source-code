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

# is linux
[[ "$OSTYPE" == "linux-gnu"* ]] ; is_linux=$?
is_linux=$((1-$is_linux))


use_gpu=false
while getopts ":gv:" opt; do
    case $opt in
        g)
            use_gpu=true
            ;;
        v)
            str=""
            if [[ $is_windows == 0 ]]; then
                echo "ERROR: The -$opt option should only be used in Windows when specifying the Visual Studio version."
                exit 1
            fi
            case $OPTARG in
                14)
                    str+="\"Visual Studio 14 2015 Win64\""
                    ;;
                15)
                    str+="\"Visual Studio 15 2017\" -A x64"
                    ;;
                16)
                    str+="\"Visual Studio 16 2019\" -A x64"
                    ;;
                *)
                    echo "ERROR: Invalid VS version after the -$opt option. It can be either 14, 15, or 16." >&2
                    exit 1
                    ;;
            esac
            cmake_args+=(-G${str})
            ;;
        \?)
            echo "ERROR: Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "ERROR: Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [ "$use_gpu" == true ]
then
    if [[ $is_linux == 0 ]] ; then
        echo "Error: CUDA usage is available for Linux users only at the moment." >&2
        exit 1
    fi
    cmake_args+=(-DUSE_CUDA=ON)
else
    cmake_args+=(-DUSE_CUDA=OFF)
fi
            
rm -f setup_log

echo "This installation relies heavily on https://xgboost.readthedocs.io/en/latest/build.html"

print "creating build directory for boxhed2.0 in ${DIR}/boxhed.kernel/"
cd "${DIR}/boxhed.kernel/"
mkdir -p build &> ${setup_log}
check_success

if [[ $is_mac == 1 ]]; then
    print "installing libomp for MacOS"
    brew install libomp >> ${setup_log} 2>&1
    check_success
fi

print "running cmake for boxhed in ${DIR}/boxhed.kernel/build/"
cd "${DIR}/boxhed.kernel/build/"
cmake ${cmake_args[@]} >> ${setup_log} 2>&1
check_success

print "running make for boxhed in ${DIR}/boxhed.kernel/build/"
cd "${DIR}/boxhed.kernel/build/"
build_args=(
    --build .
)
if [[ $is_windows == 1 ]]; then
    build_args+=(--config Release)
fi
cmake ${build_args[@]}  >> ${setup_log} 2>&1
check_success

print "setting up boxhed for python in ${DIR}/boxhed.kernel/python-package/"
cd "${DIR}/boxhed.kernel/python-package/"
python setup.py install >> ${setup_log} 2>&1
check_success


print "boxhed installed successfully"

####### setting up preprocessing #######

print "creating build directory for preprocessor in ${DIR}/build/"
cd "${DIR}"
mkdir -p build >> ${setup_log} 2>&1
check_success

print "running cmake for preprocessor in ${DIR}/build/"
cd "${DIR}/build/"
cmake ../preprocessor_installer  >> ${setup_log} 2>&1
check_success

print "running cmake --build for preprocessor in ${DIR}/build/"
cmake --build .  >> ${setup_log} 2>&1
check_success


print "preprocessor installed successfully"
