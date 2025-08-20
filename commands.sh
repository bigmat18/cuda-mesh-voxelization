#!/bin/bash

all() {
    set -e

    BUILD_TYPES=("Debug" "Release" "RelWithDebInfo")
    ROOT_DIR="build"
    
    mkdir -p "$ROOT_DIR"
    
    for TYPE in "${BUILD_TYPES[@]}"; do
        DIR="$ROOT_DIR/$TYPE"
        echo "==> Configuring $TYPE in $DIR"
        cmake -G "Ninja" -B "$DIR" -DCMAKE_BUILD_TYPE=$TYPE .
    done
    
    cd build
    ln -s Debug/compile_commands.json compile_commands.json
    cd ..
}

debug() {
    cmake --build build/Debug
}

release() {
    cmake --build build/Release
}

reldeb() {
    cmake --build build/RelWithDebInfo
}

connect() {
    ssh -i /home/bigmat18/Utils/hpc/hpc-machine mgiuntoni@131.114.51.113
}

clean() {
    rm -rf build/ out/
}

help() {
    echo "Usage: $0 {all|debug|release|reldeb|connect|clean}"
}

case "$1" in
    all) all ;;
    debug) debug ;;
    release) release ;;
    reldeb) reldeb ;;
    connect) connect ;;
    clean) clean ;;
    *) help ;;
esac
