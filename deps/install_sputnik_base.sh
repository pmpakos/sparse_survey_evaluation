# This has to be run the first time that sputnik is cloned somewhere. After that, it is ready to use.

cd ./sputnik
git reset --hard; git clean -fd

# First, set the PATH variable accordingly to use specific gcc and cmake
export PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/bin:$PATH
export PATH=/various/pmpakos/epyc5_libs/cmake-3.26.0-rc3/build/bin/:$PATH

mkdir build
cd build
export PWD=`pwd`
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DCUDA_ARCHS="80" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="80" -DCMAKE_INSTALL_PREFIX=${PWD}
make -j
make install -j

# After that, the sputnik library is ready to be used