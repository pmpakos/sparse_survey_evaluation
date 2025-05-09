# This has to be run the first time that FusedMM is cloned somewhere. After that, it is ready to use.

cd ./FusedMM
git reset --hard; git clean -fd

# First, set the PATH variable accordingly to use specific gcc
export PATH=/various/dgal/gcc/gcc-12.2.0/gcc_bin/bin:$PATH

# After that, change a small typo in the repo
sed -i 's/sLIBs/sLIBS/g' kernels/CONFIG/make.base

# Then run configure and go back to the root directory
./configure 

cd ../
echo "FusedMM is configured. Now you can run make to build the library."

cat << 'EOF'
If you want to compile for "double" precision (64-bits) instead of "float" (32-bits), 
there are some changes that need to take place in the produced Makefile (at the root directory of FusedMM)
First, precision will change from s(ingle) to d(ouble)
      pre=s -> pre=d
Also, length of vectorization needs to be halfed. 
In AMD EPYC where 256-bits vectorization takes place (AVX2), it is 8 for 32-bits precision and 4 for 64-bits precision:
      vlen=8 -> vlen=4
EOF
