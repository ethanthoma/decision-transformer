mkdir -p cuda
ln -sf /usr/lib64/libcuda.so.1 cuda/libcuda.so.1
ln -sf /usr/lib64/libnvidia-ptxjitcompiler.so cuda/libnvidia-ptxjitcompiler.so.1
ln -sf /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/12.2.2/lib/libnvrtc.so.12 cuda/libnvrtc.so.12
ln -sf /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/12.2.2/lib/libnvrtc-builtins.so cuda/libnvrtc-builtins.so.12.2

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/scratch/st-gerope-1/ethoma/decision-transformer/cuda/"
