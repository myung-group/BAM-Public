<!-- #region -->
### BAM
Bayesian Atoms Modeling

### Installation (NIPA)
Create a conda environment with python.
```shell
conda create -n $env_name python==($version) 
ex) python==3.12.2
```
Install a `cuda-tookit` that suits your GPU and environment.
```shell
conda install cuda-toolkit==($version)  
ex) cuda-toolkit==12.2.2
```
Install a `cudnn`.
```shell
conda install -c anaconda cudnn
```
Install a `jaxlib` and `nequip-jax`.
```shel
pip install --upgrade pip
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
pip install git+https://github.com/mariogeiger/nequip-jax
```

install a BAM package
```shell
pip install -e .
```

#### If the cuda module is supported, or the cuda-toolkit is not installed, proceed with the method below. (frodo, olaf)
Create a conda environment with python.
```shell
conda create -n env_name python==($version) # ex) python==3.12.2
```
Load a cuda module.
```shell
module load $cuda 
ex)cuda/cuda-12.3
ex)cudatoolkit/12.2
```
Install a cudnn that suits your CUDA version and add the path of cudnn/lib to `$LD_LIBRARY_PATH`.
`
ex) download cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
ex) export LD_LIBRARY_PATH=/$cudnn_PATH/cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib:$LD_LIBRARY_PATH
`
Install a jaxlib and nequip-jax.
```shel
pip install --upgrade pip
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/mariogeiger/nequip-jax
```

install a BAM package
```shell
pip install -e .
```






