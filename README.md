To install JAX with GPU support, find your CUDA version and include it in jaxlib==0.1.57+{CUDA version}

```
pip install --upgrade pip
pip install --upgrade numpy tqdm
pip install --upgrade jax jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
