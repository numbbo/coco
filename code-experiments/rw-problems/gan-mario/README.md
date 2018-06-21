# Requirements for the `gan-mario` test suite

## Python packages

- `pytorch 0.3.1`
- `torchvision 0.2.0` or lower

Install using `pip` or `conda`:

````
pip install pytorch==0.3.1
pip install torchvision==0.2.0
````

````
conda install pytorch=0.3.1
conda install torchvision=0.2.0
````

If the right versions are not available, download the `whl` files for your system at [https://pytorch.org/previous-versions/](https://pytorch.org/previous-versions/)

Another alternative (for the installation on Windows without CUDA):

````
conda install -c peterjc123 pytorch-cpu
pip install torchvision==0.2.0
````

## Java

Java is required to perform evaluation through simulations. 
