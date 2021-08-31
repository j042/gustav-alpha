# gustav-alpha
Complete snapshots of gustav project developments.
Last update: 2021.08.31

## Setup guide using `conda`
### Info:
`conda` is recommended, but it is possible to setup without conda - use `pip` and fufill `SplineLib`'s requirements.  
If you'd like to give `conda` a try, you could visit: https://docs.conda.io/en/latest/miniconda.html

#### Installation on RWTH Compute Cluster
```bash
mkdir miniconda3
cd miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -u
```
The script will prompt you for the directory, where `miniconda` is supposed to
be installed and will per default suggest the current directory. The `-u` option
is required for installing into an already existing directory.

### Setup (local and on RWTH Cluster)
Clone and get ready.
```bash
git clone git@github.com:j042/gustav.git
cd gustav
export GUS=$PWD
```

Create a new environment and install packages  
__pip__:  
_Note: `pip` tends to work better than conda in our case_
```bash
conda create -n gus python=3.9         # Tested with 3.7, 3.8 and 3.9
conda activate gus
pip install cython numpy scipy vedo matplotlib "meshio[all]" optimesh
```

__conda__:  
_Note: still needs some `pip`. `vedo` tends to fail, if installed using conda_
```bash
conda create -n gus python=3.9
conda activate gus
conda install -c anaconda cython
conda install -c anaconda numpy
conda install -c anaconda scipy
conda install -c conda-forge vedo  
conda install -c conda-forge matplotlib
conda install -c conda-forge meshio
pip install optimesh
```

Totally __OPTIONAL__, there's only one function that returns `trimesh` for your convenience:
```bash
conda install -c conda-forge scikit-image shapely rtree pyembree
pip install trimesh[all]
```

`SplineLib` Dependency:
_`SplineLib` requires cmake-3.19 or higher. If your system does not fulfill the requirements, you could install requirements using `conda` for your `gus` environment:_  
```bash
conda install -c conda-forge cmake      # installs cmake-3.20 (last checked: 2021.06.25)
```

Install third party modules:  
_Note: If you want to work on the library, replace `python3 setup.py install` with `python3 setup.py develop` to make your life easier._  

```bash
# the following two commands are only required on the RWTH Cluster
###########################
module switch intel gcc
module load cmake
###########################

cd third_party/triangle
python3 setup.py install

cd $GUS
cd third_party/tetgen
python3 setup.py install

cd $GUS
cd third_party/splinelibpy
python3 setup.py install

cd $GUS
python3 setup.py develop
```

## Setup on RWTH Compute Cluster without `conda`

Load the required modules
```bash
module switch intel gcc/10
module load python/3.8.7 cmake
```

Extend the `PYTHONPATH` variable with the directory where the additional
packages will be installed
```bash
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.8/site-packages
```

Install the required python dependencies
```bash
pip install cython numpy scipy vedo matplotlib "meshio[all]" optimesh
```

Install all required third party software
```bash
cd third_party/triangle
python3 setup.py install --prefix $HOME/.local/

cd ../tetgen
python3 setup.py install --prefix $HOME/.local/

cd ../splinelibpy
python3 setup.py install --prefix $HOME/.local/
```

Install the software
```bash
cd $GUS
python3 setup.py develop --prefix $HOME/.local/
```
