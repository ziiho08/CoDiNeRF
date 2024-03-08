# CoDiNeRF


## Installation
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). 
Then, clone this repository and run the commands:
```
git clone https://github.com/ziiho08/CoDiNeRF.git
conda activate nerfstudio
cd CoDiNeRF/
pip install -e .
ns-install-cli
```

## Training the CoDiNeRF
To train the CoDiNeRF, run the command:
```
ns-train codinerf --data [PATH]
```