## Installation
``` bash
conda create --name DAM python=3.10
conda activate DAM
git clone https://github.com/RobertKrajewski/DeepAerialMapper.git DeepAerialMapper
cd DeepAerialMapper
pip install -v -e .
```

## Tutorial
1. Prepare segmentation mask 
<img src="../data/seg_masks/demo.png" width="640" alt="demo mask" title="Demo mask image"/>

2. Define the configuration of segmentation masks in `configs/config.yaml` \
Select the corresponding mask color of class and GPS meta data of each image.
``` yaml
palette:
  - type: BLACK
    color: [0, 0, 0]
  - type: ROAD
    color: [128, 128, 128]
  - type: VEGETATION
    color: [0, 255, 0]
  - type: TRAFFICISLAND
    color: [153, 51, 255]
  - type: SIDEWALK
    color: [0, 0, 255]
  - type: PARKING
    color: [255, 255, 0]
  - type: SYMBOL
    color: [255, 0, 0]
  - type: LANEMARKING
    color: [0, 128, 128]
  # type: object name
  # color: RGB masking color of the object

ignore:
  # define the reigon you want to ignore in the mask

meta:
  'demo':
      origin:
          - 292161.665
          - 5630314.6051
      scale:
          - 0.05000433957147767
          - 0.050011118737163816
      proj: 'epsg:25832'
      width: 3687
      height: 1457
  # meta of GPS image

```

3. Run DeepAerialMapper algorithm
``` bash
python3 tools/create_maps.py input data/seg_masks
```