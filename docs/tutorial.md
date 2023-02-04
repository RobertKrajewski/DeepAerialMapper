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

2. Define the configuration of segmentation masks in the same directory of your segmentation mask i.e. `data/seg_masks` \
Select the corresponding mask color of class and GPS meta data of each image.

- `config.yaml`

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

symbol_detector:
  symbols:
    - Left
    - Left_Right
    - Right
    - Straight
    - Straight_Left
    - Straight_Right
    - Unknown
  weight_filepath: configs/symbol_weights.pt
```
- `meta.yaml`
```yaml
demo:
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
- `ignore_regions.json` \
You can set up irrelvant reigon of your map. It helps algorithm create lanelet of relevant region.
```json
{
    "demo.png105905": {
        "filename": "demo.png",
        "size": 105905,
        "regions": [
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        1217,
                        1538,
                        1583,
                        1577,
                        1418
                    ],
                    "all_points_y": [
                        647,
                        897,
                        847,
                        781,
                        673
                    ]
                },
                "region_attributes": {}
            },
            {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [
                        1874,
                        1838,
                        1801,
                        1722,
                        1771,
                        1835,
                        1876
                    ],
                    "all_points_y": [
                        493,
                        525,
                        553,
                        600,
                        624,
                        547,
                        499
                    ]
                },
                "region_attributes": {}
            },
          ...
        ],
        "file_attributes": {}
    }
}
```

3. Run DeepAerialMapper algorithm
``` bash
python3 tools/create_maps.py
```