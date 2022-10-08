## Installation
TODO: run pipreqs and generate requirements.txt and check if it works in a new virtual env.  (when an independent github page is created, and it works.)

## Tutorial
1. Prepare segmentation mask 
2. Define the palette of each mask in `mask.py`
```
palette_map = {
    SemanticClass.BLACK: [0, 0, 0],  # black
    SemanticClass.ROAD: [128, 128, 128],  # gray
    SemanticClass.SIDEWALK: [0, 0, 255],  # blue
    SemanticClass.TRAFFICISLAND: [153, 51, 255],  # purple
    SemanticClass.PARKING: [255, 255, 0],  # yellow
    SemanticClass.VEGETATION: [0, 255, 0],  # green
    SemanticClass.LANEMARKING: [0, 128, 128],  # cyan
    SemanticClass.SYMBOL: [255, 0, 0],  # red
}
```
<img src="imgs\mask.png" width="640"/>

3. Run DeepAerialMapper algorithm
`python3 test/create_map.py --input {directory of masks} --output {directory to save JSON file}`
TODO: check the function location.

## Hyper Parameters
Performance of extracting and grouping lanelets depends on several parameters. \
TODO: Keep every hyperparameter into one config file. --> User can edit. [Include palette_map also into config file?]