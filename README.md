<p align="center"><img src="docs\imgs\DeepAerialMapper_Logo.jpg" width="480px"></p>

<p style="font-weight: bold; text-align:center;">DeepAerialMapper is an open-source project to convert segmentation mask into High-definition map (<i>HD map</i>).</p> <br>


## Introduction

The creation of HD maps is crucial for the success of highly automated vehicles, but it faces obstacles such as the need for updated road data and accuracy in mapping. The conventional methods of obtaining road data through dedicated measurement vehicles or crowd-sourced data from series-production vehicles are costly. On the other hand, high-resolution aerial imagery is cost-effective or even free, but requires extensive manual work to create maps. To overcome this, we propose a semi-automatic approach to creating HD maps using high-resolution aerial imagery. This method involves training neural networks to semantically segment aerial images and obtain a prototypical HD map of the road elements. The map can then be easily modified for specific use cases through common tools, which is made possible by exporting the map in the [lanelet2](https://www.mrt.kit.edu/z/publ/download/2018/Poggenhans2018Lanelet2.pdf) format. This efficient method has the potential to become a scalable solution for HD map creation.


This work is based on our *paper*. You can find arXiv version of the paper here [TODO: ADD link]. </br>

## Architecture
The algorithm consists of 4 steps.
1. Extracting lanelets (road contours, lane)
2. Grouping lanelets
3. Classifying symbols
4. Generating hdmap in lanelet2 format using the output from step 2 and 3.

## Tutorial

You can find the detailed tutorial [HERE](docs/tutorial.md)

### Get Started

1. Install
```bash
conda create --name DAM python=3.10
conda activate DAM
git clone https://github.com/RobertKrajewski/DeepAerialMapper.git DeepAerialMapper
cd DeepAerialMapper
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

2. Run DAM with demo sample
```bash
python3 tools/create_maps.py
```

3. Check the output directory: `results/maps/{date_time}`
