<p align="center"><img src="docs\imgs\DeepAerialMapper_Logo.jpg" width="480px"></p>

<p style="font-weight: bold; text-align:center;">DeepAerialMapper is an open-source project to convert segmentation mask into High-definition map (<i>HD map</i>).</p> <br>


## Introduction

HD maps are essential for the development, safety validation and operation of highly automated vehicles. Central challenges in the creation of HD maps are the efficient collection of up-to-date data of roads and deriving of accurate maps from the recorded sensor data. 
Typically used approaches including dedicated measurement vehicles and crowd-sourced data from series-production vehicles are mostly only commercially viable. High-resolution aerial imagery is an alternative that is available cheaply or even free of charge, but requires significant time and effort to be turned into maps manually. We therefore present an approach for the semi-automatic creation of HD maps from high-resolution aerial imagery. For this purpose, we train neural networks to semantically segment aerial images into HD map-relevant classes. The resulting segmentation is hierarchically post-processed to obtain a prototypical HD map of the visible road elements. By exporting the map to the [lanelet2](https://www.mrt.kit.edu/z/publ/download/2018/Poggenhans2018Lanelet2.pdf) format, it can be easily extended for the intended use cases by common tools. The time savings resulting from this show the potential of the proposed method as a scalable solution to create HD maps.

This work is based on our *paper*. You can find arXiv version of the paper here [TODO: ADD link]. </br>

## Architecture
The algorithm consists of 4 parts.
1. extracting lanelets (road contours, lane)
2. grouping lanelets
3. classifying symbols
4. generating hdmap in lanelet2 format.

## Tutorial

You can find the detailed tutorial [HERE](docs/tutorial.md)

### Get Started

0. Requirements
- Python 3
- conda 

1. Install
```
conda create --name DAM python=3.10
conda activate DAM
git clone https://github.com/RobertKrajewski/DeepAerialMapper.git DeepAerialMapper
cd DeepAerialMapper
python3 setup.py install
```

2. Verify the installation
```
TODO: make a simple demo
python3 test/demo.py docs/imgs/demo-mask.png work_dirs/demo
```

If you can see the output `work_dirs/demo`, it's ready to use!
