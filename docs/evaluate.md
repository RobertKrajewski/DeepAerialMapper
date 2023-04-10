# Evaluation

For the evaluation of the created maps, we compare the lanemarkings derived from the semantic masks against manually annotated lanemarkings.
For details, check [our paper](TODO).

To reproduce the results of the paper, you can run the evaluation locally (see below).
We also execute the evaluation as a Github action on every new commit. 
Select the `evaluate` step in the latest action you can find [in the actions tab](https://github.com/RobertKrajewski/DeepAerialMapper/actions).

## Evaluation on ground truth semantic masks

```bash
python3 bin/create_maps.py --output-dir=results/maps/paper_groundtruth configs/paper_groundtruth.yaml 
python3 bin/evaluate_lanemarkings.py data/paper/lane_annotations results/maps/paper_groundtruth
```
Output of `bin/evaluate_lanemarkings.py`:
```
...
| INFO     | __main__:evaluate_lanemarkings:84 - Precision: 0.97, Recall: 0.97, RMSE: 1.95
```
## Evaluation on neural network generated semantic masks
```bash
python3 bin/create_maps.py --output-dir=results/maps/paper_prediction configs/paper_prediction.yaml
python3 bin/evaluate_lanemarkings.py data/paper/lane_annotations results/maps/paper_prediction
```
Output of `bin/evaluate_lanemarkings.py`:
```
...
| INFO     | __main__:evaluate_lanemarkings:84 - Precision: 0.96, Recall: 0.96, RMSE: 3.98
```

## Evaluate new maps

To evaluate new maps, create annotations using the open-source [VIA annotation tool V2](https://www.robots.ox.ac.uk/~vgg/software/via/via.html).
Create polygon annotations for every lanemarking and road border, and export the results for every image as a separate `.json` file.