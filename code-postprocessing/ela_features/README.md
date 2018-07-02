# Requirements for the the `ela` tool

## To execute
- `R`
- `Rscript`

## R packages
- `stringr`
- `flacco`
- `RANN`
- `e1071`

Install using in `R console`

````
install.packages("stringr")
install.packages("flacco")
install.packages("RANN")
install.packages("e1071")
````

#Usage

````
cd code-postprocessing/ela_features
Rscript read_rw_data.R <path-to-coco-output-dir> <data-file-name>
Rscript compute_ela.R <data-file-name> <feature-file-name>
````

Usage example if you have results for suite rw-gan-mario
````
cd code-postprocessing/ela_features
Rscript read_rw_data.R ../../code-experiments/build/c/exdata/rw-gan-mario marioData.RData
Rscript compute_ela.R marioData.RData marioFeatures.RData

````


