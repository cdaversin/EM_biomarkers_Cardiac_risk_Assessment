# simcardems_scripts

This folder contains Python scripts to run cardiac electro-mechanical simulations with CiPA and several CCBs drugs.
To run the scripts, you need to have the `simcardems` version used in the manuscript installed (see instructions in simcardems/README.txt).

To install the version of simcardems that was used to perform simulations presented in the paper (source code in simcardems directory) :
```
# In EM_biomarkers_Cardiac_risk_Assessment directory:
docker run -it --name simcardems -v $(pwd):/home ghcr.io/scientificcomputing/fenics-gmsh:2023-11-15
cd /home
cd simcardems
python3 -m pip install .
```

To run one population model, choose one drug and one model of the population.
Five seconds under drug effects are run in three consecutive simulations: 1-2s, 3-4s and 5s:

```
cd simcardems_scripts
export popu_file="inputPopulation/PoMcontrol_m1/input_params.json"
export drug_file="drug_factors/CiPA/Astemizole_10_timesFPC.json"

python3 run_simcardems_current_CV_beat1_2.py $popu_file $drug_file
python3 run_simcardems_current_CV_beat3_4.py $popu_file $drug_file
python3 run_simcardems_current_CV_beat5.py $popu_file $drug_file
```
