## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

2. This project uses the **MIMIC-III** and **MIMIC-IV** datasets. Please download these datasets from the following links:  
- [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)  
- [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) 

After downloading, place the datasets under the folder ```./dataset/```. You will need to select the variables you wish to use from the datasets before proceeding.

3. Run the Code

> To execute the primary algorithm, use:
```
python run.py
```

> To execute the ensemble algorithm, use:
```
python run_ensemble.py

```
> To calculate SAPS II scores, run
```
python run_score.py

```
> To run machine learning models, use:
```
python python run_ML.py
```


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.
- tsai (https://github.com/timeseriesAI/tsai)

## Contact

If you have any questions or want to use the code, feel free to contact:

* Yong Liu (lf.liu@siat.ac.cn)
