# GreenMove

![](https://img.shields.io/badge/python-3.8-green)

![image](https://github.com/yuki-feng0307/GreenMove/blob/main/img/fig1.png)



## Requirements

These dependencies can be installed using the following command:

- Python 3.8.18
- xgboost==2.0.3
- pandas==2.0.3
- numpy==1.24.4
- scikit-learn==1.3.2
- shap==0.42.0
- matplotlib==3.7.5

These dependencies can be installed using the following command:

```
pip install -r requirements.txt
```



## File Description

The repository contains the code for network construction, visualization and GBM.

#### Network construction

Although the datasets obtained from platforms such as Anjuke, Baidu Maps, and the Shanghai Big Data Center cannot be made publicly accessible, the construction process of the GreenMove network can still be understood from the code we developed here.

#### Network visualization

Corresponds to Figure2 and Figure3 in the paper.

##### Run the code

```
python daily_network_property.py
```

#### GBM

We use the GBM model to demonstrate the predictability of flow.

##### Run the code

```
python GBM_daily_pairflow.py
```



#### Data load example

If you want to load  and  view any individual files, an example of ipynb (data_load_example.ipynb) is given here.
