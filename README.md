# Predict Future Sales

In this competition we will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 
Competition link: ```https://www.kaggle.com/c/competitive-data-science-predict-future-sales```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

We are asked to predict total sales for every product and store in the next month.

### Tools

- In this project, we are going to use Jupyter Notebook which is super useful and bring some advanteges for implementation.

- Also, we are going to use some libraries for estimating future sales. For example, TenseorFlow, Keras, scikit-learn and some other effective libraries. 


### Install and Run

#### Install python 3.7 from following link 
```https://www.python.org/downloads/```

#### After install python, needed libraries:

1) pip
2) numpy
3) pandas
4) scikit-learn
5) seaborn
6) matplotlib
7) lightgbm
8) xgboost

##### Then install them in this order:

1) To install pip follow the instructions given in the following website 
```https://www.liquidweb.com/kb/install-pip-windows/```

Open cmd or terminal and write the following commands in this order

2) pip install numpy
3) pip install pandas
4) pip install -U scikit-learn
5) pip install seaborn
6) pip install matplotlib
7) pip install lightgbm
8) pip install xgboost

#### Download repository with one of two options:

1)following command or from repository 
	```git clone https://github.com/enkaranfiles/predict-future-sales.git```

2)open the following link and click the "Clone or download" after download extract zip file
	```https://github.com/enkaranfiles/predict-future-sales.git```

##### After downloading repository, extract dataset.zip using "Extract here" and see dataset file is extracted in current directory

#### For preprocessing:
- Run preprocess.py this will create a traintest dataset under dataset folder

#### For the prediction with models:
- Run one of the python files named by "mXGBoost.py", "mDecisionTree.py", "mRandomForest.py", "mlightGBM.py"
- Running models returns a csv which is name same as python filename under outputdata folder



## Authors

* **Burak Baran** 
* **Atakan Filgöz**
* **Enes Karanfil**
* **Ömer Şeker**


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


