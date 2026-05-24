# coronary-calcification-segmentation

## Structure Project
```text

├── app                         **Pipeline Enginner** 
│   ├── app.py
│   ├── components.py
│   ├── __init__.py
│   └── temp_storage
├── data    # Storage data 
├── models                      **Model Engineer**
│   ├── __init__.py
│   ├── model.py
│   └── weights
├── README.md
├── requirements.txt
├── src
│   ├── agatston_score.py       **QA**
│   ├── data_processing.py      **Data Engineer**
│   ├── __init__.py
│   └── pre_processing.py       **Pipeline Enginner**
└── tests                       **QA**

'