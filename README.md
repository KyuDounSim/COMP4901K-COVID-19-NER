# COMP4901K-Project

# Members
* Jayeol Lee (20308109)
* Kyu Doun Sim (20306527)
* Sehyun Choi (20469329)
## Repository Structure

```
./
├── Bert_Classifier.ipynb                       main experiment notbook.
├── README.md
├── data                                        provided dataset.
│   ├── test.pkl
│   ├── train.pkl
│   └── val.pkl
├── data_visualization.ipynb                    extra data visualization notebook.
├── evaluate.py                                 provided evaluate.py.
└── model                                       model modules
    ├── __init__.py
    ├── bert_model.py                           BERT define, train, and predict
    ├── hyperparameters.py                      hyperparameter class
    ├── preprocessing.py                        preprocess dataset
    └── tf2_encoder_checkpoint_converter.py     Used to convert TF1 Biobert checkpoint
                                                to TF2 Keras checkpoint.
```
