# Emails classification 
<b>Email classification</b> is a task which involves classifying emails into meaningful groups using a supervised algorithm and Natural Language Processing (NLP).

## Approach 

The problem was approached using 3 methods for text classification:
<ol>
<li> Traditional ML algorithms 
<li> Zero-shot classification using LLMs
<li> Few-shot classification using Flair framework
</ol>

## Project structure 
There are 4 folders in this directory, with one containing datasets and code snippets needed for dataset generation, and with other 3 containing source code for all text classification methods (traditional ML, zero-shot and few-shot classification). 

```
.
└── emails /
    ├── datasets /
    │   ├── csv_datasets / ...
    │   ├── unprocessed_emails / ...
    │   └── create_csv_dataset.ipynb
    ├── flair_model_training /
    │   ├── flair /
    │   │   ├── data / ...
    │   │   └── model / ...
    │   ├── flair_emails_classification.ipynb 
    │   ├── full_dataset.csv 
    │   ├── requirements.txt 
    │   ├── streamlit_app.py 
    │   ├── text_cleaning.py
    │   └── training_script.py
    ├── traditional_ml_approach /
    │   ├── deploy_streamlit.py 
    │   ├── Dockerfile
    │   ├── linear_svc_classifier.py
    │   ├── requirements.txt
    │   ├── short_dataset.csv
    │   ├── text_cleaning.py
    │   └── traditional_mp.ipynb
    ├── zero_shot_classification /
    │   └── zero_shot_classification_emails.ipynb
    └── README.md
```

## Libraries 

| Library  |<center> NLP lib  |<center>Use case |
|:---:|:---:|---|
| Flair  | :heavy_plus_sign:  |Training custom few-shot text classification model using small dataset.|
|  SpaCy | :heavy_plus_sign: | Text preprocessing, including stop words removal. Model used - de_core_news_sm. |
| quopri  | :heavy_plus_sign: | German text decoding | 
| re  |  :heavy_plus_sign: | Text cleaning. Regex can detect certain patterns in the text defined by rules. |
| transformers | :heavy_plus_sign: | Provides API and tools to download and train pretrained models. Used in zero-shot classification. | 
|  matplotlib| :heavy_minus_sign:  | Data visualization | 
| scikit-learn | :heavy_minus_sign:  | Provides Python implementations of various traditional ML algorithms and functions for data manipulations. | 
| streamlit  | :heavy_minus_sign:  | Deploy ML models as simple web apps | 
| pandas | :heavy_minus_sign: | Tabular data manipulation | 

## German Text Decoding 
<b> :no_entry: utf-8: </b> <br>
übermitteln - ьbermitteln <br>
Paßwort (High German - Passwort) - PaЯwort <br>
möglich - mцglich <br>

<b> :white_check_mark: latin-1</b> decoding reads .txt emails correctly 

``` python
# reading email using latin-1 decoding in Python
with open(cont, 'r', encoding='latin-1') as f:
    cont = f.read()
```