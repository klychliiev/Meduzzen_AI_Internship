# Emails classification 
<b>Email classification</b> is a task which involves classifying emails into meaningful groups using a supervised algorithm and Natural Language Processing (NLP). In this challenge, we deal with German-language emails belonging to different categories (21 overall). 

## Approach 

The problem was approached using 3 methods for text classification:
<ol>
<li> Traditional ML algorithms
<li> Zero-shot classification using LLMs
<li> Few-shot classification using Flair framework
</ol>

## Project structure 
There are 4 folders in this directory, with one containing datasets and code snippets needed for dataset generation ('datasets'), and with other 3 containing source code for all text classification methods (traditional ML, zero-shot and few-shot classification using Flair). 

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
    |   ├── metadata / ...
    │   ├── requirements.txt 
    │   ├── streamlit_app.py 
    │   ├── text_cleaning.py
    │   └── training_script.py
    ├── traditional_ml_approach /
    │   ├── deploy_streamlit.py 
    │   ├── Dockerfile
    │   ├── linear_svc_classifier.py
    |   ├── metadata / ...
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
Initially, utf-8 encoding was used, however, some emails were broken with German special characters being displayed incorrectly (vowels with diacritics and Eszett letter, ß). Using latin-1 encoding resolved the issue. 
| utf-8 |  latin-1 |
|:---:|:---:|
| ьbermitteln  |  übermitteln |
| PaЯwort  |  Paßwort (High German - Passwort) |
|  mцglich | möglich  |

Reading emails using latin-1 encoding and 'with open ... as' syntax:

``` python
# reading email using latin-1 decoding in Python
with open(cont, 'r', encoding='latin-1') as f:
    cont = f.read()
```

## Performance 
Small dataset (6 categories)
| Approach | Algorithm/model | Accuracy (%) |   
|---|---|:---:|
|Traditional ML | Linear SVC |  64 |   
|   | Logistics regression | 61 |   
|   | Multionmial NB | 59 |   
|   | Random forest | 46 |   
| Zero-shot classification| Gbert Large Zeroshot Nli  | 51  |   
|   | German Zeroshot | 42 |   
|  | German GPT-2 | 21 |

Full dataset (21 categories)
| Approach | Model  | Embeddings  |  Accuracy  |
|---|---|---|---|
| Few-shot classification | Flair | distil-bert  | 49 |
|   |   | bert-german-uncased  | 53 |
|   |   | bert-german-cased | 58 |