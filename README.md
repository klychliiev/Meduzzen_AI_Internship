# <b> Meduzzen & U2D AI Internship </b><br>

## Environment
To accomplish these challenges, I will use the VS Code Jupyter extension, and Google Colab whenever I need to build more sopihsitcated solutions requiring more computational capabilities.

## Projects Overview 
I'd like to try my hand at all three challenges (NLP, email classification/extraction and object detection), but as a computational limguistics student I am particularly interested in NLP.

### :speaking_head: NLP <br>

Below is the list of Python NLP modules I use in my work as a computational linguist. However, not all of them fit our requirements for this Internship, particularly, due to the limited number of supported languages.

Useful libraries for tackling NLP problems:

| <center>library  | <center> description  | <center> fits our needs |
|:---:|---|:---:|
| stanza  | Collection of tools for the linguistic analysis created by the Stanford NLP team. Supports multiple languages. For example, for NER task, stanza has pre-trained models for 34 languages.   | :white_check_mark: | 
| SpaCy  | Powerful Python NLP library with support for 70+ languages. It has in-built word vectors, has tools for tokenization, NER, POS-tagging, dependancy parsing, text classification, lemmatization, morphological analysis etc. | :white_check_mark:  |   
| pymorphy2  | Morpohlogical analyser written in Python. It is used for fetching information about grammatical properties of a particular word (POS, case, gender, number). Supports only two languages: Ukrainian and Russian. | :no_entry:  |     
| gensim  | Python library for topic modelling, document indexing and similarity retrieval with large corpora. | :white_check_mark: |      
| fasttext  | Useful Python library for working with word embeddings. Contains pre-trained word vectors for 157 (!) languages.  | :white_check_mark: |    
| langdetect  | Niche Python library designed exclusively for the language detection task. Able to detect 55 languages. | :white_check_mark: |   
| Polyglot  | Polyglot supports various multilingual applications and offers a wide range of analysis. Applications: language detection, tokenization, NER, POS-tagginf, sentiment analysis. | :white_check_mark:  |   
| nltk  | Suite of libraries and programs for symbolic and statistical NLP for English written in the Python programming language. | :no_entry: |   

I personally prefer spacy and stanza to a smaller extent for their diversity and overall accuracy for different tasks. When dealing with word vectors I use fasttext. Whenever I need a language detection I use langdetect. For example. recently I've been working on a hatespeech project and had to filter for posts written in the Ukrainian language only. 

### :love_letter: Emails classification

Working on tasks related to email classification and extraction, we deal with the text data in the first place, therefore, libraries listed in the NLP section will come in handy. For emails classification we can use sklearn and tensorflow/keras libraries.

|  library | description  | fits our need  |
|:---:|---|:---:|
|Scikit-learn | Open-source Python library which includes implementations of many traditional ML algorithms. | :white_check_mark: |
| TensorFlow | Open-source framework for prototyping and assessing machine learning models, primarily neural networks. | :white_check_mark: |

TensorFlow and Scikit-learn can be used for object detection and NLP as well. For instance, Tensorflow CNNs come in handy when working with images/video, while for NLP problems RNNs and LSTMs are often used. 

### :video_camera: Object detection (CV)

Useful Python packages for image & video data processing:

|library   | description  | fits our needs  |
|:---:|---|:---:|
| OpenCv  | CV library focused on real-time applications. The library has a modular structure and includes several hundreds of computer vision algorithms. | :white_check_mark: |
| Scikit-Image | Includes a collection of algorithms for image processing. Image processing toolbox for SciPy. | :white_check_mark:  |
| matplotlib | Library for creating static, animated and interactive visualisations. | :white_check_mark:  |
| Pillow | Contains all the basic image processing functionality; intuitive and easy-to-use. | :white_check_mark:  |
| numpy | While not being a specifically CV library, numpy provides powerful data structures and algorithms for easy image data manipulation. | :white_check_mark:  |

## Author 
:man_technologist: Kyrylo Klychliiev <br>
:house: Kyiv, Ukraine

