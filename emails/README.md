# Emails categorization 
Email categorization is a task which involves classifying emails into meaningful groups using a supervised algorithm and Natural Language Processing (NLP) 
## Libraries 
## German Text Decoding 

utf-8: <br>
übermitteln - ьbermitteln <br>
Paßwort (High German - Passwort) - PaЯwort <br>
möglich - mцglich <br>
Decoded text using utf-8 example:
![Alt text](image.png)<br>
Decoded text using latin-1 example:<br>
![](image-1.png)
## Approcahes to tackle the problem 
<ol>
<li> Traditional ML
<li> Deep learning
<li> LLMs <br>
The huge advantage of using LLMs for solving NLP problems is that these models already have some language understanding having been trained on the large amouns of data. Therefore, you don't need much training data unlike when you train a DL model from scratch. <br>

| <center> Technique   | <center>Advs | <center> Disadvs |
|---|---|---|
| Zero-Shot Classification | No data needed | Performance is usually mere. Hallucination. |
| Few-Shot Classification | Little data needed | Performance is limited. |
| Raw embedding feature extraction | Gives better accuracies when compared to the other two techniques | Big amount of annotated data needed. |

<a href="https://github.com/flairNLP/flair">Flair - the NLP framework<a>
</ol>

## Accuracies 
Traditional ML
before improving decoding method:<br>
![](image-2.png)<br>
![ ](image-3.png)
