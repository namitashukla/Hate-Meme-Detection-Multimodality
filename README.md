# Hate-Meme-Detection-Multimodality


Hateful meme detection is a well-known research area that requires both visual and lin-
guistic understanding. It matters because in today’s world information and opinions stem

from multimedia. With people smartly disguising hateful intent behind apparently harmless
images/text which when combined within cultural and societal context can hurt sentiments of
various minority groups. Thus, there is a dire need to be able to detect such hateful multimedia
in a multimodal setting.

For this purpose, we have used Facebook’s hate meme detection data set specially anno-
tated such that the unimodal priors are bound to fail, that is, the images and text individually

don’t hold much signal. We have used ResNext and RoBERTa unimodal models as the base-
lines. In order to explore the multimodality of the dataset, we used the early fusion approach

by concatenating the ResNext embeddings of pure images (2047 dimensional) and RoBERTa
embeddings of text (768 dimensional) and then subsequently performing classification using

various fine-tuned models such as Shallow Feed Forward Network, Deep Feed Forward Net-
work, CatBoost, LGBM, XGBoost and Logistic Regression.

Our initial analysis suggested that despite the fact that the visual embeddings were almost
2.6 times larger than the textual ones, the information they carried was not sufficient. Hence,
we condensed the dimensionality of both text and image embeddings using PCA in order to
capture features delivering maximum variance/signal and also experimented with the different
dimensions of these embeddings to test our classification models’ performance. In order to
extract more signal from the textual embeddings, we fine-tuned the last 4 layers of RoBERTa
pre-trained on twitter’s tweets with another hate speech centric dataset from UC Berkeley to
further tune the embeddings for our specific problem statement.
Our performance metric is AUC because of the skewness in the dataset. It is observed that
retrained RoBERTa performs the best with a AUC of 0.66 when more weightage is given to the
text embeddings (150 dimensional) over the visual embeddings (50 dimensional) after applying
PCA.
