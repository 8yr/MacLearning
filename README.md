
# MacLearning

Document classification using SVM, SMOTE, and TF-IDF

# MacLeModel Concept

The developed model aims to classify documents into three main categories: Administrative, Financial, and Specialized. 

The model uses machine learning techniques to learn patterns in the text and determine the category each document belongs to based on its content.

# How the Model Works

The model starts by reading the text data and converting it into a numerical representation using TF-IDF.
Then, an SVM model is trained on the data using techniques like SMOTE to balance the classes.
Finally, the model's performance is evaluated using Cross-Validation to ensure it can classify accurately.

# Techniques Used in the Model

TF-IDF (Term Frequency-Inverse Document Frequency):
Concept: This technique is used to convert text into numerical representations so the model can understand the words in the text and extract their importance. It calculates the frequency of a word in a document (Term Frequency) and the importance of the word across the document set (Inverse Document Frequency).
Advantages: TF-IDF helps identify the words that distinguish documents, enabling the model to understand the content more effectively.

SVM (Support Vector Machine):
Concept: A machine learning algorithm used for classification. It determines the hyperplane that separates different categories in the data.
Advantages: SVM is effective for classification when the number of features is large, as in the case of text data. Here, a linear kernel is used to separate the classes.

SMOTE (Synthetic Minority Over-sampling Technique):
Concept: A technique to increase the samples in classes with fewer data points. SMOTE creates synthetic samples for the underrepresented classes instead of just replicating the original data.
Advantages: It helps improve the balance between classes, preventing the model from being biased towards the larger classes.

Stratified K-Fold Cross-Validation:
Concept: A technique to evaluate the model by splitting the data into multiple folds, ensuring that each fold has a balanced distribution of classes.
Advantages: It helps evaluate the model more accurately, as it tests the model across multiple data splits rather than just one.
