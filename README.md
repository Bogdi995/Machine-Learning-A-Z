# Machine-Learning-A-Z
This repository consists of the exercises that I did for the machine learning A-Z course offered by Kirill Eremenko and Hadelin de Ponteves on Udemy. [Here](https://www.udemy.com/certificate/UC-14ceb79a-9a6b-4850-a0db-f40c89fce307/) are the credentials of the certification.

---

# 1. Data Preprocessing
## 1.1 Training and Test Sets
**Definition**: Training and test sets are partitions of the dataset used in machine learning. The training set is employed to train the model, while the test set evaluates the model's performance on unseen data, ensuring generalization.

**Purpose**: By separating data into training and test sets, we assess the model's ability to generalize patterns learned during training to new, unseen data, preventing overfitting.

## 1.2 Feature Scaling
**Definition**: Feature scaling is the process of normalizing or standardizing the range of independent variables in the dataset. It ensures that no variable dominates others, preventing biased model training.

**Purpose**: Scaling features improves model convergence, especially for algorithms sensitive to variable magnitudes, such as gradient-based methods.

# 2. Regression
## 2.1 Linear Regression (Ordinary Least Squares)
**Definition**: Linear regression is a statistical method modeling the relationship between a dependent variable and one or more independent variables using a linear equation.

**Purpose**: Ordinary Least Squares optimizes the model's coefficients to minimize the sum of squared differences between observed and predicted values.

## 2.2 Multiple Linear Regression
**Definition**: Multiple Linear Regression extends linear regression to multiple independent variables, modeling complex relationships.

**Purpose**: Capturing relationships among multiple variables, it provides a more nuanced understanding of the impact on the dependent variable.

## 2.2.1 Assumptions of Linear Regression
**Definition**: Assumptions include linearity, independence, homoscedasticity, and normality, ensuring robustness and reliability of linear regression results.

**Purpose**: Validating assumptions is crucial for interpreting regression coefficients accurately and making reliable predictions.

### 2.2.2 Dummy Variables
**Definition**: Dummy variables represent categorical data, allowing incorporation into regression models.

**Purpose**: Facilitating the inclusion of categorical data in regression models, enhancing model interpretability.

### 2.2.3 Dummy Variable Trap
**Definition**: Dummy Variable Trap occurs when dummy variables are multicollinear, leading to issues in model interpretation.

**Purpose**: Understanding and avoiding the dummy variable trap ensures unbiased regression coefficients.

### 2.2.4 Statistical Significance
**Definition**: Statistical significance tests determine whether coefficients significantly differ from zero.

**Purpose**: Identifying statistically significant predictors aids in feature selection and model refinement.

### 2.2.5 Building a Model
**Definition**: Building a model involves selecting predictors, training the model, and evaluating performance.

**Purpose**: Constructing a robust model involves iterative processes, enhancing predictive accuracy.

## 2.3 Polynomial Regression
**Definition**: Polynomial Regression models non-linear relationships by introducing polynomial terms.

**Purpose**: Capturing complex patterns in data beyond linear relationships, enhancing prediction accuracy.

## 2.4 Support Vector Regression (SVR) - Linear
**Definition**: Support Vector Regression utilizes support vector machines to model relationships in a high-dimensional space.

**Purpose**: Ideal for non-linear relationships, SVR efficiently handles complex datasets.

## 2.5 Decision Tree Regression
**Definition**: Decision Tree Regression uses a tree-like model to make decisions based on input features.

**Purpose**: Visualizing decision processes, Decision Trees provide interpretable models for regression tasks.

## 2.6 Random Forest Intuition
**Definition**: Random Forest combines multiple decision trees to enhance predictive accuracy and mitigate overfitting.

**Purpose**: Reducing variance and improving generalization, Random Forest is effective in various regression scenarios.

## 2.7 R Squared
**Definition**: R-squared measures the proportion of the dependent variable's variance explained by the model.

**Purpose**: Assessing goodness of fit, R-squared quantifies the model's explanatory power.

## 2.8 Adjusted R Squared
**Definition**: Adjusted R-squared modifies R-squared, considering the number of predictors in the model.

**Purpose**: Addressing the inclusion of irrelevant predictors, Adjusted R-squared provides a more reliable metric.

## 2.9 Regression Pros & Cons + Regularization
**Definition**: Evaluating the strengths and weaknesses of regression models, regularization techniques prevent overfitting.

**Purpose**: Identifying trade-offs in regression models and implementing regularization for improved model robustness.

# 3. Classification
## 3.1 Logistic Regression
**Definition**: Logistic Regression models the probability of a binary outcome using a logistic function.

**Purpose**: Ideal for binary classification, Logistic Regression is a foundational algorithm in classification tasks.

### 3.1.1 Maximum Likelihood:
**Definition**: Maximum Likelihood estimates parameters by maximizing the likelihood function.

**Purpose**: Ensuring accurate parameter estimation in Logistic Regression models.

## 3.2 K-Nearest Neighbors (K-NN)
**Definition**: K-NN classifies data points based on the majority class of their k-nearest neighbors.

**Purpose**: Robust for small datasets, K-NN excels in pattern recognition tasks.

## 3.3 SVM Intuition
**Definition**: Support Vector Machines (SVM) classify data points by finding the optimal hyperplane.

**Purpose**: Effective for linear and non-linear classification, SVMs offer versatility in various datasets.

## 3.4 Kernel SVM Intuition
**Definition**: Kernel SVM extends SVM to handle non-linear decision boundaries using kernel functions.

**Purpose**: Enabling SVM to handle complex, non-linear relationships in high-dimensional spaces.

## 3.4.1 The Kernel Trick
**Definition**: The Kernel Trick transforms input data into higher-dimensional spaces for improved separability.

**Purpose**: Overcoming limitations of linear decision boundaries, the Kernel Trick enhances SVM performance.

## 3.4.2 Types of Kernel Functions
**Definition**: Various kernel functions (linear, polynomial, radial basis function) impact SVM model performance differently.

**Purpose**: Selecting appropriate kernel functions for SVM, optimizing classification results.

## 3.4.3 Non-Linear SVR
**Definition**: Non-Linear Support Vector Regression extends SVM to regression tasks with non-linear relationships.

**Purpose**: Adapting SVM principles to regression scenarios, enhancing predictive capabilities.

## 3.5 Naive Bayes
**Definition**: Naive Bayes classifies data based on Bayes' theorem, assuming independence among predictors.

**Purpose**: Efficient for text and categorical data, Naive Bayes excels in classification tasks.

## 3.5.1 Naive Bayes Classifier Intuition
**Definition**: Naive Bayes Classifier estimates probabilities based on the independence assumption.

**Purpose**: Understanding the underlying probabilistic nature of Naive Bayes classification.

## 3.5.2 Naive Bayes Classifier Additional Comments
**Definition**: Additional considerations, strengths, and limitations associated with Naive Bayes classification.

**Purpose**: Insight into nuances of Naive Bayes, aiding informed model selection.

## 3.6 Decision Tree Classification
**Definition**: Decision Tree Classification uses a hierarchical tree structure for categorizing data.

**Purpose**: Visualizing decision rules, Decision Trees provide interpretable models for classification tasks.

## 3.7 Random Forest Classification
**Definition**: Random Forest Classification leverages ensemble learning for improved classification accuracy.

**Purpose**: Mitigating overfitting and enhancing robustness, Random Forest is effective for varied datasets.

## 3.8 Evaluating Classification Models Performance
**Definition**: Evaluation metrics like confusion matrix, accuracy, and CAP curve assess the performance of classification models.

**Purpose**: Quantifying model performance, aiding model selection, and optimization.

### 3.8.1 Confusion Matrix & Accuracy
**Definition**: Confusion Matrix tabulates true positive, true negative, false positive, and false negative values.

**Purpose**: Calculating accuracy and error rates, offering insights into classification model performance.

### 3.8.2 Accuracy Paradox
**Definition**: Accuracy Paradox highlights situations where accuracy may not be the sole determinant of model performance.

**Purpose**: Understanding nuances in accuracy interpretation, guiding comprehensive model evaluation.

### 3.8.3 CAP Curve (Cumulative Accuracy Profile)
**Definition**: CAP Curve illustrates the cumulative distribution of positive outcomes in classification models.

**Purpose**: Assessing model efficiency in capturing positive instances, aiding decision-making.

# 4 Clustering
## 4.1 K-Means Clustering
**Definition**: K-Means Clustering partitions data into k clusters based on similarity.

**Purpose**: Grouping similar data points, K-Means is effective in identifying inherent structures.

### 4.1.1 The Elbow Method
**Definition**: The Elbow Method helps determine the optimal number of clusters (k) in K-Means.

**Purpose**: Selecting an appropriate number of clusters, optimizing K-Means performance.

### 4.1.2 K-Means++
**Definition**: K-Means++ improves K-Means initialization by selecting centroids intelligently.

**Purpose**: Enhancing convergence and mitigating sensitivity to initial centroid placement.

## 4.2 Hierarchical Clustering
**Definition**: Hierarchical Clustering builds a hierarchy of clusters based on similarity.

**Purpose**: Visualizing hierarchical relationships, aiding in data exploration.

### 4.2.1 How do dendrograms work?
**Definition**: Dendrograms represent hierarchical relationships, showcasing cluster merging.

**Purpose**: Facilitating interpretation of hierarchical cluster structures.

### 4.2.2 Hierarchical Clustering Using Dendrograms
**Definition**: Utilizing dendrograms for interpreting hierarchical clustering results.

**Purpose**: Extracting meaningful insights from hierarchical clustering outcomes.

# 5 Association Rule Learning
## 5.1 Apriori
**Definition**: Apriori algorithm discovers association rules by identifying frequent itemsets.

**Purpose**: Extracting meaningful associations among items, aiding in decision-making.

## 5.2 Eclat
**Definition**: Eclat algorithm mines association rules through transaction itemset intersection.

**Purpose**: Simplifying association rule discovery, particularly in large datasets.

# 6 Reinforcement Learning
## 6.1 The Multi-Armed Bandit Problem
**Definition**: The Multi-Armed Bandit Problem addresses exploration-exploitation trade-offs in reinforcement learning.

**Purpose**: Balancing exploration of unknown options with exploitation of known options for optimal outcomes.

## 6.2 Upper Confidence Bound (UCB)
**Definition**: Upper Confidence Bound algorithm optimizes decision-making in the Multi-Armed Bandit Problem.

**Purpose**: Maximizing rewards by intelligently selecting actions based on uncertainty estimates.

## 6.3 Thompson Sampling
**Definition**: Thompson Sampling approaches the Multi-Armed Bandit Problem through probabilistic sampling.

**Purpose**: Balancing exploration and exploitation by sampling actions according to probability distributions.

## 6.4 UCB vs Thompson Sampling
**Definition**: Comparison of Upper Confidence Bound and Thompson Sampling approaches.

**Purpose**: Understanding strengths and weaknesses, aiding in algorithm selection for the Multi-Armed Bandit Problem.

# 7 Natural Language Processing (NLP)
## 7.1 Classical vs Deep Learning Models
**Definition**: Differentiating classical and deep learning models in Natural Language Processing (NLP).

**Purpose**: Identifying suitable models based on the complexity of NLP tasks.

## 7.2 Bag-Of-Words
**Definition**: Bag-Of-Words model represents text as an unordered set of words.

**Purpose**: Simplifying text data for machine learning tasks, emphasizing word frequency.

# 8 Deep Learning
**Definition**: Delving into the complex field of Deep Learning.

**Purpose**: Understanding the foundations and mechanisms of neural networks for advanced machine learning applications.

# 9 Artificial Intelligence
## 9.1 The Neuron
**Definition**: The Neuron serves as the fundamental unit in neural network architectures.

**Purpose**: Building blocks for information processing in artificial neural networks.

## 9.2 Activation Functions
**Definition**: Activation functions introduce non-linearity to neural network models.

**Purpose**: Enabling neural networks to learn complex, non-linear patterns in data.

## 9.3 How do Neural Networks work
**Definition**: The functioning of neural networks involves processing input data through interconnected layers.

**Purpose**: Gaining insights into the flow of information and decision-making within neural networks.

## 9.4 How do NNs learn
**Definition**: Neural networks learn by adjusting weights through iterative optimization processes.

**Purpose**: Understanding the adaptive learning mechanisms of neural networks.

## 9.5 Gradient Descent
**Definition**: Gradient Descent is an optimization algorithm used to minimize the error in neural network models.

**Purpose**: Iteratively adjusting weights to approach optimal model parameters.

## 9.6 Stochastic Gradient Descent
**Definition**: Stochastic Gradient Descent optimizes neural networks using random subsets of training data.

**Purpose**: Enhancing efficiency and convergence in large-scale datasets.

## 9.7 Backpropagation
**Definition**: Backpropagation is a supervised learning algorithm for training neural networks.

**Purpose**: Adjusting weights based on prediction errors, refining model performance.

# 10 Convolutional Neural Networks (CNN)
## 10.1 Convolution
**Definition**: Convolution is a fundamental operation in Convolutional Neural Networks (CNNs) for feature extraction.

**Purpose**: Capturing local patterns and hierarchical representations in image data.

## 10.2 ReLU (Rectified Linear Unit)
**Definition**: Rectified Linear Unit (ReLU) is an activation function introducing non-linearity in CNNs.

**Purpose**: Enhancing the expressive power of CNNs by introducing non-linearities.

## 10.3 Pooling
**Definition**: Pooling layers in CNNs down-sample feature maps to reduce computational complexity.

**Purpose**: Retaining important information while reducing spatial dimensions in CNNs.

## 10.4 Flattening
**Definition**: Flattening transforms pooled feature maps into a one-dimensional vector.

**Purpose**: Preparing data for fully connected layers in CNNs.

## 10.5 Full Connection
**Definition**: Fully connected layers connect all neurons from one layer to another, facilitating high-level reasoning.

**Purpose**: Capturing global patterns and relationships in CNNs.

## 10.6 Summary
**Purpose**: Summarizing key concepts covered in Convolutional Neural Networks.

## 10.7 Softmax & Cross-Entropy
**Definition**: Softmax and Cross-Entropy are used in the output layer of CNNs for multi-class classification.

### 10.7.1 Softmax
**Definition**: Softmax transforms raw scores into probability distributions in multi-class classification.

**Purpose**: Assigning probabilities to each class for comprehensive classification.

### 10.7.2 Cross-Entropy
**Definition**: Cross-Entropy measures the dissimilarity between predicted and true probability distributions.

**Purpose**: Optimizing model predictions by minimizing the cross-entropy loss

# 11 Dimensionality Reduction
## 11.1 Principal Component Analysis (PCA)
**Definition**: Principal Component Analysis (PCA) reduces the dimensionality of data by transforming it into a set of uncorrelated variables.

**Purpose**: Capturing essential information in datasets with high dimensionality, aiding in visualization and model efficiency.

## 11.2 Linear Discriminant Analysis (LDA)
**Definition**: Linear Discriminant Analysis (LDA) finds the linear combinations of features that best separate different classes.

**Purpose**: Maximizing class separability in classification tasks, emphasizing inter-class differences.

## 11.3 Kernel PCA
**Definition**: Kernel Principal Component Analysis (Kernel PCA) extends PCA to handle non-linear relationships through kernel functions.

**Purpose**: Preserving non-linear structures in high-dimensional data during dimensionality reduction.

# 12 Model Selection & Boosting
## 12.1 Model Selection
### 12.1.1 k-Fold Cross Validation
**Definition**: k-Fold Cross Validation assesses model performance by partitioning data into k subsets for training and testing.

**Purpose**: Reducing variability in performance metrics, ensuring robust model evaluation.

### 12.1.2 Grid Search
**Definition**: Grid Search explores hyperparameter combinations systematically to find optimal model configurations.

**Purpose**: Automating hyperparameter tuning, enhancing model performance.

## 12.2 Model Boosting
### 12.2.1 XGBoost
**Definition**: XGBoost (Extreme Gradient Boosting) is an ensemble learning method using decision trees for boosting.

**Purpose**: Improving predictive accuracy by combining weak learners, achieving state-of-the-art results.
