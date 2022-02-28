# Train-and-test-the-linear-Support-Vector-Machine-SVM-classifier-for-handwritten-digits-recognitio
The objective of this project is to use the SVM classifier to train a simple system for the recognition of handwritten digits (0, 1, …, 9). Use the MNIST database of handwritten digits for testing and training the system which is available on Blackboard-Homepage-Handwritten Digits Dataset. It contains 60k images that are sorted as one vector each with their labels.

1. Write a program to train and test the linear Support Vector Machine (SVM) classifier for handwritten
digits recognition.
Select a subset of the database consisting 1000 images (image indexes from 1000-1999) of handwritten digits for training the system and use another 100 images for testing the system (image indexes from 3000-3099).

       a) Train the SVM classifier with the training set (you may use any built-in function/library (e.g.
          sklearn.svm)).

       b) Classify the handwritten digits of the testing images using the trained SVM model (use any
          built-in function/library).

       c) Compute the accuracy.

2. Repeat the experiment in part 1 for training the SVM classifier with a large database. 10000 images
for training (image indexes from 20000-29999) and test with 100 images (images indexes from
30000-30100).

3. Repeat the experiment in part 2 for training the SVM classifier with the same set of the training
images (image indexes from 20000-29999) and test with another 1000 images (image indexes from
30000-31000).

4. Plot a confusion matrix of the true digit values and the predicted digit values for each part above
(parts 1,2, and 3).

5. Repeat the experiment in part 3 for training the SVM classifier with different set of kernel functions
(e. g. rbf, polynomial, etc.).

6. Create binary images of the handwritten digits by a simple thresholding (indicate the threshold value
you used) and repeat the experiment in part 3. You may try different threshold values. What can you
note comparing with part 3?

7. Create a comparison table for the accuracy of all parts above. 

Solution :

Support vector machines (SVM)

In machine learning, Support vector machines (SVM) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. It is mostly used in classification problems. In this algorithm, each data item is plotted as a point in n-dimensional space (where n is several features), with the value of each feature being the value of a particular coordinate. Then, classification is performed by finding the hyper-plane that best differentiates the two classes. 
In addition to performing linear classification, SVMs can efficiently perform a non-linear classification, implicitly mapping their inputs into high-dimensional feature spaces.

How SVM works

An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH).

Methodology:

Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier.

•	Consider a dataset with x1 and x2 as features. 

•	Classify the pair(x1, x2) of coordinates in either green or blue.

•	Find best line or decision boundary called hyperplane.

Formulation:

Let’s look at the two-dimensional case first. The two-dimensional linearly separable data can be separated by a line. The function of the line is y=ax+b. We rename x with x1 and y with x2 and we get:

![image](https://user-images.githubusercontent.com/42844121/155905740-9e5e613a-d216-4b9b-be25-a82f0786af81.png)

This equation is derived from two-dimensional vectors. But in fact, it also works for any number of dimensions. This is the equation of the hyperplane.

Handwriting Recognition:

Recognizing handwritten text is a problem that can be traced back to the first automatic machines that needed to recognize individual characters in handwritten documents. Think about, for example, the ZIP codes on letters at the post office and the automation needed to recognize these five digits. Perfect recognition of these codes is necessary to sort mail automatically and efficiently.

To address this issue in Python, the scikit-learn library provides a good example to better understand this technique, the issues involved, and the possibility of making predictions.

The problem we are solving in this blog involves predicting a numeric value, and then reading and interpreting an image that uses a handwritten font. So even in this case, we will have an estimator with the task of learning through a fit() function, and once it has reached a degree of predictive capability (a model sufficiently valid), it will produce a prediction with the predict() function. Then we will discuss the training set and validation set, created this time from a series of images.

The Digits Dataset

The scikit-learn library provides numerous datasets that are useful for testing many problems of data analysis and prediction of the results. Also, in this case there is a dataset of images called Digits.

Implementation:

Imported Required Libraries

![image](https://user-images.githubusercontent.com/42844121/155905859-4b3b84a8-5c4e-470f-be5e-9d3d6df6ad58.png)

Using MNIST load images and labels from given dataset

![image](https://user-images.githubusercontent.com/42844121/155905884-7f5c3bc7-104f-492c-9d94-d8ecc53967b6.png)

Pass the images and labels to fit method to train the and pass the test data array to predict method and using the accuracy_score method compare the expected list and predicted list.

![image](https://user-images.githubusercontent.com/42844121/155905900-55dceb7d-6156-4a31-bc4a-f7ca06f8b2d0.png)

Create a confusion matrix and plot

![image](https://user-images.githubusercontent.com/42844121/155905913-3148288c-6c6a-4f98-9902-13aa107d1e74.png)

Source code : 

![image](https://user-images.githubusercontent.com/42844121/155905924-514dee9c-4415-4b86-bae1-3d90a10aba22.png)

Output : 

![image](https://user-images.githubusercontent.com/42844121/155905935-fc3a65d3-e51c-4493-8be7-8d613e02fd62.png)

Part - 1

Train Linear SVM on 10000 images for training (image indexes from 20000-29999) and test with 100 images

![image](https://user-images.githubusercontent.com/42844121/155905948-e9141d29-e53b-491a-8e51-a78896a767ea.png)

![image](https://user-images.githubusercontent.com/42844121/155905955-c75b5f13-f35d-4544-80ef-d77b824314e3.png)

![image](https://user-images.githubusercontent.com/42844121/155905959-b5b40931-4add-4dc4-8457-691359707cc9.png)

Part - 2

Train Linear SVM on training images (image indexes from 20000-29999) and test with another 1000 images (image indexes from 30000-31000).

![image](https://user-images.githubusercontent.com/42844121/155905971-f1ee77c3-f58d-4fcc-ae5c-c050cd67fa58.png)

![image](https://user-images.githubusercontent.com/42844121/155905976-947d1994-3220-4b94-8e0a-6db8c8689486.png)

![image](https://user-images.githubusercontent.com/42844121/155905979-11fe7e29-570e-4913-9f42-77d5780f5354.png)

Part - 3

Train Linear SVM on training images (image indexes from 20000-29999) and test with another 1000 images (image indexes from 30000-31000).

![image](https://user-images.githubusercontent.com/42844121/155906075-1ec5b9b3-9592-4059-b804-2dbaf967aab6.png)

![image](https://user-images.githubusercontent.com/42844121/155906084-316e7dbb-1b5d-4e96-a55e-e7a5b88ada28.png)

![image](https://user-images.githubusercontent.com/42844121/155906087-69be3bd2-47a3-472e-a88a-c51420664408.png)

polynomial_kernel

the polynomial kernel is a kernel function commonly used with support vector machines (SVMs) and other kernelized models, that represents the similarity of vectors (training samples) in a feature space over polynomials of the original variables, allowing learning of non-linear models.

![image](https://user-images.githubusercontent.com/42844121/155906108-d6bb589c-4149-4429-90f1-7befd493fca2.png)

![image](https://user-images.githubusercontent.com/42844121/155906116-125b70ac-68ad-45e6-813c-b8966adb20b2.png)

![image](https://user-images.githubusercontent.com/42844121/155906126-2af8973a-9bdd-4542-a8d1-b32400d22d54.png)

gaussian_kernel

In SVM, kernels are used for solving nonlinear problems such as X-OR in higher dimensional where linear separation is not possible. Generally, SVM is a simple dot product operation (i.e., projection). Gaussian is one such kernel giving good linear separation in higher dimension for many nonlinear problems.

![image](https://user-images.githubusercontent.com/42844121/155906143-cd8802b4-1f26-4a96-a28d-87c60c480bb0.png)

![image](https://user-images.githubusercontent.com/42844121/155906157-096051a6-2bc2-4002-8df5-7811c5c32bcd.png)

![image](https://user-images.githubusercontent.com/42844121/155906162-5973b9c8-a75e-4176-8d70-c94786ce6f44.png)

sigmoid_kernel

The Sigmoid Kernel comes from the Neural Networks field, where the bipolar sigmoid function is often used as an activation function for artificial neurons. It is interesting to note that a SVM model using a sigmoid kernel function is equivalent to a two-layer, 
perceptron neural network.

![image](https://user-images.githubusercontent.com/42844121/155906183-952371d9-e8d0-485d-9b82-1be47655089f.png)

![image](https://user-images.githubusercontent.com/42844121/155906188-d7c7be91-06f0-44c1-893b-310fb9063222.png)

![image](https://user-images.githubusercontent.com/42844121/155906193-e1761d7f-b323-4899-afa3-f61899d273eb.png)

comparison table for the accuracy of all parts

Model	             Accuracy

Part 1	           0.7474

Part 2 	           0.85

Part 3	           0.832

Polynomial	       0.687

Gaussian	         0.953

Sigmoid	           0.8












