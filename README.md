# Convolutional-Neural-Network-CNN-model
RADI605 Modern Machine Learning assignment: Data Science for healthcare and clinical informatics (M.Sc.)


1. Apply data augmentation process to generate training, validation, and test set.  Explanation in details is a must, for example, what are the image manipulation technique that you use and why did you think it is appropriate for this problem.

	Due to overfitting given by the the original simpler sequential model, data augmentation was required to improve the model performance. In order to do that for this model, image rotation, zoom, and both horizontal and vertical image flipping were applied for the training dataset only by setting them inside the class of “ImageDataGenerator( )” presented in the data pre-processing library so that the samples belonging to the under-represented class were able to be more regenerated in order to have equal representation of both benign and malignant classes. This technique can help getting more diversity in the data resulting in helping the model generalise, while also reducing the bias. 
	Starting with a pixel normalisation technique, rescaling pixel values was done by 1.0/255 to scale the data to a range of 0-1. Rotation angle was then set to 15 degrees, while zoom range was (0.95, 0.95) to zoom the image randomly by 95%. In addition, both horizontal and vertical flipping were also applied to the model in order to create more variety of images in the dataset. Definitely, this flipping technique did not change the original meaning of this kind of image data at all.

2. Compare model at 3. (simppler sequential model) and 4. (transfer learning model) with appropriate evaluation metrics.  Explain why those metrics are suitable

	To compare between the sequential and transfer learning models, “accuracy” is a standard measurement suitable for evaluating overall model performance by showing how much the model can predict correctly, while observing “loss” values during training and validating model is also important in order to see how well it is in fitting the model to the given data. Due to imbalance dataset, “F1 score” is an additional suitable metric that should be used in model evaluation as well. Moreover, this model trying to predict diseases, benign or malignant tumours, a “sensitivity or recall” measurement is required to consider how precisely the model can predict malignant images as malignant, which return a number as 1. AUROC (Area Under the Receiver Operating Characteristics, AUC + ROC) curve is, additionally, utilised for the model evaluation as well as it is one of the most important evaluation metrics for checking any classification model’s performance, which “AUC” illustrates how much the model is capable of distinguishing between classes, which means the higher the AUC, the better the model is at distinguishing between patients with the disease and no disease, while “ROC” is a probability curve plotted “True Positive Rate (%)” against “False Positive Rate (%)”. Classifiers that can give ROC curves closer to the top-left corner indicate a better performance. Since the ROC does not depend on the class distribution, this makes it very useful for evaluating classifiers predicting rare events such as diseases or disasters. Confusion matrix and classification report are also essential tools in visualising prediction outcomes to clearly see how many predicted values are classified correctly.
	As shown from the evaluating and testing results of the simpler sequential and transfer learning models, the simpler model can provide higher in accuracy, f1 score and recall and lower in loss value than the transfer learning, comparing 91.31%  of accuracy, 69.28% of f1 score, 56% of recall and 0.23 of loss to 87.48%, 47%, 32% and 0.31, respectively. The ROC curve of the first proposed model shows its curve is closer to the top-left corner than the latter model, resulting in the area under the curve (AUC) is greater, accounting for 94% compared to 89%. 

3. Discuss on the results why one is better than another.
	
	The simpler sequential model can provide more impressive predictions than the transfer learning one since it could be about the data similarity. As transfer learning model approach relies on the assumption that the features learnt on a pre-trained model are transferable to a new task, the model may not be able to capture the important features of the new data if the data used for the pre-trained model is significantly different from the new task. Furthermore, the model’s complexity seems not appropriate to be used for this kind of simpler image data as the transfer learning approach is most effective when the pre-trained model is trained on a similar task with similar complexity. So, that is why the simpler model trained specifically for that task can perform better.

4. Suggests how to improve the weaker one and do additional experiment to show that your suggestions works.

	Training the pre-trained model for some layers inside the pre-trained architecture with our own dataset, which is fine-tuning, is a good idea to adapt the model to the specific new task in order to provide a good understanding of the model architecture and training procedures. The results after training and testing show that the transfer learning with fine-tuning model can provide better prediction performance than the simpler one, giving 93.65% of accuracy, 79.76% of f1 score, 72% of recall, and 0.17 of loss. 
