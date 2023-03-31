# Convolutional-Neural-Network-CNN-model
RADI605 Modern Machine Learning assignment: Data Science for healthcare and clinical informatics (M.Sc.)

### Model explanation
<!-- !toc (level=2) -->

1.1 [One](#one)
1.2 [Two](#two)

<!-- toc! -->

!toc (2)

1. Set up and prepare data
> 1.1 Import all necessary tools
> 1.2. Prepare data
>> 1.2.1 Create directory paths for both the locations of each train, test and validation folder as well as all label.txt files
>> 1.2.2 Create new folders comprising of benign and malignant in each train, test and val folder and 
>> 1.2.3 Categorise all images in each folder by matching their names with image names from the target-label.txt and then copy the images to the created benign and malignant folders by looking at the target labels. 
2. Conduct data processing
	2.1 Normalise and augment the data with the “ImageDataGenerator( )” class by setting various important parameters inside the class. For this model, the setting is rescale=1.0/255, rotation_range=15, zoom_range=(0.95, 0.95), horizontal_flip=True, vertical_flip=True, and data_format=“channels_last” for the image_gen_train, while image_gen_val and image_gen_test are only done with normalisation with rescale=1.0/255.
	2.2 Load the data and keep it in train_data_gen, val_data_gen, and test_data_gen with the “flow_from_directory ( )” class by setting some parameters inside including directory, target_size=(224,224), batch_size=32, color_mode=“rgb”, class_mode=“binary”, and seed, while shuffle was set as True except only test_data_gen
3. Build a simpler sequential model
	3.1  build up the model architecture by
		3.1.1 Adding three convolutional 2D, “Conv2D ( )”, layers with different numbers of filter for each layer, including 16, 32 and 16, but the same kernel_size=(3, 3) and strides=1 
		3.1.2 Adding max pooling (MaxPooling2D( )) layer after every layer of the Conv2D( ) to downsample the input along its spatial dimensions (height and width) by taking the maximum value over an input window (pool_size=(2,2)) for each channel of the input
		3.1.3 Flattening the input (from output of the last layer of maxPooling) with “Flatten( )” 
		3.1.4 Adding dense (Dense( )) and dropout (Dropout( )) layers with parameters setting of units=256 for the initial dense layer, 64 for the second, and 1 for the last one for binary classification, while the first dropout layer coming after the first dense layer was set as 0.5 and 0.2 for the last one
		3.1.5 Setting activation for all convolutional 2D and dense layers with “relu”, except the last dense, which was set as “sigmoid” for binary classification
		3.1.6 Compile the model with “BinaryCrossentropy” for loss, “Adam” for optimizer, and various metrics, including accuracy, precision, recall and auc, for metrics
	3.2 Train the model with fit( ) method for 20 epochs, at the same time, the validation set was used for evaluating during the training. “keras.callbacks.TensorBoard( )” was used to collect the history of training output parameters
	3.43Plot loss values (train_loss and val_loss) on a graph to see the trend of them
	3.4 Evaluate and test the model
	3.5 ROC curve, confusion matrix and classification report were used to visualise the prediction outputs
4. Build a transfer learning model
	4.1 Build up a transfer learning model architecture
		4.1.1 Import VGG16 to be used for this model
		4.1.2 Set up parameters inside the “VGG16( )” model with input_shape=(224, 224, 3), weight=“imagenet”, and include_top=False. 
		4.1.3 freeze the pre-trained architecture
		4.1.4 Apply the same flatten layer until the final dense binary classification layer and compile the model with the same parameters as the simpler model 
	4.2 Train, evaluate and test the model as well as visualise the prediction outputs like before.
5. Build a transfer learning model with fine-tuning
	5.1 Build up a transfer learning architecture with fine tuning
		5.1.1 Import VGG16 to be used for this model
		5.1.2 Set up parameters inside the “VGG16( )” model with input_shape=(224, 224, 3), weight=“imagenet”, and include_top=False. 
		5.1.3 freeze some layers of the pre-trained model as [ : 13] and train the rest of them to make the model architecture understand the new task.
		5.1.4 Apply the same flatten layer until the final dense binary classification layer and compile the model with the same parameters as the simpler model 
	5.2 Train, evaluate and test the model as well as visualise the prediction outputs like before.

### Questions:
1. Apply data augmentation process to generate training, validation, and test set.  Explanation in details is a must, for example, what are the image manipulation technique that you use and why did you think it is appropriate for this problem.

	Due to overfitting given by the the original simpler sequential model, data augmentation was required to improve the model performance. In order to do that for this model, image rotation, zoom, and both horizontal and vertical image flipping were applied for the training dataset only by setting them inside the class of “ImageDataGenerator( )” presented in the data pre-processing library so that the samples belonging to the under-represented class were able to be more regenerated in order to have equal representation of both benign and malignant classes. This technique can help getting more diversity in the data resulting in helping the model generalise, while also reducing the bias. 
	<br />
	<br />
	Starting with a pixel normalisation technique, rescaling pixel values was done by 1.0/255 to scale the data to a range of 0-1. Rotation angle was then set to 15 degrees, while zoom range was (0.95, 0.95) to zoom the image randomly by 95%. In addition, both horizontal and vertical flipping were also applied to the model in order to create more variety of images in the dataset. Definitely, this flipping technique did not change the original meaning of this kind of image data at all.

2. Compare model at 3. (simppler sequential model) and 4. (transfer learning model) with appropriate evaluation metrics.  Explain why those metrics are suitable

	To compare between the sequential and transfer learning models, “accuracy” is a standard measurement suitable for evaluating overall model performance by showing how much the model can predict correctly, while observing “loss” values during training and validating model is also important in order to see how well it is in fitting the model to the given data. Due to imbalance dataset, “F1 score” is an additional suitable metric that should be used in model evaluation as well. Moreover, this model trying to predict diseases, benign or malignant tumours, a “sensitivity or recall” measurement is required to consider how precisely the model can predict malignant images as malignant, which return a number as 1. AUROC (Area Under the Receiver Operating Characteristics, AUC + ROC) curve is, additionally, utilised for the model evaluation as well as it is one of the most important evaluation metrics for checking any classification model’s performance, which “AUC” illustrates how much the model is capable of distinguishing between classes, which means the higher the AUC, the better the model is at distinguishing between patients with the disease and no disease, while “ROC” is a probability curve plotted “True Positive Rate (%)” against “False Positive Rate (%)”. Classifiers that can give ROC curves closer to the top-left corner indicate a better performance. Since the ROC does not depend on the class distribution, this makes it very useful for evaluating classifiers predicting rare events such as diseases or disasters. Confusion matrix and classification report are also essential tools in visualising prediction outcomes to clearly see how many predicted values are classified correctly.
	<br />
	<br />
	As shown from the evaluating and testing results of the simpler sequential and transfer learning models, the simpler model can provide higher in accuracy, f1 score and recall and lower in loss value than the transfer learning, comparing 91.31%  of accuracy, 69.28% of f1 score, 56% of recall and 0.23 of loss to 87.48%, 47%, 32% and 0.31, respectively. The ROC curve of the first proposed model shows its curve is closer to the top-left corner than the latter model, resulting in the area under the curve (AUC) is greater, accounting for 94% compared to 89%. 

3. Discuss on the results why one is better than another.
	
	The simpler sequential model can provide more impressive predictions than the transfer learning one since it could be about the data similarity. As transfer learning model approach relies on the assumption that the features learnt on a pre-trained model are transferable to a new task, the model may not be able to capture the important features of the new data if the data used for the pre-trained model is significantly different from the new task. Furthermore, the model’s complexity seems not appropriate to be used for this kind of simpler image data as the transfer learning approach is most effective when the pre-trained model is trained on a similar task with similar complexity. So, that is why the simpler model trained specifically for that task can perform better.

4. Suggests how to improve the weaker one and do additional experiment to show that your suggestions works.

	Training the pre-trained model for some layers inside the pre-trained architecture with our own dataset, which is fine-tuning, is a good idea to adapt the model to the specific new task in order to provide a good understanding of the model architecture and training procedures. The results after training and testing show that the transfer learning with fine-tuning model can provide better prediction performance than the simpler one, giving 93.65% of accuracy, 79.76% of f1 score, 72% of recall, and 0.17 of loss. 
