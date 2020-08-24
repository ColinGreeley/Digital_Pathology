# Digital_Pathology
------
### Description of the task

The general goal of this project is to classify images of tissue as either diseased or non-diseased. These images of tissue are composed of numerous cells that are individually diseased or non-diseased. Because of this, being able to classify the entire image is not very useful. 
The approach that I am interested in to work around this problem is called image segmentation. The way this will work is by running an object detection algorithm over the image to put a bounding box around each cell. The segmented cell image will then be fed into a deep neural network classifier that will return a probability representing whether or not the cell is diseased. If this probability passes a certain threshold, the bounding box for that cell will be overlayed on the original image along with the given probability.
