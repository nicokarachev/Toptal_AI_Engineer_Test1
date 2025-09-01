# Toptal_AI_Engineer_Test1



Make all images square.
Task Description

"Test Pipeline T1.zip" is dataset

REQUIREMENTS

-Create a data loading pipeline that loads the dataset. Your solution must be robust for very large datasets, assuming that NO MORE THAN 20 percent of the images can fit in the available memory. Do not forget the labels.

Ensure that images are extracted and nothing is missed.

Implement the data loading pipeline as a generator.


Download the dataset here: https://topt.al/MrcKdb


#if you use Google Colab or have wget, you can (but do not have to) load the dataset with:
! wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GIR3hdXVVr0uYWXdZdb5XKDdJUFlFeR4' -O 'Test Pipeline T1.zip'

def score(data_loader, subset="training"):
    ds=None
    zip='Test Pipeline T1.zip'
    batch_images, batch_labels = data_loader(zip, subset)
    print('batch size', len(batch_images))
    print('Image dimensions: ', batch_images[0].shape)
    print("Classnames found: ",np.unique(np.array(batch_labels)))

    # test that data can be loaded infinitely
    unique_images = []
    try:
        for i in range(15):
            batch_images, batch_labels = data_loader(zip)
            for batch_image in batch_images:
	    # if your images are returned as numpy arrays
                unique_images.append(batch_image.astype("uint8").ravel())
	    # if your images are returned as tensors:
	    # unique_images.append(batch_image.astype("uint8").ravel())
        print("Unique images loaded: ", len(np.unique(unique_images, axis=0)))            
    except Exception as e:
        print("Error loading 10 batches")

def data_loader(zip, res_type="train"):
    # implement your code here
    # return (array_of_image_data, array_of_labels)
    pass


