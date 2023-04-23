# Databricks notebook source
pip install Pillow


# COMMAND ----------

"""

PLAN:
    Get face images - cropped and likely will need to be lowered in res for compute restrictions
        start with online dataset.
        second project is to develop a face detector to detect a floating face in an image
    Use just gray scale images
    Create train validation test split -> test and validation set must include faces not in the train set as well as faces that are
    Flatten the face images
    Calcualte the mean face
    Subtract the mean face from the population
    Compuate the eignen vectors of the subtracted population using the SVD composition
        check how to get the eigen values of this non square matrix
    Plot elbow plot for optimla choice of eigen vectors
    
    CHECK HOW TO COMPARE A NEW IMAGE TO THE FACE SPACE. (I.E is it one of the registered faces?)
    If so, which one? - Check distance to closest individual face space
    
    Build algo with SVM and 
    


"""
import os
from PIL import Image
import numpy as np
path = r'/dbfs/mnt/other_azzneuproddatalake/Operational_analytics/TS/EigenFaces/Data/'

# cwd = os.getcwd()

names = os.listdir(path)

def load_image(image_path):
    img = Image.open(image_path)
    loaded_temp_image = np.array(img)
    return loaded_temp_image

def load_images(file_path, names):
    
    # create empty dictionary to store the photos - the key is the name and the value is a list of images
    image_dic = {}
    for name in names:
        person_image_filename =  os.listdir(file_path + name)
        person_image_list = []
        for image in person_image_filename:
            loaded_temp_image = load_image( file_path + name + '/' + image)# SOME FUNCTION TO LOAD AN IMAGE - GREY SCALE, COMPRESS IMAGE TOO?
            person_image_list.append(loaded_temp_image)
        
        # add persons images to the image dictionary
        image_dic[name] = person_image_list
    return image_dic

dic = load_images(path, names)


# COMMAND ----------

for key in dic.keys():
  print(key)
  for image in dic[key]:
    print(image.shape)


# COMMAND ----------


