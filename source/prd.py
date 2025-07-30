import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

classs=["Covid19","Normal","Pneumonia"]
ml=load_model("D:\proj2\Data\covid_pneu_model.h5")

img=image.load_img("D:\\proj2\\Data\\test\\NORMAL\\NORMAL(1271).jpg",target_size=(224,224))
imgg=image.img_to_array(img)
imgg=np.expand_dims(imgg,axis=0)/255.0

prd=ml.predict(imgg)
ind=np.argmax(prd[0])
print(classs[ind])