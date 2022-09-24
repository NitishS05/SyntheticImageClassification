
def classify(imagelist):
    from keras.models import load_model
    import cv2
    import numpy as np

    import os

    from django.conf.urls.static import static
    from django.conf import settings


   
    os.chdir(settings.BASE_DIR)
    dirname = str(os.getcwd()+ "\\classifier")
    os.chdir(dirname)

    model = load_model('synthetic_classifier.h5')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

    directory = settings.MEDIA_ROOT
    os.chdir(directory)

    pred=[]
    imagepaths=[]

    for image in imagelist:
        imagepth = str(os.getcwd()+ "\\" + str(image))
        imagepaths.append(imagepth)

        img = cv2.imread(imagepth)
        img = cv2.resize(img,(224,224))
        img = np.reshape(img,[1,224,224,3])



        predict_x=model.predict(img) 
        classes_x=np.argmax(predict_x,axis=1)

        if (classes_x[0]==0):
            print("Natural")
            pred.append("Natural")
            
        else:
            print("Synthetic")
            pred.append("Synthetic")

    return pred, imagepaths
            