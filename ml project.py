#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install PyWavelets==0.5.2 opencv-python==3.4.10.37 seaborn==0.8.1


# In[2]:


pip install opencv-python


# In[1]:


import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


img = cv2.imread('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/testimages/10014.jpeg')
img.shape


# In[3]:


plt.imshow(img)


# In[4]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape


# In[5]:


gray


# In[6]:


plt.imshow(gray, cmap='gray')


# In[7]:


face_cascade = cv2.CascadeClassifier('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces


# In[8]:


(x,y,w,h) = faces[0]
x,y,w,h


# In[9]:


face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)


# In[10]:


cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(roi_color, cmap='gray')


# In[12]:


cropped_img = np.array(roi_color)
cropped_img.shape


# In[13]:


import numpy as np
import pywt
import cv2    

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


# In[14]:


im_har = w2d(cropped_img,'db1',5)
plt.imshow(im_har, cmap='gray')


# In[15]:


def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


# In[16]:


original_image = cv2.imread('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/testimages/10014.jpeg')
plt.imshow(original_image)


# In[17]:


cropped_image = get_cropped_image_if_2_eyes('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/testimages/10014.jpeg')
plt.imshow(cropped_image)


# In[18]:



org_image_obstructed = cv2.imread('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/testimages/10001.jpeg')
plt.imshow(org_image_obstructed)


# In[19]:


cropped_image_no_2_eyes = get_cropped_image_if_2_eyes('C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/testimages/10001.jpeg')
cropped_image_no_2_eyes


# In[20]:


path_to_data = "C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/dataset/"
path_to_cr_data = "C:/Users/B-Traders/OneDrive/Desktop/imageclassifier/model/dataset/cropped/"


# In[21]:


import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)


# In[22]:


img_dirs


# In[23]:


import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)


# In[24]:


def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    return None


# In[25]:


cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = os.path.basename(img_dir)  # Get the folder name
    celebrity_file_names_dict[celebrity_name] = []

    for entry in os.scandir(img_dir):
        if entry.is_file():  # Ensure it's a file
            valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']  # Added .webp
            if not any(entry.path.lower().endswith(ext) for ext in valid_extensions):
                print(f"Skipping non-image file: {entry.path}")
                continue

            roi_color = get_cropped_image_if_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = os.path.join(path_to_cr_data, celebrity_name)
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                    print(f"Generating cropped images in folder: {cropped_folder}")

                cropped_file_name = f"{celebrity_name}{count}.png"
                cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1


# In[26]:


celebrity_file_names_dict = {}
for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list
celebrity_file_names_dict


# In[27]:


class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict


# In[28]:


X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name]) 


# In[29]:


len(X[0])


# In[30]:


32*32*3 + 32*32


# In[31]:


X[0]


# In[32]:


y[0]


# In[33]:


X = np.array(X).reshape(len(X),4096).astype(float)
X.shape


# In[34]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# In[36]:


print(classification_report(y_test, pipe.predict(X_test)))


# In[ ]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[50]:


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}


# In[51]:


scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[52]:


best_estimators


# In[53]:


best_estimators['svm'].score(X_test,y_test)


# In[54]:


best_estimators['random_forest'].score(X_test,y_test)


# In[55]:


best_estimators['logistic_regression'].score(X_test,y_test)


# In[56]:


best_clf = best_estimators['svm']


# In[57]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
cm


# In[58]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[59]:


class_dict


# In[60]:


get_ipython().system('pip install joblib')
import joblib 
# Save the model as a pickle in a file 
joblib.dump(best_clf, 'saved_model.pkl') 


# In[63]:


import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))


# In[70]:


import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pywt
import joblib
import json

# --- Load your trained model and class dictionary ---
model = joblib.load('saved_model.pkl')  # Make sure the path is correct
with open('class_dictionary.json', 'r') as f:
    class_dict = json.load(f)
class_dict_rev = {v: k for k, v in class_dict.items()}

# --- Helper functions ---
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Set approximation coefficients to zero
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    img_w2d = w2d(img, 'db1', 5)
    img_w2d = cv2.resize(img_w2d, (32, 32))
    
    combined_img = np.vstack((img.reshape(32*32*3, 1), img_w2d.reshape(32*32, 1)))
    combined_img = combined_img.reshape(1, -1).astype(float)
    
    prediction = model.predict(combined_img)[0]
    celebrity_name = class_dict_rev[prediction]
    return celebrity_name

# --- UI Code ---
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
    if file_path:
        selected_image_path.set(file_path)
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
    else:
        selected_image_path.set("")
        panel.config(image="", text="No Image Selected")

def predict():
    path = selected_image_path.get()
    if path:
        pred = predict_image(path)
        messagebox.showinfo("Prediction", f"Predicted Celebrity: {pred}")
    else:
        messagebox.showwarning("No Image", "Please select an image first.")

# Main window
root = tk.Tk()
root.title("Celebrity Classifier")
root.geometry("500x700")
root.configure(bg="white")

selected_image_path = tk.StringVar()

# --- Layout ---
top_frame = tk.Frame(root, bg="white")
top_frame.pack(pady=20)

browse_btn = tk.Button(top_frame, text="Browse Image", command=browse_image, font=("Arial", 16), bg="blue", fg="white")
browse_btn.pack(pady=10)

predict_btn = tk.Button(top_frame, text="Predict", command=predict, font=("Arial", 16), bg="green", fg="white")
predict_btn.pack(pady=10)

# Image Frame
img_frame = tk.Frame(root, bg="white")
img_frame.pack(pady=30)

panel = tk.Label(img_frame, text="No Image Selected", width=250, height=250, bg="gray")
panel.pack()

root.mainloop()






