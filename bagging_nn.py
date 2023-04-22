# Ensemble learning of Neural Network Using Bagging


import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tkinter import filedialog
import os

def UploadAction1():
    global data,actual_data
    hyp_data = sio.loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select Image Data File'))
    data = hyp_data[sorted(hyp_data.keys())[-1]]
    actual_data = hyp_data[sorted(hyp_data.keys())[-1]]
    data = applyPCA(data)

def UploadAction2():
    global labels,actual_labels
    gt_data = sio.loadmat(filedialog.askopenfilename(initialdir=os.getcwd(), title='Select GT File'))
    labels = gt_data[sorted(gt_data.keys())[-1]]
    actual_labels = gt_data[sorted(gt_data.keys())[-1]]


def applyPCA(X, n_components=30, seed=1):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], n_components))
    return newX


def whole_model():

    global labels,data,actual_data,actual_labels
    label_no = len(np.unique(labels))
    print(data.shape)
    print(labels.shape)

    data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    data.shape

    labels = labels.reshape(-1,1)
    labels.shape

    combine = np.concatenate((data,labels),axis=1)
    combine

    hyp_columns = []
    for i in range(data.shape[1]):
        hyp_columns.append('b'+str(i))

    hyp_columns.append('labels')

    df = pd.DataFrame(combine,columns = hyp_columns)
    df.head()

    hyp_columns = hyp_columns[-1:] + hyp_columns[:-1]
    df = df[hyp_columns]
    df.head()

    class NeuralNetwork:

        def __init__(self, input_size, hidden_size, output_size):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size

            # Initialize weights and biases
            self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
            self.b1 = np.zeros((1, self.hidden_size))
            self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
            self.b2 = np.zeros((1, self.output_size))

        def forward(self, X):
            # Forward pass
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = np.maximum(0, self.z1)  # ReLU activation
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))  # Softmax activation
            self.a2 /= np.sum(self.a2, axis=1, keepdims=True)
            return self.a2

        def backward(self, X, y, learning_rate):
            # Backward pass
            m = X.shape[0]

            dZ2 = self.a2
            dZ2[range(m), y.astype(int)] -= 1
            dZ2 /= m

            dW2 = np.dot(self.a1.T, dZ2)
            db2 = np.sum(dZ2, axis=0)

            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * (self.a1 > 0)  # ReLU derivative

            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0)

            # Update weights and biases
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1

        def train(self, X, y, epochs, learning_rate):
            epsilon = 1e-8
            for epoch in range(epochs):
                # Forward pass
                a2 = self.forward(X)

                # Compute loss
                loss = -np.sum(np.log(a2[range(X.shape[0]), y.astype(int)]+epsilon)) / X.shape[0]

                # Backward pass
                self.backward(X, y, learning_rate)

                # Print loss
                if (epoch+1) % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}')

        def predict(self, X):
            # Predict class labels
            a2 = self.forward(X)
            y_pred = np.argmax(a2, axis=1)
            return y_pred

    def sampling(tdata,val):
        len1= int(len(tdata)*val)
        t1 = tdata.sample(frac=1)
        val_data = t1.iloc[0:len1,:]
        train = t1.iloc[len1:,:]
        return val_data,train

    model_net = []
    for i in range(int(ensemble.get())):
        validation,train_data = sampling(df,float(fraction.get()))
        v_data = validation.iloc[:,1:]
        v_label = validation.iloc[:,0]
        x_data = train_data.iloc[:,1:]
        y_data = train_data.iloc[:,0]
        nn = NeuralNetwork(x_data.shape[1],128,label_no)
        nn.train(x_data,y_data,int(iterations.get()),1)
        model_net.append(nn)
    from sklearn.metrics import accuracy_score
    y_pred1 = []
    label_op1 = []
    for i in range(len(v_data)):
        input = v_data.iloc[i]
        for model in model_net:
            prediction1 = model.predict(input)
            label_op1.append(prediction1)
        max = 0
        res1 = label_op1[0]
        for i in label_op1:
            freq1 = label_op1.count(i)
            if freq1 > max:
                max = freq1
                res1 = i
        label_op1.clear()
        y_pred1.append(res1)      
    # print(y_pred)

    cm2 = accuracy_score(v_label,y_pred1)
    # print(cm2)
    accuracy1 = cm2*100
    accuracy1 = round(accuracy1,2)


    y_pred = []
    label_op = []
    for i in range(len(data)):
        input = data[i]
        for model in model_net:
            prediction = model.predict(input)
            label_op.append(prediction)
        max = 0
        res = label_op[0]
        for i in label_op:
            freq = label_op.count(i)
            if freq > max:
                max = freq
                res = i
        label_op.clear()
        y_pred.append(res)      
    # print(y_pred)

    cm1 = accuracy_score(labels,y_pred)
    print(cm1)
    accuracy = cm1*100
    accuracy = round(accuracy,2)

    np.unique(y_pred, return_counts=True)

    arr = np.array(y_pred)
    arr.shape

    # reshape y_pred to match the shape of the original image
    y_pred_2d = np.reshape(y_pred, (actual_data.shape[0],actual_data.shape[1]))

    # # display the original image and the predicted labels side by side


    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # rest of the code for creating subplots and displaying images

    # display the original image
    axs[0].imshow(data.reshape(actual_data.shape[0],actual_data.shape[1],30)[:,:,0], cmap='gray')
    axs[0].set_title('Original Image')

    # display the predicted labels as a 2D image
    im = axs[2].imshow(y_pred_2d, cmap='twilight')
    axs[2].set_title('Predicted Labels')
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())

    im = axs[1].imshow(actual_labels, cmap='twilight')
    axs[1].set_title('Actual Labels')
    text = fig.text(0.50, 0.02, f'Accuracy is {accuracy1}',horizontalalignment='center', wrap=True )
        # plt.legend(label_cols)
    plt.show()
    

    print(accuracy)

import tkinter as tk
from tkinter import *

# Building a GUI :
root = tk.Tk()
root.title('Ensemble learning of Neural Network Using Bagging')

#for adding menu bar:
Menubar = Menu(root)
FileMenu = Menu(Menubar, tearoff=0)
FileMenu.add_command(label='Image Data File ', command=UploadAction1)
FileMenu.add_separator()
FileMenu.add_command(label='Ground Truth File', command=UploadAction2)
Menubar.add_cascade(label="Input",menu=FileMenu)
root.config(menu=Menubar)


FileMenu1 = Menu(Menubar, tearoff=0)
FileMenu1.add_command(label="Exit", command=quit)
Menubar.add_cascade(label="Exit",menu=FileMenu1)
root.config(menu=Menubar)


# creating label
from tkinter import ttk
c_p = Label(root, text = "Enter number of ensemble: ",font=('Arial', 12))
c_p.place(x = 0, y = 400)
C_value= tk.IntVar()
it = Label(root, text = "Enter number of iterations: ",font=('Arial', 12))
it.place(x = 0, y = 500)
it_value= tk.IntVar()
it2 = Label(root, text = "Enter fraction of testing data: ",font=('Arial', 12))
it2.place(x = 0, y = 600)
it2_value= tk.IntVar()



# creating entry box
ensemble = Entry(root, textvariable = C_value)
ensemble.place(x=220,y=403)
iterations = Entry(root, textvariable = it_value)
iterations.place(x=220,y=503)
fraction = Entry(root, textvariable = it2_value)
fraction.place(x=220,y=603)

Classification = Button(root, text= "Classification", command= whole_model)
Classification.place(x =500, y = 650)
root.geometry("1000x700")
root.mainloop()
