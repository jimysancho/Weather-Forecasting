# Weather-Forecasting
Using a scratch LSTM, being able to predict the temperature

# Math_LSTM file
In this pdf you will find my derivation for the backpropagation equations, equations needed in order to implement a scratch LSTM. If you want to follow along it is necessary some knowledge of Linear Algebra and Calculus. 

# LSTMClass 
Using the forward propagation and backpropagation equations derived in the `Math_LSTM` file, creating a LSTM recurrent neural network using only the `numpy` library. 

# Code folder

In this folder there are two jupyter files: 

1. Simple application. In this notebook you will find an application of the LSTM, a very simple one: predicting the next value of a sinusoidal function. 

2. Clime forecasting. Although the application is the same as in the previous file, predicting the next value of a given sequence, in this notebook I have loaded and preprocess the data in order to make predictions on the temperature. It also includes a comparison with the TensorFlow LSTM model in order to see how well the scratch implementation works. 

# Final Results on the Jena dataset

![models_comparison](https://user-images.githubusercontent.com/105709376/209570090-a0114569-1c97-4786-adfb-1353bbede7cb.png)

As we can see in the image, the scratch implementation is able to predict the temperature almost as good as the tensorflow model. 
