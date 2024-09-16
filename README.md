# Info
This is program in written entirely in C and was done for fun to learn about the math and underlying code behind pytorch.

Note: This code is highly unoptimized, it will be slower than PyTorch (or even NumPy) but that's okay! It is mainly because the program is single threaded and I will eventually try to increase the speed a bit. For now though, it has done its job because the point of this code was to learn! Even with its slowness, it was able to achieve a 99% score on the MNIST dataset in about an hour and a half :)

I would also like to try to add some more functions like mean squared error, sigmoid, leaky ReLU, etc. when I get time
## File structure

./data has the MNIST dataset for training/testing

./include has header files

./src has the main method (which includes the code for dataset handling) and the code for all the functions

Overall the program is pretty easy to figure out, just look at the main method if you would like to run it yourself
# Running the code
Just type make and run ./main, it is currently set to train on MNIST numbers.

Note: This program uses clang because it was faster than gcc, change the make file to gcc if you don't have clang
