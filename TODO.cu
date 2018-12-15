1. construct customized active functions:ReLU,sigmoid
    # I got different result with the author using his method.
2. train the network with plain text data
    # the structure of the author's network is not clear enough. He didn't mentioned filters' sizes and strides.
3. analyse the error caused by operations during the propagation of the network for Vector Homomorphic Encryption.
4. construct the network which provides operation for encrypted data with pretrained weight
    # if possible, I want the author's source code.
    # the operations depend on how the data is is encrypted
        - is every element of a feature map or a vector encrypted individually or the whole vector is encrypted within one operation
    # the operations may have to be written manually.
5. 2 options:
    1) use a c++ module to perform Homomorphic Encryption(the author used Leveled Homomorphic Encryption, I will learn about the operations provided by LHE)
        - I have to write some executable file with c++ and call it using python application
        - recently I read the paper about the specific HE scheme used in CryptoDL, it's teeming with professional terms from other related works, it's a really annoying situation. So I am working on the Vector HE scheme which is faster to implement, since I don't think I will be studying the efficiency of this encrypted network, it's OK to merely guarantte the correctness. 
    2) use pytorch or tensorflow to perform Vector Homomorphic Encryption, however the range of number is limited to 2^63, I want to try this option first.(sadly I have found this range of number is not enough for a large network, so I have to implement it with numpy with a customized data type)
6. To be determined