import mynetwork
import loader
import numpy as np

def main():
    
    net = mynetwork.MyNetwork()
    W1, W2, b1, b2 = net.sgd()
    net.save()
    #print(np.exp(-709) / 1)

if __name__ == '__main__':
    main()