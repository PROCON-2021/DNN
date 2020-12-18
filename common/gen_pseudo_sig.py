import numpy as np

def main():
    
    dlen = 10000 # 信号長

    mean = 0.0   
    std  = 1.0

    # 4ch分を想定
    sig1 = np.random.normal(mean,std,dlen)
    sig2 = np.random.normal(mean,std,dlen)
    sig3 = np.random.normal(mean,std,dlen)
    sig4 = np.random.normal(mean,std,dlen)

    # shape: len x ch
    sig = np.stack([sig1, sig2, sig3, sig4], axis=1)

    np.savetxt('./sig.npy', sig, delimiter=',')

if __name__ == "__main__":
    main()