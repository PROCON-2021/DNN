import numpy as np

def main(n, key):
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

    output_name = ['../dataset/', key, '_sig', str(n), '.npy']
    np.savetxt(''.join(output_name), sig, delimiter=',')

if __name__ == "__main__":
    
    for key in ['patience' ,'trainee']:
        for n in range(20):
            main(n, key)