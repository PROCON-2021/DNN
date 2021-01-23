import numpy as np
import csv

def main(n):
    dlen = 3866 # 信号長

    mean = 2.4
    std  = 0.05

    # 4ch分を想定
    sig1 = np.random.normal(mean,std,dlen)
    sig2 = np.random.normal(mean,std,dlen)
    sig3 = np.random.normal(mean,std,dlen)
    sig4 = np.random.normal(mean,std,dlen)
    sig5 = np.random.normal(mean,std,dlen)

    # shape: len x ch
    sig = np.stack([sig1, sig2, sig3, sig4, sig5], axis=1)


    np.savetxt('../dataset/trainee/'+str(n)+'.csv', sig, delimiter=',')
    #output_name = ['../dataset/', key, '_sig', str(n), '.npy']
    #np.savetxt(''.join(output_name), sig, delimiter=',')

if __name__ == "__main__":

#    for key in ['patience' ,'trainee']:
#        for n in range(20):
#            main(n, key)

    for key in ['trainee']:
        for n in range(1, 101):
                main(n)