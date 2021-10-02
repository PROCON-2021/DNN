import os, sys
import numpy as np
from scipy.fftpack import fft, ifft
from numpy import hamming
from torch.utils.data import DataLoader, random_split
import torch as t
import torch_optimizer as t_opt
import matplotlib.pyplot as plt

# モデルのネットワーク構造をtxt形式で出力
def export_network(model, out_dir) -> None:

    with open(f'{out_dir}/network.txt', 'w') as f:
        print(f'{model}\n', file=f)

        for param in model.state_dict():
            print(f'{param}\t{model.state_dict()[param].size()}', file=f)

def generate_dataloader(dataset, batch_size, split_ratio=0.8, shuffle=[True, False], worker=0, pin_memory=False):
    '''
    [params]
        dataset: instance of Dataset class
        batch_size: mini batch size
        split_ratio: proportion of training data in the overall data set
        shuffle: whether to shuffle the dataset (list object: [train, val])
        ---------------
        [Speed-up option]
        worker: number of workers
        pin_memory: if True, the CPU memory area will not be paged
    [return]
        train_loader: dataloader for training 
        val_loader: dataloader for validation
    '''
    samples = len(dataset)
    train_size = int(samples*split_ratio)
    val_size = samples - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle[0], num_workers=worker, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=shuffle[1], num_workers=worker, pin_memory=pin_memory)

    return train_loader, val_loader

def check_loss(epoch, itr, interval, running_loss, logger=None):
        print(f'[{epoch+1}, {itr+1:03}] loss: {running_loss/interval:.5f}')

        if logger != None:
            logger.log({'Training loss': running_loss/interval})

def get_optimzer(optimizer_name, model, lr):

    if optimizer_name == 'Adam':
        optimizer = t.optim.Adam(model.parameters(), lr)
    elif optimizer_name == 'RAdam':
        optimizer = t_opt.RAdam(model.parameters(), lr)
    elif optimizer_name == 'SGD':
        optimizer = t.optim.SGD(model.parameters(), lr)

    return optimizer

def sig2spec(src, fft_size, shift_size, window='hamming'):
    """
    Parameters
    ----------
    signal: input signal
    fft_size: frame length
    shift_size: frame shift
    window: window function
    Returns
    -------
    S: spectrogram of input signal (fft_size/2+1 x frame x ch)
    """

    signal = np.array(src)

    if window == 'hamming':
        window = hamming(fft_size+1)[:fft_size]
    else:
        print(window+' is not supported.')
        sys.exit(0)

    zeroPadSize = fft_size - shift_size
    length = signal.shape[0]
    frames = int(np.floor((length + fft_size - 1) / shift_size))
    I = int(fft_size/2 + 1)

    if len(signal.shape) == 1:
        # monoral
        signal = np.concatenate([np.zeros(zeroPadSize), signal, np.zeros(fft_size)])
        S = np.zeros([I, frames], dtype=np.complex128)

        for j in range(frames):
            sp = j * shift_size
            spectrum = fft(signal[sp: sp+fft_size] * window)
            S[:, j] = spectrum[:I]

        return S

    elif len(signal.shape) == 2:
        nch = signal.shape[1]
        signal = np.concatenate([np.zeros([zeroPadSize, nch]), signal, np.zeros([fft_size,nch])])
        S = np.zeros([I, frames, nch], dtype=np.complex128)

        for ch in range(nch):
            for j in range(frames):
                sp = j * shift_size
                spectrum = fft(signal[sp: sp+fft_size, ch] * window)
                S[:, j, ch] = spectrum[:I]

        return S

    else:
        raise ValueError('illegal signal dimension')

def show_spec(spec, fs=16000, fft_size=2048, shift_size=1024, xrange=None, yrange=None, cbar=False, output=None, imshow=False, plot_type='log'):
    '''
    Parameters
    ----------------
    spec: Specetrogram (1ch absolute scale)
    fs: Sampling frequency
    fft_size: FFT length
    shift_size: overlap length
    xrange: Range of X axis (list [x min, x max])
    yrange: Range of Y axis (same format as xrange)
    crange: Range of Colorbar (list [min(db), max(db)])
    cbar: Whether show color bar or not
    output: Output file name 
    imshow: Preview in program (Does not work on remote env.)
    '''

    if len(spec.shape) == 1:
        spec = spec.reshape([spec.shape[0],1])
        I = spec.shape[0]
        J = 1
    else:
        I, J = spec.shape

    if fft_size is None:
        fft_size = (I - 1) * 2
    if shift_size is None:
        shift_size = fft_size // 2

    t = np.round((J * shift_size - fft_size + 1) / fs)+1

    # x-y axis range
    if xrange == None:
        x_min = 0
        x_max = t  
    else:
        x_min = xrange[0]
        x_max = xrange[1]

    if yrange == None:
        y_min = fs // 2
        y_max = 0
    else:
        y_min = yrange[1]
        y_max = yrange[0]

    ax_range = [x_min, x_max, y_min, y_max]

    plt.figure()

    if plot_type == 'log':
        epsilon = sys.float_info.epsilon
        S = np.where(spec < epsilon, spec+epsilon, spec) # フロアリング
        plt.imshow(20*np.log10(S), extent=ax_range, aspect='auto', cmap='cividis', interpolation='nearest')
    elif plot_type == 'default':
        plt.imshow(spec, extent=ax_range, aspect='auto', cmap='cividis', interpolation='nearest')

    if cbar:
        plt.colorbar()

    plt.gca().invert_yaxis()
    
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    if output != None:
        plt.savefig(output, bbox_inches='tight', dpi=300)
    
    if imshow is True:
        plt.show()

    plt.close()

def opt_syn_wnd(analysis_window, shift_size):
    fft_size = analysis_window.shape[0]
    synthesized_window = np.zeros(fft_size)
    for i in range(shift_size):
        amp = 0
        for j in range(1, int(fft_size / shift_size) + 1):
            amp += analysis_window[i + (j-1) * shift_size] ** 2
        for j in range(1, int(fft_size / shift_size) + 1):
            synthesized_window[i + (j-1) * shift_size] = analysis_window[i + (j-1) * shift_size] / amp

    return synthesized_window