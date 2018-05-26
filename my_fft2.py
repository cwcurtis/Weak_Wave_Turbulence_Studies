import pyfftw

class my_fft2(object):

    def __init__(self,KT):
        physv = pyfftw.empty_aligned((KT,KT), dtype = 'complex128')
        freqv = pyfftw.empty_aligned((KT,KT), dtype = 'complex128')

        fft_f = pyfftw.FFTW(physv, freqv, axes=(0,1))
        fft_in = pyfftw.FFTW(freqv, physv, direction='FFTW_BACKWARD', axes=(0,1))

        self.physv = physv
        self.freqv = freqv
        self.fft2 = fft_f
        self.ifft2 = fft_in
        self.size = (KT,KT)

    def fft2(self,fx):
        fk = pyfftw.empty_aligned(self.size, dtype = 'complex128')
        self.physv[:,:] = fx
        self.fft_f()
        fk[:,:] = self.freqv
        return fk

    def ifft2(self,fk):
        fx = pyfftw.empty_aligned(self.size, dtype = 'complex128')
        self.freqv[:,:] = fk
        self.fft_in()
        fx[:,:] = self.physv
        return fx