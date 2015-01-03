from ca.nengo.math.impl import GaussianPDF

import nef

class NoiseNode(nef.SimpleNode):
    """Node to output gaussian noise with mean 0 and standard deviation driven
    by an input signal.
    
    :input scale: scale on the output values (modifying standard deviation from base of 1)
    :output noise: vector of noisy values
    """
    
    def __init__(self, frequency, dimension=1):
        """Initialize node variables.
        
        :param frequency: frequency to update noise values
        :param dimension: dimension of noise signal
        """
        
        self.period = 1.0/frequency
        self.scale = 0.0
        self.updatetime = 0.0
        self.state = [0.0 for _ in range(dimension)]
        # remember that mean is the shift of the curve and variance is the flatness
        self.pdf = GaussianPDF(0,1) # (mean, variance)
        
        nef.SimpleNode.__init__(self, "NoiseNode")
        
    def tick(self):
        if self.t > self.updatetime:
            self.state = [self.pdf.sample()[0]*self.scale for _ in range(len(self.state))]
            self.updatetime = self.t + self.period
        
    def termination_scale(self, x):
        self.scale = x[0] # gets set to 0.03 by bg_network
        
    def origin_noise(self):
        return self.state