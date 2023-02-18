class Kernel_regressor:
    def __init__(self, length_scale, training_data):
        self.lscale = length_scale
        self.training_data = training_data
    
    def kernel(self, x, x_):
        # lscale detemrines how much we values points that
        # are further from our selves
        return np.exp(-np.power(x - x_, 2)/self.lscale)
    
    def regress(self, x_):
        
        # caclulates sumofall(training_y * kernel)/sumofall(kernel)
        closness = 0
        estimate = 0
        for x, y in self.training_data:
            relation = self.kernel(x, x_)
            estimate += y * relation
            closness += relation
        return estimate/closness