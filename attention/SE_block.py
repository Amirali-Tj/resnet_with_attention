from tensorflow import keras
from keras.layers import Layer , GlobalAveragePooling2D , Dense , Multiply
#---
class se(Layer) :
    def __init__(self , reduction_ratio , out="channel_scaled") : #constructor
        super().__init__()
        self.rr  = reduction_ratio
        self.out = out
        self.GAP = GlobalAveragePooling2D(keepdims=True)
        self.multiply = Multiply()
    def build(self, input_shape):
        self.hidden = Dense(max(input_shape[-1]//self.rr , 1) , activation="relu" , kernel_initializer="he_normal")
        if self.out == "channel_scaled" or self.out == "channel_weights" :
            self.score = Dense(input_shape[-1] , activation="sigmoid")
        elif self.out == "RAW" :
           self.score = Dense(input_shape[-1] , activation=None)
    def _se_block(self , tensor) :
        GAP_out    = self.GAP(tensor)
        hidden_out = self.hidden(GAP_out)
        score_out  = self.score(hidden_out)
        #--
        if self.out == "channel_scaled" :
            return self.multiply([score_out , tensor])
        elif self.out == "channel_weights" or self.out == "RAW":
            return score_out
    def call(self , tensor) :
        return self._se_block(tensor)
        
        
        


