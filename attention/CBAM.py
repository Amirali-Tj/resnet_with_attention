from tensorflow import keras
from keras.ops import mean , max , broadcast_to
from keras.layers import Concatenate , Conv2D , BatchNormalization , Activation , Multiply , Add , GlobalAveragePooling2D , GlobalMaxPooling2D , Dense , Permute , Average , Layer
#--------
class CBAM(Layer) :
    def __init__(self , k_size , reduction_ratio=16 , SubModule="full" , structure="sequential") :
        super().__init__()
        self.type   = SubModule
        self.rr     = reduction_ratio
        self.struct = structure
        self.k_size = k_size
        self.conv = Conv2D(filters=1 , kernel_size=self.k_size , activation="sigmoid" , strides=1 , padding="same")
        self.GAP  = GlobalAveragePooling2D(keepdims=True) 
        self.GMP  = GlobalMaxPooling2D(keepdims=True)
        self.Add  = Add()
        self.multiply = Multiply()
        self.concat   = Concatenate() 
    def build(self , input_shape) :
        self.hidden = Dense(max(input_shape[-1]//self.rr) , activation="relu" , kernel_initializer="he_normal")
        self.out    = Dense(input_shape[-1] , activation=None)
    def _channelAtt(self , tensor) :
        GAP_out = self.GAP(tensor)
        GMP_out = self.GMP(tensor)
        hidden_GAP_out  = self.hidden(GAP_out)
        final_GAP_out   = self.out(hidden_GAP_out)
        hidden_GMP_out  = self.hidden(GMP_out)
        final_GMP_out   = self.out(hidden_GMP_out)
        unify_out       = self.Add([final_GAP_out , final_GMP_out])
        channel_weights = Activation(activation="sigmoid")(unify_out)
        channel_scaled  = self.multiply([tensor , channel_weights])
        return channel_scaled
    def _spatialAtt(self , tensor) :
        Mean_map       = mean(tensor , axis=-1 , keepdims=True)
        Max_map        = max(tensor  , axis=-1 , keepdims=True) 
        attention_map  = self.conv(self.concat([Mean_map , Max_map]))
        spatial_scaled = self.multiply([attention_map , tensor])
        return spatial_scaled
    def call(self , tensor) :
        if self.type == "full" :
            if self.struct == "sequential" :
                channel_scaled = self._channelAtt(tensor)
                spatioChannel_scaled = self._spatialAtt(channel_scaled)
                return spatioChannel_scaled
        elif self.type == "channel" :
            channel_scaled = self._channelAtt(tensor)
            return channel_scaled
        elif self.type == "spatial" :
            spatial_scaled = self._spatialAtt(tensor)
            return spatial_scaled

         