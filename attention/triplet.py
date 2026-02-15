
from tensorflow import keras
from keras.ops import mean , max , broadcast_to
from keras.layers import Concatenate , Conv2D , BatchNormalization , Activation , Multiply , Add , GlobalAveragePooling2D , Dense , Permute , Average , Layer
#--------------------------------
# tensorflow is channel last
class TripletAtt(Layer) : 
    def __init__(self , k_size):
        super().__init__()
        self.k_size   = k_size
        self.convBr1  = Conv2D(filters=1 , kernel_size=self.k_size , strides=1 , padding="same")
        self.convBr2  = Conv2D(filters=1 , kernel_size=self.k_size , strides=1 , padding="same")
        self.convBr3  = Conv2D(filters=1 , kernel_size=self.k_size , strides=1 , padding="same")
        self.bnBr1    = BatchNormalization(axis=-1)
        self.bnBr2    = BatchNormalization(axis=-1)
        self.bnBr3    = BatchNormalization(axis=-1)
        self.concat   = Concatenate(axis=-1) 
        self.multiply = Multiply()
        self.Avg      = Average()
        self.Add      = Add()
        self.Per_H_axis = Permute((1 , 3 , 2))
        self.Per_W_axis = Permute((3 , 2 , 1))
    #---
    def _z_pool(self,tensor) : 
        avgpool = mean(tensor , axis=-1 , keepdims=True)
        maxpool = max(tensor  , axis=-1 , keepdims=True)
        concat  = self.concat([maxpool , avgpool])
        return concat
    #---
    def _Permutation(self , tensor , type) :
        # standard tensor shape (none , height , width , channel)
        if type == "H-axis" : 
            tensor_hat  = self.Per_H_axis(tensor)
        if type == "W-axis" :
            tensor_hat  = self.Per_W_axis(tensor)
        #--
        return tensor_hat
    def _2DAttentionMap(self , tensor_hat_star , branch , output="weights" , training=None) :
        if branch == 1 : 
            tensor_hat_star_conv   = self.convBr1(tensor_hat_star)
            tensor_hat_star_conv_N = self.bnBr1(tensor_hat_star_conv , training=training)
            if output == "weights" :
                attention_map = Activation(activation='sigmoid')(tensor_hat_star_conv_N)
            elif output == "RAW" :
                attention_map = tensor_hat_star_conv_N
            return attention_map
        elif branch == 2 :
            tensor_hat_star_conv   = self.convBr2(tensor_hat_star)
            tensor_hat_star_conv_N = self.bnBr2(tensor_hat_star_conv , training=training)
            if output == "weights" :
                attention_map = Activation(activation='sigmoid')(tensor_hat_star_conv_N)
            elif output == "RAW" :
                attention_map = tensor_hat_star_conv_N
            return attention_map
        elif branch == 3 : 
            tensor_hat_star_conv   = self.convBr3(tensor_hat_star)
            tensor_hat_star_conv_N = self.bnBr3(tensor_hat_star_conv , training=training)
            if output == "weights" :
                attention_map = Activation(activation='sigmoid')(tensor_hat_star_conv_N)
            elif output == "RAW" :
                attention_map = tensor_hat_star_conv_N
            return attention_map
    def _branch_H_C(self , tensor , training=None): # branch one
        tensor_hat       = self._Permutation(tensor , "H-axis")
        tensor_hat_star  = self._z_pool(tensor_hat)
        attention_map    = self._2DAttentionMap(tensor_hat_star , branch=1 , training=training)
        attention_out    = self.multiply([tensor_hat , attention_map])
        rotated_tensor   = self._Permutation(attention_out , "H-axis")
        return rotated_tensor
    def _branch_W_C(self , tensor , training=None): # branch two
        tensor_hat       = self._Permutation(tensor , "W-axis")
        tensor_hat_star  = self._z_pool(tensor_hat)
        attention_map    = self._2DAttentionMap(tensor_hat_star , branch=2 ,  training=training)
        attention_out    = self.multiply([tensor_hat , attention_map])
        rotated_tensor   = self._Permutation(attention_out , "W-axis")
        return rotated_tensor
    def _branch_identify(self , tensor , training=None): # branch identify (third branch)
        tensor_hat    = self._z_pool(tensor)
        attention_map = self._2DAttentionMap(tensor_hat , branch=3 , training=training)
        attention_out = self.multiply([tensor , attention_map])
        return attention_out
    def triplet_Attention(self , tensor , training=None) :
        br_1_out  = self._branch_H_C(tensor , training=training)
        br_2_out  = self._branch_W_C(tensor , training=training)
        br_3_out  = self._branch_identify(tensor , training=training)
        #---
        br_unify = self.Avg([br_1_out , br_2_out , br_3_out]) # notice ==> averaging layer
        return br_unify
    def call(self , tensor , training=None):
        return self.triplet_Attention(tensor , training=training)
class TripletSeAttention(TripletAtt) :
    def __init__(self , k_size , varient="TriSE1" , reduction_ratio=8):
        super().__init__(k_size)
        self.var = varient
        self.rr = reduction_ratio
    def __SE_block(self , tensor , output="channel_scaled") :
        GavgPoolTensor  = GlobalAveragePooling2D(keepdims=True)(tensor)
        n_channels      = GavgPoolTensor.shape[-1]
        hidden_out      = Dense(n_channels//self.rr , activation="relu" , kernel_initializer="he_normal")(GavgPoolTensor)
        if output != "RAW" : # may be an error occure (Rub a debug test)
            channel_weights = Dense(n_channels , activation="sigmoid")(hidden_out)
            channel_scaled  = ([channel_weights , tensor])
        elif output == "RAW" : 
            channel_raw_score = Dense(n_channels , activation=None)(hidden_out)
        #--
        if output == "channel_scaled" :
            return channel_scaled
        elif output == "channel_weights" :
            return channel_weights
        elif output == "RAW" :
            return channel_raw_score
    def __TriSE1(self , tensor) :
        br_1_out = self._branch_H_C(tensor)
        br_2_out = self._branch_W_C(tensor)
        br_3_out = self._branch_identify(tensor)
        br_unify = self.Add([br_1_out , br_2_out , br_3_out])
        tensor_out = self.__SE_block(br_unify)
        return tensor_out
    def __TriSE2(self , tensor) :
        tensor_hat_br1       = self._Permutation(tensor , "H-axis")
        tensor_hat_br1       = self.__SE_block(tensor_hat_br1)
        tensor_hat_start_br1 = self._z_pool(tensor_hat_br1)
        attention_map        = self._2DAttentionMap(tensor_hat_start_br1)
        attention_out        = ([tensor_hat_br1 , attention_map])
        output_tensor_br1   = self._Permutation(attention_out , "H-axis")
        #---
        tensor_hat_br2       = self._Permutation(tensor , "W-axis")
        tensor_hat_br2       = self.__SE_block(tensor_hat_br2)
        tensor_hat_start_br2 = self._z_pool(tensor_hat_br2)
        attention_map        = self._2DAttentionMap(tensor_hat_start_br2)
        attention_out        = ([tensor_hat_br2 , attention_map])
        output_tensor_br2   = self._Permutation(attention_out , "W-axis")
        #---
        tensor_br3           = self.__SE_block(tensor)
        tensor_hat_br3       = self._z_pool(tensor_br3)
        attention_map        = self._2DAttentionMap(tensor_hat_br3)
        output_tensor_br3    = ([tensor , attention_map])
        #---
        tensor_out = self.Avg([output_tensor_br1 , output_tensor_br2 , output_tensor_br3]) # error
        return tensor_out
    def __TriSE3(self , tensor) :
        tensor_hat_br1         = self._Permutation(tensor , "H-axis")
        tensor_hat_start_br1   = self._z_pool(tensor_hat_br1)
        attention_map          = self._2DAttentionMap(tensor_hat_start_br1)
        attention_out          = ([tensor_hat_br1 , attention_map])
        tensor_hat_br1_weights = self.__SE_block(tensor_hat_br1 , output="channel_weights")
        branch1_scaled         = ([attention_out , tensor_hat_br1_weights])
        output_tensor_br1      = self._Permutation(branch1_scaled , "H-axis")
        #-----
        tensor_hat_br2         = self._Permutation(tensor , "W-axis")
        tensor_hat_start_br2   = self._z_pool(tensor_hat_br2)
        attention_map          = self._2DAttentionMap(tensor_hat_start_br2)
        attention_out          = ([tensor_hat_br2 , attention_map])
        tensor_hat_br2_weights = self.__SE_block(tensor_hat_br2 , output="channel_weights")
        branch2_scaled         = ([attention_out , tensor_hat_br2_weights])
        output_tensor_br2      = self._Permutation(branch2_scaled , "W-axis")
        #-----
        tensor_hat_br3         = self._z_pool(tensor)
        attention_map          = self._2DAttentionMap(tensor_hat_br3)
        attention_out          = ([tensor , attention_map])
        tensor_hat_br3_weights = self.__SE_block(tensor , output="channel_weights")
        output_tensor_br3      = ([attention_out , tensor_hat_br3_weights])
        #-----
        tensor_out = self.Avg([output_tensor_br1 , output_tensor_br2 , output_tensor_br3])
        return tensor_out
    def __TriSE4(self,tensor) :
        tensor_hat_br1           = self._Permutation(tensor , "H-axis")
        tensor_hat_star_br1      = self._z_pool(tensor_hat_br1)
        attention_map_raw        = self._2DAttentionMap(tensor_hat_star_br1 , output="RAW") # 2D vector
        tensor_hat_br1_raw       = self.__SE_block(tensor_hat_br1 , output="RAW") # 1D vector
        tensor_hat_br1_raw_B     = broadcast_to(tensor_hat_br1_raw , (attention_map_raw.shape[0] , attention_map_raw.shape[1] , attention_map_raw.shape[2] , tensor_hat_br1_raw.shape[-1]))
        AffineTransfomer         = self.Add([attention_map_raw , tensor_hat_br1_raw_B])
        AffineTransfomer_weights = Activation(activation="sigmoid")(AffineTransfomer)
        attention_out            = ([AffineTransfomer_weights , tensor_hat_br1]) 
        output_tensor_br1        = self._Permutation(attention_out , "H-axis")
        #-----
        tensor_hat_br2           = self._Permutation(tensor , "W-axis")
        tensor_hat_star_br2      = self._z_pool(tensor_hat_br2)
        attention_map_raw        = self._2DAttentionMap(tensor_hat_star_br2 , output="RAW") # 2D vector
        tensor_hat_br2_raw       = self.__SE_block(tensor_hat_br2 , output="RAW") # 1D vector
        tensor_hat_br2_raw_B     = broadcast_to(tensor_hat_br2_raw , (attention_map_raw.shape[0] , attention_map_raw.shape[1] , attention_map_raw.shape[2] , tensor_hat_br2_raw.shape[-1]))
        AffineTransfomer         = self.Add([attention_map_raw , tensor_hat_br2_raw_B])
        AffineTransfomer_weights = Activation(activation="sigmoid")(AffineTransfomer)
        attention_out            = ([AffineTransfomer_weights , tensor_hat_br2]) 
        output_tensor_br2        = self._Permutation(attention_out , "W-axis")
        #-----
        tensor_hat_br3           = self._z_pool(tensor)
        attention_map_raw        = self._2DAttentionMap(tensor_hat_br3 , output="RAW") # 2D vector
        tensor_br3_raw           = self.__SE_block(tensor , output="RAW") # 1D vector
        tensor_br3_raw_B         = broadcast_to(tensor_br3_raw , (attention_map_raw.shape[0] , attention_map_raw.shape[1] , attention_map_raw.shape[2] , tensor_br3_raw.shape[-1]))
        AffineTransfomer         = self.Add([attention_map_raw , tensor_br3_raw_B])
        AffineTransfomer_weights = Activation(activation="sigmoid")(AffineTransfomer)
        output_tensor_br3        = ([AffineTransfomer_weights , tensor]) 
        #------
        br_unify   = self.Add([output_tensor_br1 , output_tensor_br2 , output_tensor_br3])
        tensor_out = self.__SE_block(br_unify , output="channel_scaled")
        return tensor_out
    def __call__(self, tensor):
        if self.var == "TriSE1" :
            return self.__TriSE1(tensor)
        if self.var == "TriSE2" :
            return self.__TriSE2(tensor)
        if self.var == "TriSE3" :
            return self.__TriSE3(tensor)
        if self.var == "TriSE4" :
            return self.__TriSE4(tensor)