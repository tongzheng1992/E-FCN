import tensorflow as tf

#This is the pignistic transformation layer to convert mass functions from a DS layer into pignistic probabilisty
#The orginal defination of pignistic transformation is in the paper 'Decision-making with belief function: a review'

class DM_pignistic(tf.keras.layers.Layer):
    def __init__(self, num_class):
        super(DM_pignistic, self).__init__()
        self.num_class=num_class
        
    def call(self, inputs):
        aveage_Pignistic=tf.divide(inputs[:,:,:,-1], self.num_class)
        aveage_Pignistic=tf.expand_dims(aveage_Pignistic, -1)
        mass_class = inputs[:,:,:,0:-1]
        Pignistic_prob=tf.add(mass_class, aveage_Pignistic, name=None)

        return Pignistic_prob