import tensorflow as tf
nn = tf.keras

# FFN
def FFN(d_model:int, n_classes:int):
    """
    input: Transformer_Output => `(B, N, d_model)`\n
    output: [Class_Prob, BBox_Pred] => `[(B, N, n_classes), (B, N, 4)]`
    """
    inputs = nn.layers.Input(shape=(None, d_model)) # (B, N, d_model)
    
    class_prob = nn.layers.Dense(n_classes, activation="softmax")(inputs) # (B, N, n_classes)
    bbox_pred = nn.layers.Dense(4, activation="sigmoid")(inputs) # (B, N, 4)
    
    return nn.Model(inputs=inputs, outputs=[class_prob, bbox_pred], name="ffn")
