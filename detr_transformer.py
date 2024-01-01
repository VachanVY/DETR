import tensorflow as tf
tnp = tf.experimental.numpy
nn = tf.keras

# Fixed Pos embed
class PositionalEncoding(nn.layers.Layer):
    """Sine-Cosine Positional Embedding"""
    def __init__(self, max_length:int, d_model:int, **kwargs):
        super().__init__(**kwargs)
        p, i = tf.meshgrid(tf.range(max_length), 2*tf.range(d_model//2))
        p, i = tf.cast(p, tf.float32), tf.cast(i, tf.float32)
        
        theta = p/10_000**(i/d_model)
        angle = tf.transpose(theta)

        pos_embed = tf.stack([tf.sin(angle), tf.cos(angle)], axis=-1)
        self.pos_embed = tf.reshape(pos_embed, (max_length, d_model))

    def call(self, x=None):
        if x is None:
            return self.pos_embed # (max_length, d_model)
        return self.pos_embed[:x.shape[1], :] # (T, d_model)

# Attention
class Attention(nn.layers.Layer):
    def __init__(
            self, 
            causal:bool, 
            d_model:int, 
            n_heads:int, 
            dropout_rate:float, 
            **kwargs
    ):
        super().__init__(**kwargs)
        self.causal = causal
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.dq = self.dk = self.dv = d_model//n_heads

        self.w = nn.layers.Dense(self.d_model, use_bias=False)

        self.wq = nn.layers.Dense(d_model, use_bias=False)
        self.wk = nn.layers.Dense(d_model, use_bias=False)
        self.wv = nn.layers.Dense(d_model, use_bias=False)
    
    def _mask(self, x:tf.Tensor):
        tril = tnp.tril(tf.ones_like(x))
        return tf.where(tril==0., -tnp.inf, x)
    
    def call(
            self, 
            inp2q:tf.Tensor, 
            inp2k:tf.Tensor, 
            inp2v:tf.Tensor
        ):
        N, d_model = inp2q.shape[1:]
        T = inp2k.shape[1]

        # compute q, k, v
        q = self.wq(inp2q) # (B, N, d_model)
        k = self.wk(inp2k) # (B, T, d_model)
        v = self.wv(inp2v) # (B, T, d_model)
        
        # seperate heads
        q = tf.reshape(q, (self.n_heads, -1, N, self.dq)) # (h, B, N, dq)
        k = tf.reshape(k, (self.n_heads, -1, T, self.dk)) # (h, B, T, dk)
        v = tf.reshape(v, (self.n_heads, -1, T, self.dv)) # (h, B, T, dv)

        # compute attention weights
        att_wei = tf.matmul(q, k, transpose_b=True)/d_model**0.5 # (h, B, N, T)
        att_wei = self._mask(att_wei) if self.causal else att_wei # (h, B, N, T)
        att_wei = tf.nn.softmax(att_wei, axis=-1) # (h, B, N, N)

        # apply attention weights to v
        att_out = att_wei @ v # (h, B, N, dv)
        # combine heads
        att_out = tf.reshape(att_out, (-1, N, d_model)) # (B, T, h*dv) == (B, T, d_model)

        # linear of att_out
        linear_att_out = self.w(att_out) # (B, T, d_model)
        return linear_att_out       

# Detr Transformer
class Transformer(nn.Model):
    """
    Transformer Model
    Args:
        d_model: Embedding dimension
        n_heads: number of heads computed in attention layer in parallel
        n_encoder_layers: number of encoder layers
        n_decoder_layers: number of decoder layers
        dff_in: dimension inside feed-forward layer
        dropout_rate: probability of dropping units
        max_length: [encoder_max_length = max(H*W), decoder_max_length = N]

    Input:
        flatImgFeatures: shape(B, H, W, C)    
    """
    def __init__(
            self, 
            d_model: int,
            n_heads: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            dropout_rate: float,
            max_length: list,
            **kwargs
            ):
        super().__init__(**kwargs)
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.dropout_rate = dropout_rate
        self.max_length = max_length
        self.T, self.N = max_length[0], max_length[1]

        self.FeedForward = lambda : nn.Sequential([
            nn.layers.Dense(d_model*4),
            nn.layers.ReLU(),
            nn.layers.Dense(self.d_model),
            nn.layers.Dropout(self.dropout_rate)
        ])
        object_queries = nn.layers.Embedding(self.N, d_model)(tf.range(self.N))
        self.object_queries = lambda B: object_queries * tf.ones((B, self.N, self.d_model))
    
    def call(self, flatImgFeatures):
        pos_embed = PositionalEncoding(self.T, self.d_model)(flatImgFeatures) # (T, d_model)
        x = flatImgFeatures # (B, T, d_model)

        #######Encoder-Blocks########
        for _ in range(self.n_encoder_layers):
            z = Attention(
                    causal=False, 
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    dropout_rate=self.dropout_rate
                )(x+pos_embed, x+pos_embed, x) # (q, k, v)
            x = nn.layers.LayerNormalization()(z+x) # (B, T, d_model)

            z = self.FeedForward()(x)
            x = nn.layers.LayerNormalization()(z+x)
        encoder_output = x # (B, T, d_model)
        object_queries = self.object_queries(x.shape[0]) # (B, N, d_model)
        x = object_queries # intially only object queries is the input to decoder
        
        #######Decoder-Blocks########
        for _ in range(self.n_decoder_layers):
            z = Attention(
                    causal=False, 
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    dropout_rate=self.dropout_rate
                )(x+object_queries, x+object_queries, x) # (q, k, v):shape((B, N, d_model), (B, N, d_model) (B, N, d_model))
            x = nn.layers.LayerNormalization()(z+x) # (B, N, d_model)
            
            # cross attention
            z = Attention(
                    causal=False, 
                    d_model=self.d_model, 
                    n_heads=self.n_heads, 
                    dropout_rate=self.dropout_rate
                )(x+object_queries, encoder_output+pos_embed, encoder_output) # (q, k, v):shape((B, N, d_model), (B, T, d_model), (B, T, d_model))
            x = nn.layers.LayerNormalization()(z+x) # (B, N, d_model)

            z = self.FeedForward()(x)
            x = nn.layers.LayerNormalization()(z+x) # (B, N, d_model)
        outputs = x # (B, N, d_model)

        return outputs