from keras.models import Model
from keras.layers import Dense, Embedding, Conv1D, multiply, GlobalMaxPool1D, Input, Activation

def Malconv(max_len=200000, win_size=500, vocab_size=256, num_classes=2):    
    inp = Input((max_len,))
    
    # Make sure vocab_size is 256 (not num_classes)
    emb = Embedding(input_dim=vocab_size, output_dim=8)(inp)

    conv1 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    conv2 = Conv1D(kernel_size=(win_size), filters=128, strides=(win_size), padding='same')(emb)
    a = Activation('sigmoid', name='sigmoid')(conv2)
    
    mul = multiply([conv1, a])
    a = Activation('relu', name='relu')(mul)
    p = GlobalMaxPool1D()(a)
    d = Dense(64)(p)
    if num_classes == 2:
        out = Dense(1, activation='sigmoid')(d)
    elif num_classes > 2:
        # Use num_classes for the final output layer
        out = Dense(num_classes, activation='softmax')(d)
    else:
        return None
    return Model(inp, out)