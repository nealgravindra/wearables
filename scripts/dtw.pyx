import fastdtw

def dtw(x, y):
    return fastdtw.fastdtw(x, y)[0]