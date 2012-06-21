def horizontal_left_to_right_start_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for i in xrange(block_mem_size):
        serie[t] = i
        t += 1
    return serie

def horizontal_right_to_left_start_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(block_size[0]*(j+1)-1, block_size[0]*j-1, -1):
            serie[t] = i
            t += 1
    return serie

def horizontal_left_to_right_start_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(block_size[0]*(block_size[0]-j-1), block_size[0]*(block_size[0]-j), 1):
            serie[t] = i
            t += 1
    return serie

def horizontal_right_to_left_start_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for i in xrange(block_mem_size-1, -1, -1):
        serie[t] = i
        t += 1
    return serie

def vertical_up_to_down_start_left(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(j, block_mem_size, block_size[0]):
            serie[t] = i
            t += 1
    return serie

def vertical_down_to_up_start_left(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(block_mem_size-block_size[0]+j, j-1, -block_size[0]):
            serie[t] = i
            t += 1
    return serie

def vertical_down_to_up_start_right(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(block_mem_size-j-1, -1, -block_size[0]):
            serie[t] = i
            t += 1
    return serie

def vertical_up_to_down_start_right(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for j in xrange(block_size[0]):
        for i in xrange(block_size[0]-j-1, block_mem_size, block_size[0]):
            serie[t] = i
            t += 1
    return serie

def diagonal_start_left_up_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for i in xrange(block_size[0]):
        for j in xrange(i, i*block_size[0]+1, block_size[0]-1):
            serie[t]= j
            t = t+1
    for i in xrange(2*block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, block_mem_size, block_size[0]-1):
            serie[t] = j
            t = t+1
    return serie

def diagonal_start_left_up_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    serie[t] = 0
    t += 1
    for i in xrange(0, block_mem_size, block_size[1]):
        for j in xrange(i, 0, -(block_size[0]-1)):
            serie[t] = j
            t += 1
    s = 0
    for i in xrange(block_mem_size-block_size[0]+1, block_mem_size, 1):
        s += 1
        for j in xrange(i, s * block_size[0], -(block_size[0]-1) ):
            serie[t] = j
            t += 1
    return serie


def diagonal_start_right_up_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    s = 1
    for i in xrange(block_size[0]-1, -1, -1):
        for j in xrange(i, s * block_size[0], (block_size[0]+1)):
            serie[t] = j
            t += 1
        s += 1
    for i in xrange(block_size[0], block_mem_size, block_size[0]):
        for j in xrange(i, block_mem_size, (block_size[0]+1) ):
            serie[t] = j
            t += 1
    return serie

def diagonal_start_right_up_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = 0
    for i in xrange(block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, -1, -(block_size[0]+1)):
            serie[t] = j
            t += 1
    s = 1
    for i in xrange(block_mem_size-2, block_mem_size-block_size[0]-1, -1):
        for j in xrange(i, s * block_size[0]-1, -(block_size[0]+1) ):
            serie[t] = j
            t += 1
        s += 1
    return serie

def diagonal_start_left_down_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    r = diagonal_start_right_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie

def diagonal_start_left_down_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    r = diagonal_start_right_up_dir_down(block_size)
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie

def diagonal_start_right_down_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    r = diagonal_start_left_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie

def diagonal_start_right_down_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    r = diagonal_start_left_up_dir_down(block_size)
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie

def horizontal_snake_start_up_left(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = horizontal_left_to_right_start_up(block_size)
    serie_up = horizontal_right_to_left_start_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def horizontal_snake_start_up_right(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = horizontal_left_to_right_start_up(block_size)
    serie_up = horizontal_right_to_left_start_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 1):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def horizontal_snake_start_down_left(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = horizontal_left_to_right_start_down(block_size)
    serie_up = horizontal_right_to_left_start_down(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def horizontal_snake_start_down_right(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = horizontal_left_to_right_start_down(block_size)
    serie_up = horizontal_right_to_left_start_down(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 1):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def vertical_snake_start_up_left(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = vertical_up_to_down_start_left(block_size)
    serie_up = vertical_down_to_up_start_left(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def vertical_snake_start_up_right(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = vertical_up_to_down_start_right(block_size)
    serie_up = vertical_down_to_up_start_right(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[1]):
        for j in xrange(block_size[0]):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def diagonal_snake_start_left_up_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = diagonal_start_left_up_dir_down(block_size)
    serie_up = diagonal_start_left_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[0]):
        for j in xrange(i, i*block_size[0]+1, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    s = 0
    for i in xrange(2*block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, block_mem_size, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_dn[t]
            else:
                serie[t]= serie_up[t]
            t += 1
        s += 1
    return serie

def diagonal_snake_start_left_up_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = diagonal_start_left_up_dir_down(block_size)
    serie_up = diagonal_start_left_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[0]):
        for j in xrange(i, i*block_size[0]+1, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    s = 0
    for i in xrange(2*block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, block_mem_size, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    return serie

def diagonal_snake_start_right_up_dir_up(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = diagonal_start_right_up_dir_down(block_size)
    serie_up = diagonal_start_right_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[0]):
        for j in xrange(i, i*block_size[0]+1, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    s = 0
    for i in xrange(2*block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, block_mem_size, block_size[0]-1):
            if(s % 2 == 0):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    return serie

def diagonal_snake_start_right_up_dir_down(block_size):
    block_mem_size = block_size[0] * block_size[1]
    serie_dn = diagonal_start_right_up_dir_down(block_size)
    serie_up = diagonal_start_right_up_dir_up(block_size)
    serie = [None] * block_mem_size
    t = 0
    s = 0
    for i in xrange(block_size[0]):
        for j in xrange(i, i*block_size[0]+1, block_size[0]-1):
            if(s % 2 == 1):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    s = 0
    for i in xrange(2*block_size[0]-1, block_mem_size, block_size[1]):
        for j in xrange(i, block_mem_size, block_size[0]-1):
            if(s % 2 == 1):
                serie[t]= serie_up[t]
            else:
                serie[t]= serie_dn[t]
            t += 1
        s += 1
    return serie

def diagonal_snake_start_left_down_dir_up(block_size):
    if(block_size[0] % 2 == 0):
        r = diagonal_snake_start_right_up_dir_down(block_size)
    else:
        r = diagonal_snake_start_right_up_dir_up(block_size)
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie 

def diagonal_snake_start_left_down_dir_down(block_size):
    if(block_size[0] % 2 == 1):
        r = diagonal_snake_start_right_up_dir_down(block_size)
    else:
        r = diagonal_snake_start_right_up_dir_up(block_size)
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie 

def diagonal_snake_start_right_down_dir_up(block_size):
    if(block_size[0] % 2 == 0):
        r = diagonal_snake_start_left_up_dir_down(block_size)
    else:
        r = diagonal_snake_start_left_up_dir_up(block_size)
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie 

def diagonal_snake_start_right_down_dir_down(block_size):
    if(block_size[0] % 2 == 1):
        r = diagonal_snake_start_left_up_dir_down(block_size)
    else:
        r = diagonal_snake_start_left_up_dir_up(block_size)
    block_mem_size = block_size[0] * block_size[1]
    serie = [None] * block_mem_size
    t = len(r)-1
    for i in r:
        serie[t] = i
        t -= 1
    return serie 

'''block_size = (4,4)
serie = diagonal_snake_start_right_down_dir_down(block_size)
for i in serie:
    print i'''
