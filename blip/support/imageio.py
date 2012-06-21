opencv_available = True
try:
    import cv
except:
    print 'no valid opencv found'
    opencv_available = False

def is_ipl_float_type(image):
    return image.depth in [cv.IPL_DEPTH_32F, cv.IPL_DEPTH_64F]

def from_opencv_image(image):
    if not opencv_available:
        raise Exception('no valid open cv found')

    height, width = image.height, image.width
    nChannels = image.channels
    
    is_float_type = is_ipl_float_type(image) 

    res = []
    for i in xrange(height):
        res_row = []
        for j in xrange(width):
            if nChannels == 1:
                if is_float_type:
                    res_row.append(image[i,j])
                else:
                    res_row.append(int(image[i,j]))
            else:
                for k in xrange(nChannels):
                    if is_float_type:
                        res_row.append(image[i,j][k])
                    else:
                        res_row.append(int(image[i,j][k]))
        res.append(res_row)
    return res, nChannels

def to_opencv_image(image, channels):
    if not opencv_available:
        raise Exception('no valid open cv found')

    height = len(image)
    width = len(image[0])/channels
    
    im_type = type(image[0][0])
    depth = cv.IPL_DEPTH_8U if im_type == int else cv.IPL_DEPTH_32F
    res = cv.CreateImage((width, height), depth, channels)
    for i in xrange(height):
        for j in xrange(width):
            if channels == 1:
                if im_type == int:
                    cv.Set2D(res, i, j, int(image[i][j]))
                else:
                    cv.Set2D(res, i, j, image[i][j])
            else:
                for k in xrange(channels):
                    if im_type == int:
                        cv.Set2D(res, i, j, int(image[i][channels*j + k]))
                    else:
                        cv.Set2D(res, i, j, image[i][channels*j + k])
    return res


def from_opencv_mat(mat):
    if not opencv_available:
        raise Exception('no valid open cv found')

    res = []
    for i in xrange(mat.rows):
        res_row = []
        for j in xrange(mat.cols):
            if mat.channels == 1:
                res_row.append(mat[i,j])
            else:
                for k in xrange(mat.channels):
                    res_row.append(mat[i,j][k])
        res.append(res_row)
    return res, mat.channels

def to_opencv_mat(image, channels):
    if not opencv_available:
        raise Exception('no valid open cv found')

    height = len(image)
    width = len(image[0])/channels
    print 'to_opencv_mat'
    print width, height
    
    im_type = type(image[0][0])
    depth = cv.CV_8U if im_type == int else cv.CV_32F
    mat_type = cv.CV_MAKETYPE(depth, channels)
    res = cv.CreateMat(height, width, mat_type)
    for i in xrange(height):
        for j in xrange(width):
            if channels == 1:
                if im_type == int:
                    cv.Set2D(res, i, j, int(image[i][j]))
                else:
                    cv.Set2D(res, i, j, image[i][j])
            else:
                for k in xrange(channels):
                    if im_type == int:
                        cv.Set2D(res, i, j, int(image[i][channels*j + k]))
                    else:
                        cv.Set2D(res, i, j, image[i][channels*j + k])
    return res


# opencv io
def write_opencv(filename, im_data, channels):
    ''' saves an image using opencv '''
    if not opencv_available:
        raise Exception('no valid open cv found')

    height = len(im_data)
    width = len(im_data[0])/channels
    out = cv.CreateImage((width, height), cv.IPL_DEPTH_8U, channels)
    cv.SetZero(out)
    for i, row in enumerate(im_data):
        for j, v in enumerate(row):
                cv.Set2D(out, i, j, int(v))
    cv.SaveImage(filename, out)

def read_opencv(filename):
    ''' loads an image as grayscale using opencv '''
    if not opencv_available:
        raise Exception('no valid open cv found')

    cv_image = cv.LoadImage(filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
    image = [[cv.GetReal2D(cv_image,y,x) for x in xrange(cv_image.width)] for y in xrange(cv.height)]
    return image


# pure python png io
def write_png(filename, im_data, channels):
    ''' saves an image using pypng '''
    import png
    import array
    height = len(im_data)
    width = len(im_data[0])/channels
    writer = png.Writer(width=width, height=height, planes = channels, greyscale = (channels == 1))
    def yield_image():
        for row in im_data:
            yield array.array('B', row)
    out = open(filename, 'wb')
    writer.write(out, yield_image())
    out.close()

def read_png(filename):
    ''' loads an image as grayscale using pypng '''
    import png
    reader = png.Reader(filename)
    w, h, data, meta = reader.read()
    channels = meta['planes']
    plain_data = [[x for x in y] for y in data]

    if meta['greyscale']:
        return plain_data 
    else:
        # grayscale conversion
        return [[int(y[i]*0.299 + y[i+1]*0.587 + y[i+2]*0.114) for i in xrange(0, len(y), channels)] for y in plain_data]


# main interface
def write(filename, im_data, channels):
    ''' save an image '''
    height = len(im_data)
    width = len(im_data[0])/channels
    try:
        write_png(filename, im_data, channels)
    except:
        print 'pypng writer failed, trying opencv'
        try:
            write_opencv(filename, im_data, channels)
        except:
            print 'could not write image %s'%filename

def read(filename):
    ''' reads an image as grayscale '''
    image = None
    try:
        image = read_png(filename)
    except:
        print 'pypng reader failed, trying opencv'
        try:
            image = read_opencv(filename)
        except:
            print 'could not load image %s'%filename
    return image

