
import math

from violajones import parse_haar
from blip.support import visualisation
from blip.support import imageio

opencv_available = True
try:
	import cv
except:
	print 'no valid opencv found'
	opencv_available = False


def gen_integral_image(image):
	height, width = len(image), len(image[0])
	int_im = [[0. for x in range(width)] for y in range(height)]

	# use double pass, can be accelerated (see VJ paper)
	for i in range(height):
		for j in range(width):
			if j > 0:
				int_im[i][j] += (float(image[i][j]) + float(int_im[i][j-1]))
			else:
				int_im[i][j] = float(image[i][j])
	for j in range(width):
		for i in range(height):
			if i > 0:
				int_im[i][j] += float(int_im[i-1][j])
	return int_im

def gen_integral_squared_image(image):
	sq_image  = [[float(x*x) for x in y] for y in image]
	return gen_integral_image(sq_image) 

def calc_integral_value(im, x, y, width, height):
	''' extract value from integral image
	note that this calculation mirrors the opencv implementation
	so the actual width is width+1, height+1 as these values are inclusive 
	'''
	w, h = width, height
	v1 = im[y  ][x  ]
	v2 = im[y  ][x+w]
	v3 = im[y+h][x  ]
	v4 = im[y+h][x+w]
	return v1 - v2 - v3 + v4

def get_variance(x, y, int_im, int_sq_im, win_width, win_height):
#        size_t pOffset = pt.y * (sum.step/sizeof(int)) + pt.x;
#        size_t pqOffset = pt.y * (sqsum.step/sizeof(double)) + pt.x;
#        int valsum = CALC_SUM(p, pOffset);
#        double valsqsum = CALC_SUM(pq, pqOffset);
#        double nf = (double)normrect.area() * valsqsum - (double)valsum * valsum;
#        if( nf > 0. )
#            nf = sqrt(nf);
#        else
#            nf = 1.;
#        varianceNormFactor = 1./nf;
#        offset = (int)pOffset;
	w, h = win_width, win_height
	val_sum = calc_integral_value(int_im, x, y, w, h)
	val_sq_sum = calc_integral_value(int_sq_im, x, y, w, h)
	nf = float(w*h)* val_sq_sum - float(val_sum*val_sum)

	return (math.sqrt(float(nf)) if nf > 0. else 1.)


def equalizeHist(image):
	''' based on opencv implementation '''
	height, width = len(image), len(image[0])

	hist = [0 for i in range(256)]
	for row in image:
		for v in row:
			hist[v]+= 1
	scale = 255./(height*width)
	lut = [0 for i in range(len(hist)+1)]
	hsum = 0
	for i, h in enumerate(hist):
		hsum += h
		v = int(round(float(hsum)*scale))
		lut[i] = v
	lut[0] = 0

	return [[lut[x] for x in y] for y in image]

def scale_image(image, channels, scale):
	height, width = len(image), len(image[0])
	scale_width = round(width/scale)
	scale_height = round(height/scale)

	if not opencv_available:
		raise Exception('scale not implemented yet without opencv')


	scaled_image = cv.CreateImage((scale_width, scale_height), cv.IPL_DEPTH_8U, channels)	
	cv.Resize(imageio.to_opencv_image(image, channels), scaled_image)
	res, _ = imageio.from_opencv_image(scaled_image)
	return res
	

def detect_faces_multiscale(image, haar_classifier, scale_factor, min_size):
	height, width = len(image), len(image[0])
	min_width, min_height = min_size
	scale = 1.
	all_faces = []
	while 1:
		sc_height = int(round(height/scale))
		sc_width = int(round(width/scale))
		if sc_height < min_height or sc_width < min_width:
			break

		sc_image = scale_image(image, 1, scale)
		faces_sc = detect_faces(sc_image, haar_classifier)

		# transform them back to original image coordinates
		faces = [(int(round(x*scale)), int(round(y*scale)), int(round(w*scale)), int(round(h*scale))) for x, y, w, h in faces_sc] 
		for f in faces:
			all_faces.append(f)
		scale *= scale_factor
	return all_faces
	

	
def detect_faces(image, haar_classifier):
	height, width = len(image), len(image[0])
	haar_width, haar_height = haar_classifier.size

	image = equalizeHist(image)

	int_im = gen_integral_image(image)
	int_sq_im = gen_integral_squared_image(image)

	# scan image with VJ window
	old_progress = 0
	detections = []
	for i in range(height-haar_height):
		for j in range(width-haar_width):
			var = get_variance(j, i, int_im, int_sq_im, haar_width, haar_height)
			var_norm = 1./var
			nodetect = False
			for stage in haar_classifier.stages:
				value = 0.
				for feat in stage.features:
					feat_val = 0.
					for shape in feat.shapes:
						pos, coeff = shape
						x, y, w, h = pos
						xx = j + x
						yy = i + y
						val_sum = calc_integral_value(int_im, xx, yy, w, h)
						feat_val += val_sum*coeff*var_norm
					# left val belongs to under threshold condition
					value +=  feat.left_val if feat_val < feat.threshold else feat.right_val

				# if accumulated value is too small
				# no face will be detected so go to next window
				if value < stage.stage_threshold: 
					nodetect = True
					break
			if not nodetect:
				detection = (j, i, haar_width, haar_height)
				print detection
				detections.append(detection)


		progress = 100.*float(i)/(height-haar_height)
		if progress - old_progress > 5.:
			old_progress = progress
			print 'progress: %.2f'%progress
	return detections



def main(image_filename, cascade_filename, res_filename, use_multiscale = False):
	image = imageio.read(image_filename)

	haar_classifier = parse_haar.parse_haar_xml(cascade_filename)
	print str(haar_classifier)


	# parameters
	scale_factor = 1.2
	min_size = (40, 40)

	# process image
	detected_faces = []
	if use_multiscale:
		detected_faces = detect_faces_multiscale(image, haar_classifier, scale_factor, min_size)
	else:
		detected_faces = detect_faces(image, haar_classifier)

	res = visualisation.draw_faces(image, detected_faces)
	imageio.write(res_filename, res, 3)
	


if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-s', '--src_image', dest='src_image', default='data/vakgroep128_64.png', help='source image')
	parser.add_option('-o', '--output', dest='output', default='violajones_out.png', help='output image')
	parser.add_option('-c', '--cascade', dest='cascade', default='data/haarcascade_frontalface_alt.xml', help='haar cascade')
        parser.add_option('-m', '--multi_scale', action='store_true', dest='multi_scale', default=False, help='multiscale detection')
	(options, args) = parser.parse_args()

	cascade_filename = options.cascade
	image_filename = options.src_image
	res_filename = options.output

	main(image_filename, cascade_filename, res_filename, options.multi_scale)

