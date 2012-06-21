import sys
import os
from violajones import equalizeHist, gen_integral_image
from violajones import calc_integral_value, get_variance, gen_integral_squared_image
import parse_haar
import imageio
import visualisation


def detect_faces_idle_raport(image, haar_classifier):
	height, width = len(image), len(image[0])
	haar_width, haar_height = haar_classifier.size

	reject_stage = [[255 for x in xrange(width)] for y in xrange(height)]

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
			for istage, stage in enumerate(haar_classifier.stages):
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
					if reject_stage[i][j] == 255:
						reject_stage[i][j] = istage
					break
			if not nodetect:
				detection = (j, i, haar_width, haar_height)
				print detection
				detections.append(detection)

		progress = 100.*float(i)/(height-haar_height)
		if progress - old_progress > 5.:
			old_progress = progress
			print 'progress: %.2f'%progress
	return detections, reject_stage

def test():
	pass

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-t', '--test', action='store_true', dest='do_test', default=False, help='execute test code')
	parser.add_option('-s', '--src_dir', dest='src_dir', default='.', help='source dir')
	parser.add_option('-o', '--output_dir', dest='output_dir', default='out', help='output dir')
	parser.add_option('-c', '--cascade', dest='cascade', default='data/haarcascade_frontalface_alt.xml', help='haar cascade')
        parser.add_option('-m', '--multi_scale', action='store_true', dest='multi_scale', default=False, help='multiscale detection')
	(options, args) = parser.parse_args()
	
	(options, args) = parser.parse_args()

	if options.do_test:
		test()
		exit(0)

	cascade_filename = options.cascade
	src_dir = options.src_dir
	output_dir = options.output_dir
	use_multiscale = options.multi_scale


	haar_classifier = parse_haar.parse_haar_xml(cascade_filename)
	print str(haar_classifier)

	# get all images in dir
	image_filenames = [src_dir + os.path.sep + x for x in os.listdir(src_dir) if '.png' in x.lower()]

	# parameters
	scale_factor = 1.2
	min_size = (40, 40)

	if not os.path.isdir(output_dir):
		print 'warning: creating outputdir %s'%output_dir
		os.mkdir(output_dir)

	for i, image_filename in enumerate(image_filenames):
		image = imageio.read(image_filename)

		# process image
		detected_faces = []
		reject_stage = None
		if use_multiscale:
			raise Exception('need to implement multiscale edit')
			detected_faces, reject_stage = detect_faces_multiscale(image, haar_classifier, scale_factor, min_size)
		else:
			detected_faces, reject_stage = detect_faces_idle_raport(image, haar_classifier)

		res = visualisation.draw_faces(image, detected_faces)

		imageio.write(output_dir + os.path.sep + 'res_%06i.png'%i, res, 3)

		imageio.write(output_dir + os.path.sep + 'reject_%06i.png'%i, reject_stage, 1)

		f = open(output_dir + os.path.sep + 'detections_%06i.txt'%i, 'w')
		f.write('\n'.join(str(x) for x in detected_faces)) 
		f.close()


