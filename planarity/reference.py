''' Reference implementation for flatness filters. '''
import math

from blip.support import visualisation
from blip.support import imageio

from planarity.filters import Filterbank


def _apply_filter(inputimage, filterobj):
	''' Basic convolution using sparse coefficients. '''
	rows, cols = len(inputimage), len(inputimage[0])
	frows, fcols = filterobj.size()
	hfrow, hfcol = frows//2, fcols//2
	out = [[0. for _ in xrange(cols)] for _ in xrange(rows)]
	for i in xrange(rows):
		for j in xrange(cols):
			acc = 0
			for x, y, v in filterobj.coefficients:
				ii = i + y - hfrow
				jj = j + x - hfcol
				if ii >= 0 and ii < rows and jj >= 0 and jj < cols:
					acc += inputimage[ii][jj] * v
			out[i][j] = acc
	return out

def _gather_local_max(inputimage, filterobj):
	''' Gather local max over neighborhood according to a certain binary mask. '''
	rows, cols = len(inputimage), len(inputimage[0])
	frows, fcols = filterobj.size()
	hfrow, hfcol = frows//2, fcols//2
	out = [[-float('inf') for _ in xrange(cols)] for _ in xrange(rows)]
	for i in xrange(rows):
		for j in xrange(cols):
			maxv = -float('inf') 
			for ii, mrow in enumerate(filterobj.mask):
				for jj, m in enumerate(mrow):
					if not m: continue # skip if mask is not valid here
					x = j + jj - hfcol
					y = i + ii - hfrow
					if x >= 0 and x < cols and y >= 0 and y < rows:
						maxv = max(maxv, inputimage[y][x])
					else:
						maxv = max(maxv, 0) # compliance with zero padding on blip
			out[i][j] = maxv
	return out

def _elementwise_max(m1, m2):
	rows, cols = len(m1), len(m1[0])
	total_max = [[0. for _ in xrange(cols)] for _ in xrange(rows)]
	for i in xrange(rows):
		for j in xrange(cols):
			total_max[i][j] = max(m1[i][j], m2[i][j])
	return total_max


def calc_planarity(inputimage, filters):
	''' Apply filters and max operator to image. '''

	rows, cols = len(inputimage), len(inputimage[0])
	total_max = [[-float('inf') for _ in xrange(cols)] for _ in xrange(rows)]
	for f in filters:
		response = _apply_filter(inputimage, f)
		# absolute value of response
		response = [[abs(x) for x in y] for y in response]
		
		# gather in region of interest
		local_max = _gather_local_max(response, f)
		total_max = _elementwise_max(local_max, total_max)

	return total_max

def _scale_to_integer(image, scaling = True, clipping = True, float_out = False):
	rows, cols = len(image), len(image[0])

	if scaling:
		minv = 9999999
		maxv = -999999
		for i, row in enumerate(image):
			for j, p in enumerate(row):
				print p
				v = p
				minv = min(minv, v)
				maxv = max(maxv, v)
		vrange = maxv-minv
		voffset = -minv
		vscale = 255./vrange if not vrange == 0 else 1.
		print 'minv %f maxv %f'%(minv, maxv)
	else:
		voffset = 0.
		vscale  = 1.

	out_image = [[0 for i in xrange(cols)] for j in xrange(rows)]
	for i, row in enumerate(image):
		for j, p in enumerate(row):
			v = p
			v = (v + voffset)*vscale
			if not float_out: v = int(v)
			if v > 255 and clipping:
				print 'clip v = %i to 255'%v
				v = 255
			out_image[i][j] = v
        return out_image

def main(filtersfile, inputimage, outputimage):
	''' Main entry function. '''
	filterbank = Filterbank.load(filtersfile)
	filters = filterbank.filters

	inputimage = imageio.read(inputimage)

	output = calc_planarity(inputimage, filters)

	output = _scale_to_integer(output)
	imageio.write(outputimage, output, 1)

	

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 4:
		print 'usage: %s flatnessxml inputimage outputimage'%sys.argv[0]
		exit(1)

	filtersfile, inputimage, outputimage = sys.argv[1:]

	try:
		main(filtersfile, inputimage, outputimage)
	except Exception, e:
		print str(e)
                pdb.post_mortem(sys.exc_traceback)
