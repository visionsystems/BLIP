import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

import traceback
import pdb
import random

from violajones.parse_haar import HaarClassifier, HaarStage, HaarFeature 
from violajones import parse_haar
from violajones import reference


from blip.support import imageio

from blip.simulator.opcodes import *
from blip.simulator.interpreter import Interpreter
from blip.code.codegen import Code
from blip.code.trace_optimiser import Optimiser, MemoryPass, PeepholePass, ImmediatePass

from violajones import gen_code
from violajones.gen_code import InvalidBlocksizeException, IllegalInstructionException
from violajones.gen_code import optimiser_wrapper


from tester import compare_images, run_tests, skip_test
from tester import get_test_options, parse_test_options

# ====================================================================================
# tests

def test_split_filter_correctness():
	''' test correctness of split_filter '''
	block_size = 4
	pos = (1,1)
	h = HaarFeature([((0, 0, 5, 5), -1)], False, 0.01, 0.2, 0.8)
	res = gen_code.split_filter(h, pos, block_size)
	# this result may seem strange, but the pos offset needs to be added to get the correct result
	ref = [((0, 0, 3, 3), -1), ((0, 3, 3, 2), -1), ((3, 0, 2, 3), -1), ((3, 3, 2, 2), -1)]

	assert ref == res.shapes

def test_split_filter_invalid_blocksize():
	''' test blocksize argument handling '''
	h = HaarFeature([((0, 0, 5, 5), -1)], False, 0.01, 0.2, 0.8)
	try:
		res = gen_code.split_filter(h, (1,1), (4,5)) # blocksizes for w and h should be equal
	except InvalidBlocksizeException:
		assert True # this exception should be raised
		return
	assert False # should never get here

def test_all_points_in_same_block():
	block_size = (4,4)
	size = tuple(2*b for b in block_size)
	def gen_points(shape):
		x, y, w, h = shape
		return ((x, y), (x,y+h), (x+w, y), (x+w, y+h))
	test_shapes = [\
		((1, 1, 2, 2), True),\
		((5, 5, 2, 2), True),\
		((2, 2, 4, 4), False),\
		((2, 2, 1, 4), False)]
	for s, gt in test_shapes:
		if gen_code.all_points_in_same_block(gen_points(s), block_size) != gt:
			print s, gt, 'failed'
			assert False

def test_neighbouring_blocks():
	block_size = (4,4)
	test_blocks = [\
		([(0, 0), (0, 1), (5, 0), (5, 1)], True),\
		([(0, 0), (0, 5), (5, 0), (5, 1)], False),\
		([(0, 0), (3, 2), (5, 5), (5, 1)], False),\
		([(2, 2), (3, 2), (2, 6), (3, 6)], True),\
		([(1, 1), (2, 2), (6, 6), (7, 7)], False)
	]
	for p, gt in test_blocks:
		res = gen_code.all_points_in_two_neighbouring_blocks(p, block_size)
		if res != gt:
			print p, gt, 'failed'
			print 'res was', res
			assert False

def gen_integral_image_correctness():
	''' test if generated integral image is correct, 
	    note that this relies on the corectness of interpreter and reference.py '''

#		size = (120, 80)
#		block_size = (40, 40)
	size = (80, 80)
	block_size = size 

	# generate random test image
	test_image = [[float(random.randint(0, 255)) for i in xrange(size[0])] for j in xrange(size[1])]

	# reference implementation
	integral_ref = reference.gen_integral_image(test_image)
	sq_integral_ref = reference.gen_integral_squared_image(test_image)
	
	# pointer config
	buffer_size = block_size[0]*block_size[1]
	src_ptr = 0
	integral_ptr = buffer_size
	sq_integral_ptr = 2*buffer_size

	# set up interpreter for integral image calculation
	def code_gen(code, block_size, args):
		return gen_code.gen_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, block_size)
	code = Code()
	code.set_generator(optimiser_wrapper(code_gen), block_size)

	sim = Interpreter(code, test_image, block_size)
	sim.run()

	# get result of simulator with scaling, truncation turned off and float output
	integral_test = sim.gen_output_image(1, False, False, True)
	sq_integral_test = sim.gen_output_image(2, False, False, True)

	# comparison of reference with blip sim
	integral_err = compare_images(integral_ref, integral_test)
	sq_integral_err = compare_images(sq_integral_ref, sq_integral_test)

	err_eps = 0.001
	if not ((integral_err < err_eps) and (sq_integral_err < err_eps)):
		print 'integral comp:', integral_err
		print 'squared integral comp:', sq_integral_err 
		assert False

def test_full_integral_image_correctness():
	''' Test generated full integral image correctness,
	    note that this relies on the corectness of interpreter and reference.py '''

	block_size = (20, 20)
	size = tuple(x*3 for x in block_size)

	# generate random test image
	test_image = [[float(random.randint(0, 255)) for i in xrange(size[0])] for j in xrange(size[1])]

	# reference implementation
	integral_ref = reference.gen_integral_image(test_image)
	sq_integral_ref = reference.gen_integral_squared_image(test_image)

	# pointer config
	buffer_size = block_size[0]*block_size[1]
	src_ptr = 0
	integral_ptr = buffer_size
	sq_integral_ptr = 2*buffer_size

	# set up interpreter for integral image calculation
	pe_dim = [s//b for s, b in zip(size, block_size)]
	def code_gen(code, block_size, args):
		return gen_code.gen_full_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, pe_dim, block_size)
	code = Code()
	code.set_generator(optimiser_wrapper(code_gen), block_size)

	sim = Interpreter(code, test_image, block_size)
	sim.run()

	# get result of simulator with scaling, truncation turned off and float output
	integral_test = sim.gen_output_image(1, False, False, True)
	sq_integral_test = sim.gen_output_image(2, False, False, True)

	# comparison of reference with blip sim
	integral_err = compare_images(integral_ref, integral_test)
	sq_integral_err = compare_images(sq_integral_ref, sq_integral_test)

	err_eps = 0.001
	if not ((integral_err < err_eps) and (sq_integral_err < err_eps)):
		print 'integral comp:', integral_err
		print 'squared integral comp:', sq_integral_err
		
		print 'rendering instruction stream to file, can take a while'
		try:
			f = open('unoptimised_full_integral_image_trace.txt', 'w')
			def tag_str(instr): return ', '.join(instr.tag) if hasattr(instr, 'tag') else ''
			f.write('\n'.join(str(x).ljust(40) + ' tags: ' + tag_str(x) for x in code_gen(Code())))
			f.close()

			optim_gen = optimiser_wrapper(code_gen, block_size, {})
			f = open('bad_full_integral_image_trace.txt', 'w')
			def tag_str(instr): return ', '.join(instr.tag) if hasattr(instr, 'tag') else ''
			f.write('\n'.join(str(x).ljust(40) + ' tags: ' + tag_str(x) for x in optim_gen(Code())))
			f.close()
		except Exception, e:
			print 'could render instruction stream to file'
			print 'err: ' + str(e)
		
		assert False

def test_integral_sum():
	block_size = (12,12)
	size = block_size # this way there's only one PE, so we only check gen_integral_sum
	test_image = [[random.random() for i in xrange(size[0])] for  j in xrange(size[1])]

	def run_test(image, position, shape, block_size):
		code = Code()
		out_reg = code.alloc_reg()
		def code_gen(code, block_size, args): 
			return gen_code.gen_integral_sum(code, out_reg, position, shape, ptr, block_size)
		code.set_generator(optimiser_wrapper(code_gen), block_size)

		sim = Interpreter(code, image, block_size)
		sim.run()
		# extract value
		return sim.procs[0][0].get_reg_by_name(str(out_reg))

	def run_ref(image, position, shape):
		px, py = position
		x, y, w, h = shape
		xx = x + px
		yy = y + py
		return reference.calc_integral_value(image, xx, yy, w-1, h-1)
	
	# set up interpreter for integral image calculation
	ptr = 0
	max_shape_width = 6
	max_shape_height = 6
	for i in xrange(size[1] - max_shape_width):
		for j in xrange(size[0] - max_shape_height):
			position = (j, i)
			shape = (0, 0, random.randint(1, max_shape_width), random.randint(1, max_shape_height))
			test_res = run_test(test_image, position, shape, block_size)
			ref_res = run_ref(test_image, position, shape)
			assert test_res == ref_res

def test_gen_fullintegral_sum2_2():
	block_size = (12,12)
	size = tuple(b*2 for  b in block_size)
	image = [[float(random.randint(0, 255)) for i in xrange(size[0])] for  j in xrange(size[1])]
	test_image = reference.gen_integral_image(image)
	ptr = 0

	def run_test(image, position, shape, ptr, block_size):
		px, py = position
		x, y, w, h = shape
		xx = px + x
		yy = py + y
		points =  ((xx, yy), (xx+w-1, yy), (xx, yy+h-1), (xx+w-1, yy+h-1))

		def code_gen(code, block_size, args):
			return gen_code.gen_fullintegral_sum2_2(code, code.r(4), ptr, points, block_size)
		code = Code()
		code.set_generator(optimiser_wrapper(code_gen), block_size)

		sim = Interpreter(code, image, block_size)
		sim.run()
		# extract value
		return sim.procs[0][0].get_reg_by_name('r4')

	def run_ref(image, position, shape):
		px, py = position
		x, y, w, h = shape
		xx = x + px
		yy = y + py
		return reference.calc_integral_value(image, xx, yy, w-1, h-1)
	shape = (0, 0, 6, 6)
	# cover the 4 position for which this function is valid
	positions = [(8, 0), (0, 8), (8, 14), (14, 8)]
	failed = False
	for position in positions:
		test_res = run_test(test_image, position, shape, ptr, block_size)
		ref_res = run_ref(test_image, position, shape)
		if test_res != ref_res:
			print 'position', position, 'failed, ref = ', ref_res, 'res = ', test_res
			failed = True
	assert not failed

def test_fullintegral_sum():
	block_size = (12,12)
	size = tuple(b*3 for  b in block_size)
	image = [[float(random.randint(0, 255)) for i in xrange(size[0])] for  j in xrange(size[1])]
	test_image = reference.gen_integral_image(image)

	def run_test(image, position, shape, block_size):
		def code_gen(code, block_size, args):
			return gen_code.gen_fullintegral_sum(code, code.r(4), position, shape, ptr, block_size)
		code = Code()
		code.set_generator(optimiser_wrapper(code_gen), block_size)

		sim = Interpreter(code, image, block_size)
		sim.run()
		# extract value
		return sim.procs[0][0].get_reg_by_name('r4')

	def run_ref(image, position, shape):
		px, py = position
		x, y, w, h = shape
		xx = x + px
		yy = y + py
		return reference.calc_integral_value(image, xx, yy, w-1, h-1)
	
	# set up interpreter for integral image calculation
	ptr = 0
	max_shape_width = 6
	max_shape_height = 6
	success = True
	for i in xrange(block_size[1]):
		for j in xrange(block_size[0]):
			position = (j, i)
			shape = (0, 0, random.randint(1, max_shape_width), random.randint(1, max_shape_height))
			test_res = run_test(test_image, position, shape, block_size)
			ref_res = run_ref(test_image, position, shape)
			if test_res != ref_res:
				print 'error on', position, 'ref = ', ref_res, 'res = ', test_res
				success = False
	assert success

def test_single_variance_calculation():
	block_size = (12,12)
	size = block_size # this way there's only one PE, so we only check gen_integral_sum
	integral_test = [[random.random() for i in xrange(size[0])] for  j in xrange(size[1])]
	sq_integral_test = [[random.random() for i in xrange(size[0])] for  j in xrange(size[1])]

	def run_test(position, integral_test, sq_integral_test, haar_size, block_size):
		integral_ptr = 0
		sq_integral_ptr = block_size[0]*block_size[1]

		code = Code()
		out_reg = code.alloc_reg()
		def code_gen(code, block_size, args):
			return gen_code.gen_calc_variance(code, out_reg, position, integral_ptr, sq_integral_ptr, haar_size, block_size)
		code.set_generator(optimiser_wrapper(code_gen), block_size)

		sim = Interpreter(code, integral_test, block_size)
		# hack: in order to avoid calculating integral images, inject random values into the sq_integral buffer
		# this is easy since their is only a single PE
		for i, row in enumerate(sq_integral_test):
			for j, v in enumerate(row):
				sim.procs[0][0].memory.set(sq_integral_ptr + len(row)*i+j, v)

		sim.run()

		pe = sim.procs[0][0]
		# extract value
		return (1./(pe.get_reg_by_name(str(out_reg)))), pe

	def run_ref(position, integral_test, sq_integral_test, haar_size):
		x, y = position
		haar_width, haar_height = haar_size
		var = reference.get_variance(x, y, integral_test, sq_integral_test, haar_width, haar_height)
		var_norm = 1./var
		return var_norm
	
	# set up interpreter for integral image calculation
	ptr = 0
	max_haar_size = 6
	for i in xrange(size[1] - max_haar_size):
		for j in xrange(size[0] - max_haar_size):
			position = (j, i)
			rand_size = random.randint(1, max_haar_size)
			haar_size = (rand_size, rand_size)
			test_res, pe = run_test(position, integral_test, sq_integral_test, haar_size, block_size)
			ref_res = run_ref(position, integral_test, sq_integral_test, haar_size)
			if test_res != ref_res:
				print '%i, %i'%(j, i)
				print 'test_res: %f\nref_res: %f'%(test_res, ref_res)
				raise Exception('mismatch')

def gen_faces_detect_no_errors():
	''' just check if codegen generates no errors '''

	block_size = (24,24)
	size = tuple([x*2 for x in block_size])

	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	cascade = parse_haar.parse_haar_xml(cascade_filename)

	args = {'haar_classifier':cascade}
	code = Code()
	code.set_generator(gen_code.gen_detect_faces_opt, block_size, args)

	# extract some data from simulator
	sim = Interpreter(code, [[0 for x in xrange(size[0])] for y in xrange(size[1])], block_size)
	nr_reg = sim.procs[0][0].nr_reg
	mem_size = sim.procs[0][0].memory.size
	del sim

	# run through instructions and apply some checks
	cnt = 0
	current_imm = 0
	for x in code.gen(code):
		if x.opcode() == 'imm':
			current_im = x.value
		# heuristic test, not all errors are detected as memw(rx, ry) is also possible
		if (x.opcode() == 'memr' and str(x.src) == 'imm') \
		or (x.opcode() == 'memw' and str(x.dest) == 'imm'):
			if current_im >= mem_size:
				raise IllegalInstructionException('memx out of bounds, addr = %i'%int(current_im))
		cnt+=1

	print '# instructions: %i'%cnt

@skip_test # skip this test until the calculation of read points is fixed
def test_detect_faces():
	''' check if whole program works '''
	# settings
	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	image_filename = '../data/vakgroep128_64.png'

	def run_test(image, cascade):
		block_size = (64, 64)

		print 'XXX histogram equalisation is not implemented yet, use violajones impl'
		print '    before executing simulator'
		image = reference.equalizeHist(image)

		args = {'haar_classifier':cascade}
		# now execute the codegen
		code = Code()
		code.set_generator(gen_code.gen_detect_faces_opt, block_size, args)
		#print '# instructions: %i'%(code.instr_size())

		sim = Interpreter(code, image, block_size, 4)
		sim.run()

		detections_pixmap = sim.gen_output_image(1) # result is saved in first buffer

		# convert the number of rejections in the stages to detections
		detections = gen_code.convert_pixelmap_to_detections(detections_pixmap, cascade.size)
		return detections

	def run_ref(image, cascade):
		return reference.detect_faces(image, cascade)

	# load image and cascade

	# first load the cascade
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	print cascade

	image = imageio.read(image_filename)
	if not image: raise Exception('image %s not found or not supported'%image_filename)

	detections_test = run_test(image, cascade)
	detections_ref = run_ref(image, cascade)
	assert detections_test == detections_ref

def test_detect_faces_fullintegral():
	''' check if whole program works '''
	# settings
	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	image_filename = '../data/vakgroep128_64.png'

	def run_test(image, cascade):
		block_size = (64, 64)
		im_size = len(image[0]), len(image)
		pe_dim = tuple(s//b for s, b in zip(im_size, block_size))

		print 'XXX histogram equalisation is not implemented yet, use violajones impl'
		print '    before executing simulator'
		image = reference.equalizeHist(image)

		args = {'haar_classifier': cascade, 'pe_dim':pe_dim}
		# now execute the codegen
		code = Code()
		code.set_generator(gen_code.gen_detect_faces_fullintegral_opt, block_size, args)
		#print '# instructions: %i'%(code.instr_size())

		sim = Interpreter(code, image, block_size, 4)
		sim.run()

		detections_pixmap = sim.gen_output_image(1) # result is saved in first buffer

		# convert the number of rejections in the stages to detections
		detections = gen_code.convert_pixelmap_to_detections(detections_pixmap, cascade.size)
		return detections

	def run_ref(image, cascade):
		return reference.detect_faces(image, cascade)

	# first load the cascade
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	print cascade

	image = imageio.read(image_filename)
	if not image: raise Exception('image %s not found or not supported'%image_filename)

	detections_test = run_test(image, cascade)
	detections_ref = run_ref(image, cascade)
	assert detections_test == detections_ref
		
@skip_test
def test_compare_implementations():
	''' check if the to implementation of detect_faces yield the same result '''
	# settings
	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	image_filename = '../data/vakgroep128_64.png'

	def run_test(codegen_function, image, cascade, block_size):
		print 'running %s'%codegen_function.__name__
		print 'XXX histogram equalisation is not implemented yet, use violajones impl'
		print '    before executing simulator'
		image = reference.equalizeHist(image)

		width, height = block_size
		pe_dim = (len(image[0])//width, len(image)//height)

		args = {'haar_classifier': cascade, 'pe_dim':pe_dim}
		# now execute the codegen
		code = Code()
		code.set_generator(optimiser_wrapper(codegen_function), block_size, args)

		sim = Interpreter(code, image, block_size, 4)
		sim.run()

		detections_pixmap = sim.gen_output_image(1) # result is saved in first buffer

		# convert the number of rejections in the stages to detections
		detections = gen_code.convert_pixelmap_to_detections(detections_pixmap, cascade.size)
		return detections


	# load image and cascade
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	print cascade

	image = imageio.read(image_filename)
	if not image: raise Exception('image %s not found or not supported'%image_filename)


	block_size = (64, 64)
	implementations = [\
		gen_code.gen_detect_faces_opt,\
		gen_code.gen_detect_faces_stage_outer_opt,\
		gen_code.gen_detect_faces_fullintegral_opt]
	detections = [run_test(impl, image, cascade, block_size) for impl in implementations]
	for i in xrange(len(implementations)-1):
		d1 = detections[i]
		d2 = detections[i+1]
		n1 = implementations[i].__name__
		n2 = implementations[i+1].__name__
		assert d1 == d2

def test_no_manual_alloc():
	''' Manual register allocation interferes with scoped_alloc. '''

	block_size = (22,22)
	size = tuple(x*2 for x in block_size)
	width, height = block_size
	pe_dim = (size[0]//width, size[1]//height)

	cascade_filename = '../data/haarcascade_frontalface_alt.xml'
	cascade = parse_haar.parse_haar_xml(cascade_filename)

	class TestCode(Code):
		def r(self, nr):
			raise Exception('Manual allocation')

	args = {'haar_classifier':cascade}
	code = TestCode()
	code.set_generator(gen_code.gen_detect_faces_opt, block_size, args)

	cnt = 0
	for x in code.gen(code): cnt += 1
	print cnt

	args = {'haar_classifier':cascade, 'pe_dim':pe_dim}
	code2 = TestCode()
	code.set_generator(gen_code.gen_detect_faces_fullintegral_opt, block_size, args)

	cnt2 = 0
	for x in code.gen(code): cnt2 += 1
	print cnt2

# ====================================================================================
# eval 
def all_test(options = {}):
	tests = [test_split_filter_correctness, \
		 test_split_filter_invalid_blocksize,\
		 test_all_points_in_same_block,\
		 test_neighbouring_blocks,\
		 gen_integral_image_correctness,\
		 test_full_integral_image_correctness,\
	 	 test_integral_sum, \
		 test_gen_fullintegral_sum2_2,\
		 test_fullintegral_sum, \
		 test_single_variance_calculation,\
		 #test_no_manual_alloc,\
		 #gen_faces_detect_no_errors, \
		 test_detect_faces_fullintegral,\
		 test_detect_faces\
		 #test_compare_implementations\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

