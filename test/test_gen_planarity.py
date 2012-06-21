import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

import random

from tester import compare_images, run_tests, skip_test
from tester import get_test_options, parse_test_options

from blip.code.codegen import Code
from blip.simulator.interpreter import Interpreter

from planarity.filters import Filter, Filterbank
from planarity.gen_code import *


block_size = (32, 32)


# ====================================================================================
def gen_filter():
	return Filter([(2, 3, -0.25), (6, 3, 0.25), (2, 5, 0.25), (6, 5, -0.25)], 9, 9)

def gen_random_image(rows, cols):
	return [[float(random.randint(0, 255)) for _ in xrange(cols)] for _ in xrange(rows)]


def run_codegen_function(test_image, code_gen, block_size, args, buffer_sel = 1, **kwargs):
	image2buffer = kwargs['image2buffer'] if 'image2buffer' in kwargs else {}

	im_size = len(test_image[0]), len(test_image)
	pe_dim = [s//b for s,b in zip(im_size, block_size)]
	# fill this in for all functions
	args['pe_dim'] = pe_dim

	code = Code()
	code.set_generator(code_gen, block_size, args)

	sim = Interpreter(code, test_image, block_size)
	for buffer_nr, image in image2buffer.iteritems():
		sim.set_src_image(image, buffer_nr)

	sim.run()

	return sim.gen_output_image(buffer_sel, False, False, True), sim


# ====================================================================================
def test_intern_image2buffer_func():
	''' Internal test for multiple buffer copy. '''
	rows, cols = 32, 64
	d1 = gen_random_image(rows, cols)
	d2 = gen_random_image(rows, cols)
	def code_gen(code, block_size, args):
		yield Nop()
	test_res, _ = run_codegen_function(d1, code_gen, block_size, {}, 1, image2buffer={1:d2})
	assert d2 == test_res

def test_gen_apply_sparse_filter():
	''' Test sparse filter codegen. '''
	from planarity.reference import _apply_filter
	rows, cols = 32, 64
	image = gen_random_image(rows, cols)
	f = gen_filter()
 
	ref_res = _apply_filter(image, f)
	test_res, _ = run_codegen_function(image, gen_apply_sparse_filter, block_size, {'filter': f}, 1)

	assert ref_res == test_res

def test_gen_gather_local_max():
	''' Test the code for local maximum calculation. '''
	from planarity.reference import _gather_local_max
	rows, cols = 32, 64 
	image = gen_random_image(rows, cols)
	f = gen_filter()
 
	ref_res = _gather_local_max(image, f)
	test_res, sim = run_codegen_function(image, gen_gather_local_max, block_size, {'filter':f}, 1)

	assert ref_res == test_res

def test_gen_global_max():
	''' Test global maximum calculation. '''
	from planarity.reference import _elementwise_max
	rows, cols = 32, 32
	image = gen_random_image(rows, cols)
	image2 = gen_random_image(rows, cols)

	args = {'in_ptr_1':0, 'in_ptr_2':block_size[0]*block_size[1]}
	test_res, _ = run_codegen_function(image, gen_global_max, block_size, args, 1, image2buffer={1:image2})
	ref_res = _elementwise_max(image, image2)

	assert ref_res == test_res

def test_gen_abs_value():
	''' Test absolute value code generation. '''
	rows, cols = 32, 64
	image = gen_random_image(rows, cols)

	test_res, _ = run_codegen_function(image, gen_abs_value, block_size, {}, 1)
	ref_res = [[abs(x) for x in y] for y in image]

	assert ref_res == test_res

def _test_gen_calc_impl(codegen_impl):
	from planarity.reference import calc_planarity
	rows, cols = 32, 64
	image = gen_random_image(rows, cols)

	filterbank_filename = '../data/asym_16_3_opencv.xml'
	filterbank = Filterbank.load(filterbank_filename)

	test_res, _ = run_codegen_function(image, codegen_impl, block_size, {'filterbank':filterbank}, 1)
	ref_res = calc_planarity(image, filterbank.filters)

	assert ref_res == test_res

def test_gen_calc_planarity():
	''' Test full planarity calculation. '''
	_test_gen_calc_impl(gen_calc_planarity)

def test_gen_calc_planarity_inlined():
	''' Test full planarity calculation, optimised inlined version. '''
	_test_gen_calc_impl(gen_calc_planarity_inlined)

def test_gen_calc_planarity_opt():
	''' Test full planarity calculation. '''
	_test_gen_calc_impl(gen_calc_planarity_opt)

def test_gen_calc_planarity_inlined_opt():
	''' Test full planarity calculation, optimised inlined version. '''
	_test_gen_calc_impl(gen_calc_planarity_inlined_opt)

# ====================================================================================
# eval 
def all_test(options = {}):
	tests = [\
		test_gen_apply_sparse_filter,\
		test_gen_gather_local_max,\
		test_gen_global_max,\
		test_gen_abs_value,\
		test_gen_calc_planarity,\
		test_gen_calc_planarity_opt,\
		test_intern_image2buffer_func,\
		test_gen_calc_planarity_inlined,\
		test_gen_calc_planarity_inlined_opt,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

