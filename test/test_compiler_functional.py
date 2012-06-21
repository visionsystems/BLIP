import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

from tester import run_tests, OpMatcher, match_code, compare_images
from tester import get_test_options, parse_test_options

from blip.code.BlipCompiler import Compiler, KernelObject, CompilerDriver, BodyVisitor
from blip.code.codegen import InstrAdapter, Code
from blip.simulator.interpreter import Interpreter


# Boiler plate code
def compile_code(src_code, args, block_size, nr_reg = 8, no_sequencer=True):
	''' Compile a kernel and return a codegen function. '''
	comp = CompilerDriver(nr_reg, no_sequencer=True)
	main_object = comp.run(src_code)
	patched_object = Compiler.patch_arguments_before_run(main_object, [args[argname] for argname in main_object.arguments])
	def codegen_func(code, block_size, args):
		for x in patched_object.code:
			yield InstrAdapter(x, use_reg_wrapper=True)
	code = Code()
	code.set_generator(codegen_func, block_size, args)
	return code

def run_kernel(src_code, image_src, block_size, args, nr_reg = 8, no_sequencer=True):
	''' Run a kernel, this assumes that the output is in buffer #1 and output is unscaled float. '''
	code = compile_code(src_code, args, block_size, nr_reg, no_sequencer)

	sim = Interpreter(code, image_src, block_size, 4, nr_reg)
	sim.run_kernel()
	return sim.gen_output_image(1, False, False, False)



# Actual tests
def test_gray():
	''' Basic test, generate a gray image. '''
	block_size = (8, 8)

	im_size = tuple(x*2 for x in block_size)
	src_image = [[0 for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main():
	y = get_local_id(0)
	x = get_local_id(1)
	index = y*%(bwidth)i + x
	out_ptr = %(bwidth)i * %(bheight)i
	out_ptr[index] = 128
'''%{'bwidth':block_size[0], 'bheight':block_size[1]}

	test_out = run_kernel(src, src_image, block_size, {}, 16)
	ref_out = [[128. for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]
	assert compare_images(test_out, ref_out) < 0.001

def test_treshold():
	import random
	block_size = (8, 8)
	th = 100
	out_ptr = block_size[0]*block_size[1]

	im_size = tuple(x*2 for x in block_size)
	src_image = [[random.randint(0, 255) for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main(in_ptr, out_ptr, th):
	y = get_local_id(0)
	x = get_local_id(1)
	index = %(bwidth)i*y + x
	in_v = in_ptr[index]
	out_v = 255 if in_v > th else 0
	out_ptr[index] = out_v
'''%{'bwidth':block_size[0]}

	test_out = run_kernel(src, src_image, block_size, {'in_ptr':0, 'out_ptr':out_ptr, 'th':th}, 16)
	ref_out = [[255 if x > th else 0 for x in y] for y in src_image]
	assert compare_images(test_out, ref_out) < 0.001

def test_conv_3x1():
	''' Simple 1 dimensional convolution test. '''
	import random
	block_size = (8, 8)
	out_ptr = block_size[0]*block_size[1]

	im_size = tuple(x*1 for x in block_size) # only single block, this code doesn't account for interblock comm
	src_image = [[random.randint(0, 255) for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main(in_ptr, out_ptr):
	coeff = [-1, 0, 1]
	y = get_local_id(0)
	x = get_local_id(1)
	index = %(bwidth)i*y
	acc = 0
	current_x = x - 1
	for i in range(3):
		in_v = in_ptr[index+current_x]
		acc += (coeff[i] * in_v if current_x < %(bwidth)i else 0) if current_x >= 0 else 0
		current_x += 1
	out_ptr[index+x] = acc
'''%{'bwidth':block_size[0]}

	test_out = run_kernel(src, src_image, block_size, {'in_ptr':0, 'out_ptr':out_ptr}, 16)
	ref_out = [[0 for x in y] for y in src_image]
	coeffs = [-1, 0, 1]
	for i, src_row in enumerate(src_image):
		for j, src_px in enumerate(src_row):
			acc = 0
			for k, coeff in enumerate(coeffs):
				x = j + k -1
				if x >= 0 and x < len(src_row):
					acc += coeff * src_row[x]
			ref_out[i][j] = acc

	assert compare_images(test_out, ref_out) < 0.001

def test_get2D_func():
	''' Test if get2D codegen is correct. '''
	import random
	block_size = (5, 5)
	out_ptr = block_size[0]*block_size[1]

	im_size = tuple(x*3 for x in block_size) # only single block, this code doesn't account for interblock comm
	src_image = [[random.randint(0, 255) for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main(in_ptr, out_ptr):
	y = get_local_id(0)
	x = get_local_id(1)
	index = %(bwidth)i*y
	out_ptr[index+x] = get2D(in_ptr, x-1, y-1, %(bwidth)i, %(bheight)i)
'''%{'bwidth':block_size[0], 'bheight':block_size[1]}

	test_out = run_kernel(src, src_image, block_size, {'in_ptr':0, 'out_ptr':out_ptr}, 32)
	ref_out = [[0 for x in y] for y in src_image]
	for i, src_row in enumerate(src_image):
		for j, src_px in enumerate(src_row):
				x = j-1
				y = i-1
				if x >= 0 and x < im_size[0] and y >= 0 and y < im_size[1]:
					ref_out[i][j] = src_image[y][x]
	assert compare_images(test_out, ref_out) < 0.001

def test_conv_3x1_multiple_block():
	''' Simple 1 dimensional convolution test with multiple blocks. '''
	import random
	block_size = (8, 8)
	out_ptr = block_size[0]*block_size[1]

	im_size = tuple(x*2 for x in block_size) # only single block, this code doesn't account for interblock comm
	src_image = [[random.randint(0, 255) for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main(in_ptr, out_ptr):
	coeff = [-1, 0, 1]
	y = get_local_id(0)
	x = get_local_id(1)
	index = %(bwidth)i*y
	acc = 0
	current_x = x - 1
	for i in range(3):
		in_v = get2D(in_ptr, current_x, y, %(bwidth)i, %(bheight)i) 
		acc += coeff[i] * in_v
		current_x += 1
	out_ptr[index+x] = acc
'''%{'bwidth':block_size[0], 'bheight':block_size[1]}

	test_out = run_kernel(src, src_image, block_size, {'in_ptr':0, 'out_ptr':out_ptr}, 32)
	ref_out = [[0 for x in y] for y in src_image]
	coeffs = [-1, 0, 1]
	for i, src_row in enumerate(src_image):
		for j, src_px in enumerate(src_row):
			acc = 0
			for k, coeff in enumerate(coeffs):
				x = j + k -1
				if x >= 0 and x < len(src_row):
					acc += coeff * src_row[x]
			ref_out[i][j] = acc

	assert compare_images(test_out, ref_out) < 0.001

def test_conv_3x3_multiple_block():
	''' Simple 2 dimensional convolution test with multiple blocks.
	'''
	import random
	block_size = (8, 8)
	out_ptr = block_size[0]*block_size[1]

	im_size = tuple(x*2 for x in block_size) # only single block, this code doesn't account for interblock comm
	src_image = [[random.randint(0, 255) for _ in xrange(im_size[0])] for _ in xrange(im_size[1])]

	src = '''
@kernel
def main(in_ptr, out_ptr):
	coeff = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
	y = get_local_id(0)
	x = get_local_id(1)
	index = %(bwidth)i*y+x
	acc = 0
	current_y = y - 1
	for i in range(3):
		current_x = x - 1
		for j in xrange(3):
			current_coeff = coeff[i][j]
			in_v = get2D(in_ptr, current_x, current_y, %(bwidth)i, %(bheight)i)
			v = current_coeff * in_v
			acc += v
			current_x += 1
		current_y += 1
	out_ptr[index] = acc
'''%{'bwidth':block_size[0], 'bheight':block_size[1]}

	test_out = run_kernel(src, src_image, block_size, {'in_ptr':0, 'out_ptr':out_ptr}, 32)
	ref_out = [[0 for x in y] for y in src_image]
	coeffs = [[-1, 0, 1]]*3
	for i, src_row in enumerate(src_image):
		for j, src_px in enumerate(src_row):
			acc = 0
			for k, coeff_row in enumerate(coeffs):
				y = i + k - 1
				for l, coeff in enumerate(coeff_row):
					x = j + l -1
					if x >= 0 and x < len(src_row) and y >= 0 and y < len(src_image):
						acc += coeff * src_image[y][x]
			ref_out[i][j] = acc

	assert compare_images(test_out, ref_out) < 0.001

def all_test(options = {}):
	tests = [\
		test_gray,\
		test_treshold,\
		test_conv_3x1,\
		test_get2D_func,\
		test_conv_3x1_multiple_block,\
		test_conv_3x3_multiple_block,\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

