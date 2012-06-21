from __future__ import with_statement
import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

import random
from tester import run_tests, compare_images, skip_test
from tester import get_test_options, parse_test_options

from blip.code.codegen import Code, scoped_alloc
from blip.code.skeletons import *
from blip.simulator.opcodes import *
from blip.simulator.interpreter import Interpreter

# test of skeletons
def test_map_pixel_to_pixel():
	def pixel_op(code, pixel_in, pixel_out, args, block_size):
		th = args['th']
		with scoped_alloc(code, 3) as (th_r, v, const_255):
			yield Imm(th_r, th)
			yield Cmp(pixel_in, th_r)
			yield Imm(const_255, 255)
			yield Mov(pixel_out, const_255, cond='GT')
			yield Xor(pixel_out, pixel_out, pixel_out, cond='LE')

	def run_test(image, th):
		code = Code()
		block_size = (16, 16)
		in_ptr = 0
		out_ptr = block_size[0]*block_size[1]
		args = {'th': th}
		def codegen(code, block_size, args): 
			return map_pixel_to_pixel(code,  in_ptr, out_ptr, pixel_op, args, block_size)
		code.set_generator(codegen, block_size, args)
		sim = Interpreter(code, image, block_size)
		sim.run()
		return sim.gen_output_image(1)
	def run_ref(image, th):
		return [[255 if x > th else 0 for x in y] for y in image]

	th = 100
	image = [[random.randint(0, 255) for x in xrange(16)] for y in xrange(16)]
	res_test = run_test(image, th)
	res_ref  = run_test(image, th)

	assert compare_images(res_test, res_ref) < 0.01

def test_map_neighborhood_to_pixel():
	def pixel_op(code, mask_val, image_val, acc, args, block_size):
		''' Simple convolution implementation. '''
		with scoped_alloc(code, 2) as (v, mask_val_r):
			yield Imm(mask_val_r, mask_val)
			yield Mul(v, mask_val_r, image_val)
			yield Add(acc, acc, v)

	def run_test(image, coeff):
		code = Code()
		block_size = (16, 16)
		in_ptr = 0
		out_ptr = block_size[0]*block_size[1]
		args = {}
		def codegen(code, block_size, args):
			return map_neighborhood_to_pixel(code, in_ptr, out_ptr, coeff, pixel_op, args, block_size)
		code.set_generator(codegen, block_size, args)
		sim = Interpreter(code, image, block_size)
		sim.run()
		return sim.gen_output_image(1)
	def run_ref(image, coeff):
		iwidth, iheight = len(image[0]), len(image)
		res = [[0 for x in xrange(iwidth)] for y in xrange(iheight)]
		for i, row in enumerate(image):
			for j, v in enumerate(row):
				acc = 0
				for ii, c_row in enumerate(coeff):
					if ii >= 0 and ii < iheight:
						for jj, c in enumerate(c_row):
							if jj >= 0 and jj < iwidth:
								acc += c*image[ii][jj]
				res[j][i] = acc
		return res

	coeff = [[-1, 0, 1]]*3
	image = [[random.randint(0, 255) for x in xrange(32)] for y in xrange(32)]
	res_test = run_test(image, coeff)
	res_ref  = run_test(image, coeff)

	assert compare_images(res_test, res_ref) < 0.00001

# Tests
def test_map_image_to_pixel():
	def pixel_op(code, pos, in_ptr, out_ptr, args, block_size):
		''' Simple image shift implementation. '''
		offset = args['offset']
		x, y = pos
		width, height = block_size
		c_in_ptr = in_ptr + width*y + (x + offset)
		c_out_ptr = out_ptr + width*y + x
		with scoped_alloc(code, 1) as v:
			for instr in load_mem_value(code, c_in_ptr, pos, v, block_size):
				yield instr
			yield MemWImm(c_out_ptr, v)

	def run_test(image, offset):
		code = Code()
		block_size = (16, 16)
		in_ptr = 0
		out_ptr = block_size[0]*block_size[1]
		args = {'offset' : offset}
		def codegen(code, block_size, args):
			return map_image_to_pixel(code, in_ptr, out_ptr, pixel_op, args, block_size)
		code.set_generator(codegen, block_size, args)
		sim = Interpreter(code, image, block_size)
		sim.run()
		return sim.gen_output_image(1)
	def run_ref(image, offset):
		iwidth, iheight = len(image[0]), len(image)
		res = [[0 for x in xrange(iwidth)] for y in xrange(iheight)]
		def in_image(j, i):
			return  i >= 0 and i < iheight and j >= 0 and j < iwidth
		for i, row in enumerate(image):
			for j, _ in enumerate(row):
				x, y = j + offset, i
				res = image[y][x] if in_image(x, y) else 0
		return res

	offset = 2
	image = [[random.randint(0, 255) for x in xrange(32)] for y in xrange(32)]
	res_test = run_test(image, offset)
	res_ref  = run_test(image, offset)

	assert compare_images(res_test, res_ref) < 0.00001

@skip_test
def test_map_image_to_object():
	assert False

def all_test(options = {}):
	tests = [\
		test_map_pixel_to_pixel,\
		test_map_neighborhood_to_pixel,\
		test_map_image_to_pixel,\
		test_map_image_to_object\
	]
	return run_tests(tests, options)

if __name__ == '__main__':
	opt_parser = get_test_options()
	test_options = parse_test_options(opt_parser)
	if not all_test(test_options): exit(1)

