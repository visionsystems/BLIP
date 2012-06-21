from __future__ import with_statement
from blip.code.codegen import scoped_alloc, Code
from blip.simulator.opcodes import *

def load_mem_value(code, ptr, pos, reg, block_size):
	''' Load memory value, handles copying from other blocks. '''
	x, y = pos
	width, height = block_size
	copy_from_left = False
	copy_from_above = False
	if not (x >= 0):
		copy_from_left = True
		x += width
	if not (y >= 0):
		copy_from_above = True
		y += height

	copy_from_right = False
	copy_from_below = False
	if not (x < width):
		copy_from_right = True
		x -= width
	if not (y < height):
		copy_from_below = True
		y -= height

	# copy straight to output register if not transfer
	# between blocks is needed
	if copy_from_below or copy_from_right or copy_from_above or copy_from_left:
		yield MemRImm(code.out, ptr + y*width + x)
	else:
		yield MemRImm(reg, ptr + y*width + x)

	if copy_from_right and copy_from_below:
		x = Mov(code.out, code.east)
		y = Mov(reg, code.south)
		code.tag_com_overhead_instr(x)
		code.tag_com_overhead_instr(y)
		yield x
		yield y
	elif copy_from_right and copy_from_above:
		x = Mov(code.out, code.east)
		y = Mov(reg, code.north)
		code.tag_com_overhead_instr(x)
		code.tag_com_overhead_instr(y)
		yield x
		yield y
	elif copy_from_left and copy_from_below:
		x = Mov(code.out, code.west)
		y = Mov(reg, code.south)
		code.tag_com_overhead_instr(x)
		code.tag_com_overhead_instr(y)
		yield x
		yield y
	elif copy_from_left and copy_from_above:
		x = Mov(code.out, code.west)
		y = Mov(reg, code.north)
		code.tag_com_overhead_instr(x)
		code.tag_com_overhead_instr(y)
		yield x
		yield y
	elif copy_from_right:
		x = Mov(reg, code.east)
		code.tag_com_overhead_instr(x)
		yield x
	elif copy_from_below:
		x = Mov(reg, code.south)
		code.tag_com_overhead_instr(x)
		yield x
	elif copy_from_left:
		x = Mov(reg, code.west)
		code.tag_com_overhead_instr(x)
		yield x
	elif copy_from_above:
		x = Mov(reg, code.north)
		code.tag_com_overhead_instr(x)
		yield x

def map_pixel_to_pixel(code, in_ptr, out_ptr, pixel_op, args, block_size):
	''' Apply one to one pixel operations. '''
	bwidth, bheight = block_size
	for i in xrange(bheight):
		for j in xrange(bwidth):
			off = bwidth*i + j
			with scoped_alloc(code, 2) as (in_reg, out_reg):
				yield MemRImm(in_reg, in_ptr + off)
				for x in pixel_op(code, in_reg, out_reg, args, block_size):
					yield x
				yield MemWImm(out_ptr + off, out_reg)

def map_neighborhood_to_pixel(code, in_ptr, out_ptr, neighborhood, pixel_op, args, block_size):
	''' Apply neigborhood to pixel operations. '''

	bwidth, bheight = block_size
	nheight, nwidth = len(neighborhood[0]), len(neighborhood)
	assert(nheight%2 != 0 and nwidth%2 != 0) # mask size must be odd
	h_nheight = nheight//2
	h_nwidth = nwidth//2
	
	def process_pixel(code, in_ptr, pos, acc, neigborhood, pixel_op, args, block_size):
		j, i = pos
		for ii, row in enumerate(neighborhood):
			for jj, m in enumerate(row):
				if m: # works implicitly for booleans and coefficients
					apos = (j + jj - h_nwidth, i + ii - h_nheight) 
					with scoped_alloc(code, 1) as v:
						for x in load_mem_value(code, in_ptr, apos, v, block_size): 
							yield x
						for x in pixel_op(code, m, v, acc, args, block_size):
							yield x
	for i in xrange(bheight):
		for j in xrange(bwidth):
			pos = (j, i)
			with scoped_alloc(code, 1) as acc:
				# XXX apply assignment-instead-of-accum-on-first-iteration optimalisation
				yield Xor(acc, acc, acc)
				for x in process_pixel(code, in_ptr, pos, acc, neighborhood, pixel_op, args, block_size):
					yield x
				yield MemWImm(out_ptr + bwidth*i+j, acc)
					
def map_image_to_pixel(code, in_ptr, out_ptr, operator, args, block_size):
	''' Apply global image to single pixel operations. '''
	bwidth, bheight = block_size
	for i in xrange(bheight):
		for j in xrange(bwidth):
			pos = (j, i)
			for x in operator(code, pos, in_ptr, out_ptr, args, block_size):
				yield x

def map_image_to_object(code, operator, args, block_size):
	''' Apply global image to object/vector operations. '''
	pass


def demo(im_filename, out_filename, no_optimalisations):
	from blip.code.trace_optimiser import Optimiser, ImmediatePass, PeepholePass, MemoryPass
	from blip.simulator import interpreter
	from blip.support import imageio
	from blip.simulator.opcodes import Imm, Mul, Add

	# settings
	block_size = (32, 32)
	out_ptr = block_size[0]*block_size[1]
	coeff = [[1, -2, 1]]*3

	# convolution implementation with map_neighborhood_to_pixel skeleton
	def convolution_op(code, coeff_v, val, acc, args, block_size):
		''' Simple convolution implementation. '''
		with scoped_alloc(code, 2) as (v, coeff_r):
			yield Imm(coeff_r, coeff_v)
			yield Mul(v, coeff_r, val)
			yield Add(acc, acc, v)

	def codegen(code, block_size, args):
		''' Map convolution to image. '''
		return map_neighborhood_to_pixel(code, 0, out_ptr, coeff, convolution_op, args, block_size)



	# Wrap optimisers
	optimiser = Optimiser(50)
	optimiser.register_pass(ImmediatePass(optimiser))
	#optimiser.register_pass(PeepholePass(optimiser))
	optimiser.register_pass(MemoryPass(optimiser))
	def optim_wrapper(code, block_size, args):
		if no_optimalisations:
			print 'optimalisations disabled'
			return codegen(code, block_size, args)
		else:
			return optimiser.run(code, codegen, block_size, args)

	# Render instruction trace
	f = open(out_filename + '_trace.txt', 'w')
	def tag_str(instr): return ', '.join(instr.tag) if hasattr(instr, 'tag') else ''
	f.write('\n'.join(str(x).ljust(40) + ' tags: ' + tag_str(x) for x in optim_wrapper(Code(), block_size, {})))
	f.close()

	# Run simulation
	code = Code()

	code.set_generator(optim_wrapper, block_size, {})
	image = imageio.read(im_filename)
	sim = interpreter.Interpreter(code, image, block_size)
	sim.run()
	out = sim.gen_output_image(1)
	imageio.write(out_filename, out, 1)

if __name__ == '__main__':
	from optparse import OptionParser
	import sys
	parser = OptionParser()
	parser.add_option('--no_opt', action='store_true', dest='no_opt', default=False, help='disable code optimisation')
	(options, args) = parser.parse_args()

	if len(args) != 2:
		print 'usage: %s input_image output_image'%sys.argv[0]
		exit(1)
	im_filename, out_filename = args

	demo(im_filename, out_filename, options.no_opt)

