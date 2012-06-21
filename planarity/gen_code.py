''' Code generation for flatness filters. '''
from blip.simulator.opcodes import *
from blip.code.codegen import Code, scoped_alloc
from blip.code.skeletons import load_mem_value
from blip.code.trace_optimiser import Optimiser, ImmediatePass, MemoryPass, PeepholePass

from planarity.filters import Filterbank

def default_argument_setup(filterbank_filename='data/asym_16_3_opencv.xml'):
	filterbank = Filterbank.load(filterbank_filename)
	return {'filterbank':filterbank}

def gen_calc_planarity_inlined(code, block_size, args):
	''' Optimised version by manually inlining all code. '''
	filterbank = args['filterbank']
	f = filterbank.filters[0]
	rows, cols = block_size
	frows, fcols = f.size()
	hfrow, hfcol = [x//2 for x in f.size()]

	out_ptr = args['out_ptr'] if 'out_ptr' in args else rows*cols
	in_ptr = args['in_ptr'] if 'in_ptr' in args else 0
	buffer_ptr = rows*cols*2
	assert buffer_ptr != in_ptr
	assert buffer_ptr != out_ptr

	for filter_nr, f in enumerate(filterbank.filters):
		# convolution + abs
		for i in xrange(rows):
			for j in xrange(cols):
				with scoped_alloc(code, 1) as acc:
					# convolution
					yield Xor(acc, acc, acc)
					for x, y, coeff in f.coefficients:
						ii = i + y - hfrow
						jj = j + x - hfcol
						with scoped_alloc(code, 2) as (coeff_reg, v):
							yield Imm(coeff_reg, coeff)
							for instr in load_mem_value(code, in_ptr, (jj, ii), v, block_size):
								yield instr
							yield Mul(v, v, coeff_reg)
							yield Add(acc, acc, v)

					# take max
					with scoped_alloc(code, 1) as const0:
						yield Imm(const0, 0)
						yield Cmp(acc, const0)
					yield Neg(acc, acc, cond='LT')
					yield MemWImm(buffer_ptr+i*cols + j, acc)
		# gather
		for i in xrange(rows):
			for j in xrange(cols):
				with scoped_alloc(code, 1) as max_v:
					# local max
					yield Imm(max_v, -float('inf'))
					for ii in xrange(frows):
						for jj in xrange(fcols):
							if not f.mask[ii][jj]: continue # skip if not enabled
							iii = i + ii - hfrow
							jjj = j + jj - hfcol
							with scoped_alloc(code, 1) as v:
								for instr in load_mem_value(code, buffer_ptr, (jjj, iii), v, block_size):
									yield instr
								yield Cmp(v, max_v)
								yield Mov(max_v, v, cond='GT')
					# global max
					if filter_nr != 0:
						with scoped_alloc(code, 1) as old_v:
							yield MemRImm(old_v, out_ptr+i*cols+j)
							yield Cmp(old_v, max_v)
							yield Mov(max_v, old_v, cond='GT')
					yield MemWImm(out_ptr+i*cols + j, max_v)

def gen_apply_sparse_filter(code, block_size, args):
	''' Apply sparse filter code generation. '''
	f = args['filter']
	rows, cols = block_size
	hfrow, hfcol = [x//2 for x in f.size()]

	out_ptr = args['out_ptr'] if 'out_ptr' in args else rows*cols
	in_ptr = args['in_ptr'] if 'in_ptr' in args else 0

	for i in xrange(rows):
		for j in xrange(cols):
			with scoped_alloc(code, 1) as acc:
				yield Xor(acc, acc, acc)
				for x, y, coeff in f.coefficients:
					ii = i + y - hfrow
					jj = j + x - hfcol
					with scoped_alloc(code, 2) as (coeff_reg, v):
						yield Imm(coeff_reg, coeff)
						for instr in load_mem_value(code, in_ptr, (jj, ii), v, block_size):
							yield instr
						yield Mul(v, v, coeff_reg)
						yield Add(acc, acc, v)
				yield MemWImm(out_ptr+i*cols + j, acc)

def gen_gather_local_max(code, block_size, args):
	''' Gather local maximum from mask code generation. '''
	f = args['filter']
	rows, cols = block_size
	frows, fcols = f.size()
	hfrow, hfcol = [x//2 for x in f.size()]

	out_ptr = args['out_ptr'] if 'out_ptr' in args else rows*cols
	in_ptr = args['in_ptr'] if 'in_ptr' in args else 0

	for i in xrange(rows):
		for j in xrange(cols):
			with scoped_alloc(code, 1) as max_v:
				yield Imm(max_v, -float('inf'))
				for ii in xrange(frows):
					for jj in xrange(fcols):
						if not f.mask[ii][jj]: continue # skip if not enabled
						iii = i + ii - hfrow
						jjj = j + jj - hfcol
						with scoped_alloc(code, 1) as v:
							for instr in load_mem_value(code, in_ptr, (jjj, iii), v, block_size):
								yield instr
							yield Cmp(v, max_v)
							yield Mov(max_v, v, cond='GT')
				yield MemWImm(out_ptr+i*cols + j, max_v)

def gen_global_max(code, block_size, args):
	''' Calculate element-wise max over two buffers. '''
	rows, cols = block_size

	in_ptr_1 = args['in_ptr_1']
	in_ptr_2 = args['in_ptr_2']
	out_ptr = args['out_ptr'] if 'out_ptr' in args else rows*cols

	for i in xrange(rows):
		for j in xrange(cols):
			addr = i*cols + j
			with scoped_alloc(code, 2) as (v2_res, v1):
				yield MemRImm(v1, in_ptr_1 + addr)
				yield MemRImm(v2_res, in_ptr_2 + addr)
				yield Cmp(v1, v2_res)
				yield Mov(v2_res, v1, cond='GT')
				yield MemWImm(out_ptr + addr, v2_res)
	

def gen_abs_value(code, block_size, args):
	''' Generate element-wise absolute value of a buffer. '''
	rows, cols = block_size
	out_ptr = args['out_ptr'] if 'out_ptr' in args else rows*cols
	in_ptr = args['in_ptr'] if 'in_ptr' in args else 0

	with scoped_alloc(code, 1) as const0:
		yield Imm(const0, 0)
		for i in xrange(rows):
			for j in xrange(cols):
				addr = i*cols + j
				with scoped_alloc(code, 1) as tmp:
					yield MemRImm(tmp, in_ptr + addr)
					yield Cmp(tmp, const0)
					yield Neg(tmp, tmp, cond='LT')
					yield MemWImm(out_ptr + addr, tmp)

def gen_calc_planarity(code, block_size, args):
	''' Main filter processing implementation. '''
	filterbank = args['filterbank']
	pe_dim = args['pe_dim']

	page_size = block_size[0]*block_size[1]

	result_ptr = 1*page_size
	gather_local_ptr = 2*page_size
	src_ptr = 0
	abs_response_ptr = 3*page_size
	for i, f in enumerate(filterbank.filters):
		for instr in gen_apply_sparse_filter(code, block_size, {'filter':f, 'out_ptr':abs_response_ptr}):
			yield instr
		for instr in gen_abs_value(code, block_size, {'out_ptr': abs_response_ptr, 'in_ptr':abs_response_ptr}):
			yield instr
		for instr in gen_gather_local_max(code, block_size, {'in_ptr':abs_response_ptr, 'out_ptr':(result_ptr if i==0 else gather_local_ptr), 'filter':f}):
			yield instr
		if i != 0:
			for instr in gen_global_max(code, block_size, {'in_ptr_1':result_ptr, 'in_ptr_2':gather_local_ptr}):
				yield instr

	yield Nop()

def run_implementation(block_size, implementation, image_filename, filterbank_filename, res_filename_prefix):
	''' Execution wrapper '''
	from blip.simulator import interpreter
	from blip.support import imageio

	# first load the cascade
	filterbank = Filterbank.load(filterbank_filename)

	image = imageio.read(image_filename)
	if not image: raise Exception('image %s not found or not supported'%image_filename)

	im_size = len(image[0]), len(image)
	pe_dim = [s//b for s,b in zip(im_size, block_size)]

	args = {'filterbank':filterbank, 'pe_dim':pe_dim}

	# now execute the codegen
	code = Code()
	code.set_generator(implementation, block_size, args)

	sim = interpreter.Interpreter(code, image, block_size, 4)
	sim.run()

	result = sim.gen_output_image(1) # result is saved in first buffer

	imageio.write(res_filename_prefix + '.png', result, 1)



def optimiser_wrapper(codegen):
	optimiser = Optimiser(400) # the loop hint doesn't improve the optimiser
	optimiser.register_pass(ImmediatePass(optimiser))
	optimiser.register_pass(MemoryPass(optimiser))
	# use this pass to avoid eliminate Nop annotations
	optimiser.register_pass(PeepholePass(optimiser))
	def optim_wrapper(code, block_size, args):
		for x in optimiser.run(code, codegen, block_size, args):
			yield x
	return optim_wrapper

gen_calc_planarity_inlined_opt = optimiser_wrapper(gen_calc_planarity_inlined)
gen_calc_planarity_opt = optimiser_wrapper(gen_calc_planarity)

if __name__ == '__main__':
	from blip.code.codegen import load_codegen
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
	parser.add_option('-s', '--src_image', dest='src_image', default='data/vakgroep128_64.png', help='source image')
	parser.add_option('-o', '--output', dest='output', default='blip_detections', help='output prefix')
	parser.add_option('-c', '--filterbank', dest='filterbank', default='data/asym_16_3_opencv.xml', help='haar filterbank')
	parser.add_option('--codegen_implementation', dest='codegen_impl', default='gen_code.gen_calc_planarity',\
			  help='override the codegen implementation')
	(options, args) = parser.parse_args()

	block_size = (int(options.block_size), int(options.block_size))
	filterbank_filename = options.filterbank
	image_filename = options.src_image
	res_filename = options.output

	implementation = None
	if options.codegen_impl:
		module_name, _, impl_name = options.codegen_impl.rpartition('.')
		implementation = load_codegen(module_name, impl_name)
	if not implementation:
		print 'could not load codegen %s'%options.codegen_impl
		exit(1)
	print 'using codegen: %s'%implementation.__name__

	try:
		run_implementation(block_size, implementation, image_filename, filterbank_filename, res_filename)
	except Exception, e:
		import pdb
		import sys
		print str(e)
		pdb.post_mortem(sys.exc_traceback)
		raise e

