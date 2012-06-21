from BlipCompiler import Compiler, VariableContextManager, CompilerDriver
import interpreter
import imageio
import os
import codegen
import traceback
import pdb
import sys
import blip


class RegProxy(object):
	def __init__(self, name):
		self.name = name
		self.reg_id = blip.regname_to_id(name)
	def __str__(self):
		return self.name

class InstrAdapter(object):
	def __init__(self, instr):
		self.instr = instr
	def opcode(self):
		return self.instr[0]
	def cond(self):
		return self.instr[1]
	def __getattr__(self, name):
		try:
			if name in ['value', 'dest']:
				if not self.opcode() in ['imm']:
					return RegProxy(self.instr[2])
				else:
					return self.instr[2]
			elif name in ['src', 'src1']:
				if not self.opcode() in ['cmp']:
					return RegProxy(self.instr[3])
				else:
					return RegProxy(self.instr[2])
			elif name in ['src2']:
				if not self.opcode() in ['cmp']:
					return RegProxy(self.instr[4])
				else:
					return RegProxy(self.instr[3])
		except Exception, e:
			print 'error:', str(e)
			print 'can\'t fake attribute [%s]'%name
			raise AttributeError()
	def cond_str(self):
        	return '{%s} '%self.cond() if self.cond() else ''
	def __str__(self):
		return '%s %s %s'%(self.instr[0], self.cond_str(), ' '.join(str(x) for x in self.instr[2:]))



if __name__ == '__main__':
	# settings
	block_size = (16, 16)
	in_ptr = 0
	out_ptr = block_size[0]*block_size[1]
	nr_registers = 20

	# Threshold code
	th_src = '''
@kernel
def main(in_ptr, out_ptr, width, th):
	y = get_local_id(0)
	x = get_local_id(1)
	index = y*width + x
	in_v = in_ptr[index]
	out_v = 255 if in_v > th else 0
	out_ptr[index] = out_v
'''
	th_args = [in_ptr, out_ptr, block_size[0], 100]

	# Average code
	avg_src = '''
@kernel
def main(in_ptr, out_ptr, width):
	y = get_local_id(0)
	x = get_local_id(1)
	res = 0
	x_start = x-1 if x > 0 else x
	y_start = y-1 if y > 0 else y
	rind  = y_start*width
	for i in xrange(3):
		ind = rind + x_start
		for j in xrange(3):
			res += in_ptr[ind]
			ind += 1 if x+j < width else 0
		rind += width if y+i < width else 0
	res_ind = y*width + x
	out_ptr[res_ind] = res
'''
	avg_args = [in_ptr, out_ptr, block_size[0]]

	# Convolution code
	edge_src = '''
@kernel
def main(in_ptr, out_ptr, width):
	coeff = [-1, 0, 1]
	y = get_local_id(0)
	x = get_local_id(1)
	res = 0
	x_start = x-1 if x > 0 else x
	y_start = y-1 if y > 0 else y
	rind  = y_start*width
	for i in xrange(3):
		ind = rind + x_start
		res += in_ptr[ind] * coeff[0] if x < width else 0
		res += in_ptr[ind+1] * coeff[1] if x+1 < width else 0
		res += in_ptr[ind+2] * coeff[2] if x+2 < width else 0
		rind += width if y+i < width else 0
	res_ind = y*width + x
	out_ptr[res_ind] = res
'''
	edge_args = [in_ptr, out_ptr, block_size[0]]


	# compile code
	driver = CompilerDriver(nr_registers)
	main_object = driver.run(edge_src, True)


	# patch code before run
	print '#'*100
	patched_object = Compiler.patch_arguments_before_run(main_object, edge_args)
	Compiler.print_object_code(patched_object)

	def codegen_impl(code, block_size):
		for c in patched_object.code:
			yield InstrAdapter(c)

	# try codegen wrapper
	print '@'*100
	print 'codegen'
	print '\n'.join(str(x) for x in codegen_impl(codegen.Code(), block_size))

	codegen_args = {}
	try:
		wrapped_impl = codegen.wrap_codegen(codegen_impl, block_size, codegen_args)
	except InvalidCodegenArgumentsException:
		argstr = ', '.join(codegen.get_codegen_parameters(codegen_impl, True))
		print 'error: invalid arguments for codegen, needed arguments are [%s]'%argstr
		exit(1)

	if not wrapped_impl:
		print 'failed to wrap codegen'

	print '@'*100
	print 'wrapped codegen'
	print '\n'.join(str(x) for x in wrapped_impl(codegen.Code()))

	code = codegen.Code()
	code.set_generator(wrapped_impl)

	# load image
	im_name = 'data' + os.path.sep + 'lena_crop.png'
	image = imageio.read(im_name)

	# actually run code
	print '@'*100
	print 'run interpreter'
	interpreter = interpreter.Interpreter(code, image, block_size, 4, nr_registers)

	try:
		interpreter.run_kernel()
	except Exception, e:
		print str(e)
                traceback.print_tb(sys.exc_traceback)
                pdb.post_mortem(sys.exc_traceback)

	# write output
	output_filename = 'test.png'		
	out_image = interpreter.gen_output_image()
	imageio.write(output_filename, out_image, 1)

