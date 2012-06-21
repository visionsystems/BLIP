import interpreter
from blip import *
import imageio
import os
import codegen

default_sequence_dir = os.path.sep.join(['data', 'sequence'])

def png_images_from_directory(dir_name):
	im_names = sorted([x for x in  os.listdir(dir_name) if '.png' in x])
	return (imageio.read(os.path.sep.join([dir_name, x])) for x in im_names)


def test():
	image_dir = default_sequence_dir 
	print image_dir
	for x in png_images_from_directory(image_dir):
		print len(x), len(x[0])

def process_image_sequence(codegen_wrapper, block_size, images):
	code = Code()
	code.set_generator(codegen_wrapper)

	sim = None
	for im in images:
		if not sim:
			sim = interpreter.Interpreter(code, im, block_size)
		else:
			sim.reset() # restart code gen
                        sim.set_src_image(im)
		sim.run()
		yield sim.gen_output_image(1)
	

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-t', '--test', action='store_true', dest='do_test', default=False, help='execute test code')
	parser.add_option('-s', '--sourcedir', dest='source_dir', default=default_sequence_dir, help='sequence dir')
	parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
	parser.add_option('--codegen', dest='codegen_impl', default='blipcode.gen_bbs',\
					  help='override the codegen implementation')
	parser.add_option('-a', '--codegen_args', dest='codegen_args', default="{'alpha':0.9,'th':20}",\
					  help='codegen arguments, evaluated as dict')
	(options, args) = parser.parse_args()

	if options.do_test:
		test()
		exit(0)

	block_size = int(options.block_size)
	block_size = (block_size, block_size)

	# codegen arguments processing
	# XXX hack
	try:
		codegen_args = eval(options.codegen_args)
		if not isinstance(codegen_args, dict):
			raise TypeError()
	except:
		print 'error: unable to parse the codegen argument'
		print '	   please enclose the dict in double quotes'
		print '	   raw args: >>>%s<<<'%options.codegen_args
		exit(1)

	# load codegen
	from codegen import load_codegen, wrap_codegen, get_codegen_parameters
	from codegen import InvalidCodegenArgumentsException
	module_name, codegen_name = options.codegen_impl.split('.')
	codegen_impl = load_codegen(module_name, codegen_name)
	if not codegen_impl:
		print 'error: could not load codegen %s'%codegen_impl
		exit(1)
	try:
		wrapped_impl = wrap_codegen(codegen_impl, block_size, codegen_args)
	except InvalidCodegenArgumentsException:
		argstr = ', '.join(get_codegen_parameters(codegen_impl, True))
		print 'error: invalid arguments for codegen, needed arguments are [%s]'%argstr
		exit(1)
	
	images = png_images_from_directory(options.source_dir)
	for i, res in enumerate(process_image_sequence(wrapped_impl, block_size, images)):
		imageio.write('res_%06i.png'%i, res, 1)
	
