################################################
#         Analysis of generated code
################################################
# idea: put all instruction tagging code in a special comment
# for s in shapes:
#     dosomestuff
#     for x in gen_some_stuff()
#         #instr_tag(x, 'blah') [analysis]
# and preprocess the source file to remove the # and the [analysis] keyword
# this way analysis code is cleary seperated from other code
# and doesn't slow down codegen

import pickle

from blip.simulator.interpreter import Interpreter
from blip.simulator.opcodes import *

from blip.code.codegen import Code

from blip.support import visualisation
from blip.support import imageio

# analysers
class Analyser(object): pass

class OpcodeFreq(Analyser):
	''' Analysis of opcode frequencies '''

	def __init__(self, interpreter = None):
		self.freq = {}
	def process(self, instr):
		opcode = instr.opcode()
		if not opcode in self.freq:
			self.freq[opcode] = 0
		self.freq[opcode]+=1
	def __str__(self):
		return 'Opcode Analysis'
	def report(self):
		res = ''
		rev_freq = {}
		for op, nr in self.freq.iteritems():
			if not nr in rev_freq: rev_freq[nr] = []
			rev_freq[nr].append(op)

		res += 'opcode frequency:\n'
		for nr in sorted(rev_freq.keys(), reverse=True):
			res += '\n'.join('%s:\t\t%i'%(op, nr) for op in rev_freq[nr]) + '\n'
		res += 'total instructions: %i'%self.total_nr_instructions()
		return res
	def total_nr_instructions(self):
		return sum(v for k, v in self.freq.iteritems())

class Communication(Analyser):
	''' Analysis of inter PE communication '''

	def __init__(self, interpreter = None):
		self.overhead = 0
		self.nr_instr = 0
	def process(self, instr):
		if hasattr(instr, 'tag') and 'communication overhead' in instr.tag:
			self.overhead += 1
		self.nr_instr += 1
	def __str__(self):
		return 'Communication Analysis'
	def report(self):
		rel_overhead = float(self.overhead)/self.nr_instr
		return 'communication overhead: %3.2f%%'%(100.*rel_overhead)

class MemIO(Analyser):
	''' Analyze memory io. 

	Note that this can only be used from a running interpreter
	as the value of MemR(reg, reg_containing_address) is needed

	'''

	def __init__(self, interpreter):
		self.interpreter = interpreter
		self.mem_size = interpreter.procs[0][0].memory.size
		self.block_size = interpreter.block_size
		self.cnt_read = [0 for x in xrange(self.mem_size)]
		self.cnt_write = [0 for x in xrange(self.mem_size)]
	def process(self, instr):
		opcode = instr.opcode()
		pe = self.interpreter.procs[0][0]
        	cond = instr.cond()
        	if cond and not pe.cond[cond]: return 
		if opcode == 'memr':
			self.cnt_read[int(pe.get_reg(instr.src.reg_id))] += 1
		elif opcode == 'memr_imm':
			self.cnt_read[instr.src] += 1
		elif opcode == 'memw':
			self.cnt_write[int(pe.get_reg(instr.dest.reg_id))] += 1
		elif opcode == 'memw_imm':
			self.cnt_write[instr.dest] += 1
	def report(self):
		bw, bh = self.block_size
		block_mem_size = bw*bh
		nr_buffer = self.mem_size// block_mem_size
		
		res = ''
		res += 'memr:\n'
		for k in xrange(nr_buffer):
			res += '\n'.join(' '.join('%8i'%self.cnt_read[k*block_mem_size+i*bw+j] for j in xrange(bw)) for i in xrange(bh))
			res += '\n\n'
		res += 'memw:\n'
		for k in xrange(nr_buffer):
			res += '\n'.join(' '.join('%8i'%self.cnt_write[k*block_mem_size+i*bw+j] for j in xrange(bw)) for i in xrange(bh))
			res += '\n\n'
		return res
	def __str__(self):
		return 'Memory io'

class SimulationInformation(Analyser):
	''' Provide global simulation information '''
	def __init__(self, interpreter):
		self.block_size = interpreter.block_size
		self.nr_pe = len(interpreter.procs), len(interpreter.procs[0])
		pe = interpreter.procs[0][0]
		self.nr_reg = pe.nr_reg
		self.nr_buffer = pe.memory.size //reduce(lambda x,y: x*y, self.block_size)
		self.codegen_name = 'unknown'
		try:
			self.codegen_name = interpreter.code.gen(interpreter.code).__name__
		except:
			pass
	def process(self, instr):
		pass
	def __str__(self):
		return 'Simulation information'
	def report(self):
		res = ''
		res += 'blocksize    = (%i, %i)\n'%self.block_size
		res += 'nr of pe     = (%i, %i)\n'%self.nr_pe
		res += 'nr of reg    = %i\n'%self.nr_reg
		res += 'nr of buffer = %i\n'%self.nr_buffer
		res += 'codegen      = %s\n'%self.codegen_name
		return res

class CodeSectionProfiler(Analyser):
	''' Count number of instruction in a tagged section. '''
	def __init__(self, interpreter = None):
		self.interpreter = interpreter
		self.section_cnt = {}
	def process(self, instr):
		try:
			tags = instr.tag
			for tag in tags:
				if 'section:' in tag:
					_, section = tag.split(':')
					if not section in self.section_cnt:
						self.section_cnt[section] = 0
					self.section_cnt[section] += 1
		except AttributeError:
			pass
	def __str__(self):
		return 'Code section profiler'
	def report(self):
		res = ''
		res = '\n'.join('%s:\t%i'%(k,v) for k,v in self.section_cnt.iteritems())
		return res

def analyse_code(code, analysers):
	if not code.gen: raise Exception('invalid codegen')

	for instr in code.gen(code):
		for a in analysers: a.process(instr)
	
	for a in analysers: 
		print '- '*20
		print str(a)
		print a.report()
	print '- '*20

class AnalysisInterpreter(Interpreter):
	''' Interpreter with an Analysis hook 
	
	Before each interpreter step call, this interpreter first calls
	the process(instr) method of all registered analysers
	'''
	def __init__(self, code, image, block_size, pe_nr_buffer = 4, pe_nr_reg = NR_REG):
		Interpreter.__init__(self, code, image, block_size, pe_nr_buffer, pe_nr_reg)
		self.analysis = []
	def set_analysis(self, analysis):
		self.analysis.append(analysis)
	def step(self, instr):
		for a in self.analysis:
			a.process(instr)
		Interpreter.step(self, instr)



def main(block_size, codegen_args, image_filename, result_filename, codegen_implementation, analyser_classes):

	# image
	image = imageio.read(image_filename)
	if not image:
		print 'could not read image %s'%image_filename
		image = [[0 for i in xrange(128)] for j in xrange(128)]

	im_size = len(image[0]), len(image)
	pe_dim = [s//b for s,b in zip(im_size, block_size)]
	codegen_args['pe_dim'] = pe_dim

	# code
	code = Code()
	code.set_generator(codegen_implementation, block_size, codegen_args)
	
	# setup interpreter
	interpreter = AnalysisInterpreter(code, image, block_size)

	# setup analysers, create instances of analysis classes
	analysers = [x(interpreter) for x in analyser_classes]
	for a in analysers: interpreter.set_analysis(a)
	
	# run interpreter
	interpreter.run()
	
	# generate report for all analysers
	res = ''
	for a in analysers:
		res += '- '*40 + '\n'
		res += str(a) + '\n'
		try:
			res += a.report() + '\n'
		except Exception, e:
			res += 'could not print report\n'	

	# write report to file
	logf = open(res_filename + '.log', 'w')
	logf.write(res)
	logf.close()
	
	# dump for graph generation
	for a in analysers: a.interpreter = None # remove interpreter from pickle
	print 'dumping to file'
	f = open(res_filename + '.pickle', 'w')
	pickle.dump(analysers, f)
	f.close()


if __name__ == '__main__':
	import sys
	from optparse import OptionParser
	from blip.code.codegen import is_valid_codegen, load_codegen, load_function
	parser = OptionParser()
	parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
	parser.add_option('-s', '--src_image', dest='src_image', default='data/vakgroep128_64.png', help='source image')
	parser.add_option('-o', '--output', dest='output', default='all_analysis', help='output prefix')
	parser.add_option('-c', '--argument_setup', dest='argument_setup', default='violajones.gen_code.default_argument_setup', help='argument setup')
	parser.add_option('--codegen_implementation', dest='codegen_impl', default='violajones.gen_code.gen_detect_faces',\
			  help='override the codegen implementation')
	parser.add_option('--disable_all', action='store_false', dest='enable_all', default=True, help='disable all analysis modules')
	parser.add_option('--enable', dest='enable_analysis', default='', help='enable specific modules')
	(options, args) = parser.parse_args()

	block_size = (int(options.block_size), int(options.block_size))
	print block_size
	image_filename = options.src_image 
	res_filename = options.output

	codegen_implementation = None 
	if options.codegen_impl:
		module_name, _, impl_name = options.codegen_impl.rpartition('.')
		codegen_implementation = load_codegen(module_name, impl_name)
	if not codegen_implementation:
		print 'could not load codegen %s'%options.codegen_impl
		exit(1)
	print 'using codegen: %s'%codegen_implementation.__name__


	argument_setup = None 
	if options.argument_setup:
		module_name, _, impl_name = options.argument_setup.rpartition('.')
		argument_setup = load_function(module_name, impl_name)
	if not argument_setup:
		print 'could not load codegen %s'%options.argument_setup
		exit(1)
	print 'using argument setup: %s'%argument_setup.__name__
	codegen_args = argument_setup()
	
	def is_valid_analyser_cls(cls):
		try:
			return issubclass(cls, Analyser) and not cls == Analyser
		except:
			return False
	analyser_selection = []
	analyser_classes = [(name, v) for name, v in globals().items() if is_valid_analyser_cls(v)]
	if options.enable_all:
		analyser_selection = [v for name, v in analyser_classes]
	elif options.enable_analysis:
		selection = [x.strip() for x in options.enable_analysis.split(',') if x.strip()]
		analyser_selection = [v for name, v in analyser_classes if name in selection]
	print analyser_selection
	main(block_size, codegen_args, image_filename, res_filename, codegen_implementation, analyser_selection)

































