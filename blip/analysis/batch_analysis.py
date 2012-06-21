import blip.simulator.opcodes
from blip.analysis import analysis
from blip.code.codegen import Code
from blip.blipcode import *

import pickle

def run_codegen(args, block_size, codegen_implementation):
	''' Codegen to run codegen_implementation for different parameters  '''
	pe_dim = (2, 1)
	args = {'mask_size':args['mask_size'], 'pe_dim':pe_dim}

	code = Code()
	code.set_generator(codegen_implementation, block_size, args)

	# init analysers
	opcodeFreq = analysis.OpcodeFreq()
	communication = analysis.Communication()
	analysers = [opcodeFreq, communication]

	analysis.analyse_code(code, analysers)
	return analysers

def plot_res(filename, save_plots = False, disable_show_plots = False):
	''' Visualisation of sweep results

	Note that this function uses numpy so it can't be executed on PyPy
	'''
	import matplotlib.pyplot as plt
	import matplotlib.ticker as ticker
	from matplotlib.backends.backend_pdf import PdfPages
	from mpl_toolkits.mplot3d import Axes3D
	import numpy as np

	f = open(filename)
	results = pickle.load(f)
	f.close()

	# find all scales and blocksizes
	scales = results.keys()
	mask_sizes = []
	for k, v in results.iteritems():
		for mask_size in v.keys():
			if not mask_size in mask_sizes:
				mask_sizes.append(mask_size)
	mask_sizes = sorted(mask_sizes)
	print scales, mask_sizes
	
	# lookup of scales and blocksizes to fill surface value array
	lookup_scales = {}	
	for i, v in enumerate(scales): lookup_scales[v] = i
	lookup_bsizes = {}	
	for i, v in enumerate(mask_sizes): lookup_bsizes[v] = i


	# result extractors
	def nr_instructions(res):
		analysis = res['Opcode Analysis']
		return analysis.total_nr_instructions()
	def relative_overhead(res):
		analysis = res['Communication Analysis']
		rel_overhead = float(analysis.overhead)/analysis.nr_instr
		return rel_overhead

	# now plot all analysis results
	plots = [('relative overhead [%]', relative_overhead)]

# '''('nr instructions', nr_instructions),\ '''

	X, Y = np.meshgrid(scales, mask_sizes)
	for name, f in plots:
		print name
#		Z = [[0 for x in scales] for y in block_sizes]
#		for scale, v in results.iteritems():
#			for bsize, res in v.iteritems():
#				Z[lookup_bsizes[bsize]][lookup_scales[scale]] = float(f(res))
#		Z = np.array(Z)
#		print Z
#
#		fig = plt.figure()
#		fig.set_label(name)
#		fig.canvas.manager.set_window_title(name)
#		ax = Axes3D(fig)
#		surf=ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0)
#		ax.w_xaxis.set_major_locator(ticker.FixedLocator(scales))
#		ax.set_xlabel('scales')
#		ax.w_yaxis.set_major_locator(ticker.FixedLocator(block_sizes))
#		ax.set_ylabel('block_sizes')
		fig = plt.figure()
		fig.set_label(name)
		fig.canvas.manager.set_window_title(name)
		ax = fig.add_subplot(111)
		ax.set_color_cycle(['r', 'g', 'b', 'c'])
		lines = []
		linestyle = ['k--', 'k:', 'k', 'k-']
		linesel = 0
		for scale, v in results.iteritems():
			x = [mask_size for mask_size in sorted(v.keys())]
			y = [float(f(v[mask_size]))*100 if mask_size in v else 0. for mask_size in x]
			print 'scale', scale
			print x
			print y
			lines.append(x)
			lines.append(y)
			lines.append(linestyle[linesel])
			linesel +=1
			if linesel >= len(linestyle): linesel = 0
		ax.plot(*lines)
		leg = ax.legend(tuple('block size = %s'%str(scale) for scale, v in results.iteritems()))
		ax.set_xlabel('mask size')
		ax.set_ylabel(name)
		#ax.set_xticklabels([])
		#ax.set_yticklabels([])

		if save_plots:
			try:
				p = PdfPages(name + '.pdf')
				p.savefig()
				p.close()
			except Exception, e:
				print 'could not save graph to pdf file'
				print 'error:', str(e)

	# finally show all plots
	if not disable_show_plots:
		plt.show()
			
	

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
	parser.add_option('-o', '--output', dest='output', default='sweep_results.pickle', help='output')
	parser.add_option('--show_results', action='store_true', dest='show_results', default=True,\
			  help='show results of sweep, requires numpy and matplotlib')
	parser.add_option('--save_results', action='store_true', dest='save_results', default=True,\
			  help='save results of sweep dump as pdfs, requires numpy and matplotlib')
	parser.add_option('--codegen_implementation', dest='codegen_impl', default='gen_code.gen_detect_faces',\
			  help='override the codegen implementation')
	(options, args) = parser.parse_args()

	result_file = options.output
	block_size = (int(options.block_size), int(options.block_size))

	# load codegen
	from blip.code.codegen import load_codegen, get_codegen_parameters
	from blip.code.codegen import InvalidCodegenArgumentsException

	# support for alternative codegen implementations
	codegen_implementation = options.codegen_impl
	if options.codegen_impl:
		try:
			module_name, _, impl_name = options.codegen_impl.rpartition('.')
			codegen_implementation = load_codegen(module_name, impl_name)
			print 'using implementation %s'%impl_name
		except Exception, e:
			print 'could not load custom implementation %s'%impl_name
			print 'error', str(e)
			exit(1)

	'''results = {}
	args = {}
	for mask_size in xrange(1,5,1):
		results2 = {}
		args['mask_size'] = mask_size
		print 'scale =', mask_size, 'mask_size =', args['mask_size']
		block_size = (64, 64)
		analysers = run_codegen(args, block_size, codegen_implementation)
		analysers_dict = {}
		for a in analysers:
			analysers_dict[str(a)] = a
		results2[mask_size] = analysers_dict
		results[mask_size] = results2'''
	results = {}
	args = {}
	for b_size in [64]:
		results2 = {}
		for mask_size in xrange(1,34,2):
			print '#'*40
			args['mask_size'] = mask_size
			print 'block_size =', b_size, 'mask_size =', args['mask_size'] 
			block_size = (b_size, b_size)
			analysers = run_codegen(args, block_size, codegen_implementation)
			analysers_dict = {}
			for a in analysers:
				analysers_dict[str(a)] = a
			results2[mask_size] = analysers_dict
		results[b_size] = results2
	f = open(result_file, 'w')
	pickle.dump(results, f)
	f.close()

	if options.show_results or options.save_results:
		plot_res(result_file, options.save_results, not options.show_results)
		exit(0)


