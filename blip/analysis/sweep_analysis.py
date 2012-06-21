import blip.simulator.opcodes
from blip.analysis import analysis
from blip.code.codegen import Code

from violajones import gen_code
from violajones import parse_haar
from violajones.parse_haar import HaarClassifier, HaarStage, HaarFeature

import pickle

def run_codegen(cascade, block_size, codegen_implementation):
	''' Codegen to run violajones for a set of parameters

	This function runs the VJ codegen to analyse the code
	for a certain set of parameters
	'''

	pe_dim = (2, 1)
	args = {'haar_classifier':cascade, 'pe_dim':pe_dim}

	code = Code()
	code.set_generator(codegen_implementation, block_size, args)

	# init analysers
	opcodeFreq = analysis.OpcodeFreq()
	communication = analysis.Communication()
	analysers = [opcodeFreq, communication]

	analysis.analyse_code(code, analysers)
	return analysers

def scale_cascade(cascade, scale):
	''' Scale a Haar Cascade

	This function scales the position and size of each haar shape
	'''

	new_stages = []
	for stage in cascade.stages:
		new_features = []
		for f in stage.features:
			new_shapes = []
			for shapecoeff in f.shapes:
				shape, coeff= shapecoeff
				new_shape = tuple(x*scale for x in shape)
				new_shapes.append((new_shape,coeff))
			new_feature = HaarFeature(new_shapes, f.tilted, f.threshold, f.left_val, f.right_val)	
			new_features.append(new_feature)
		new_stages.append(HaarStage(new_features, stage.stage_threshold, stage.parent))
	return HaarClassifier(tuple(x*scale for x in cascade.size), new_stages)

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
	block_sizes = []
	for k, v in results.iteritems():
		for bsize in v.keys():
			if not bsize in block_sizes:
				block_sizes.append(bsize)
	block_sizes = sorted(block_sizes)
	print scales, block_sizes
	
	# lookup of scales and blocksizes to fill surface value array
	lookup_scales = {}	
	for i, v in enumerate(scales): lookup_scales[v] = i
	lookup_bsizes = {}	
	for i, v in enumerate(block_sizes): lookup_bsizes[v] = i


	# result extractors
	def nr_instructions(res):
		analysis = res['Opcode Analysis']
		return analysis.total_nr_instructions()
	def relative_overhead(res):
		analysis = res['Communication Analysis']
		rel_overhead = float(analysis.overhead)/analysis.nr_instr
		return rel_overhead

	# now plot all analysis results
	plots = [('nr instructions', nr_instructions),\
		 ('relative overhead', relative_overhead)]

	X, Y = np.meshgrid(scales, block_sizes)
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
			x = [bsize for bsize in sorted(v.keys())]
			y = [float(f(v[bsize])) if bsize in v else 0. for bsize in x]
			print 'scale', scale
			print x
			print y
			lines.append(x)
			lines.append(y)
			lines.append(linestyle[linesel])
			linesel +=1
			if linesel >= len(linestyle): linesel = 0
		ax.plot(*lines)
		leg = ax.legend(tuple('scale = %s'%str(scale) for scale, v in results.iteritems()))
		ax.set_xlabel('block size')
		ax.set_ylabel(name)
		ax.set_xticklabels([])
		ax.set_yticklabels([])

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
	parser.add_option('-c', '--cascade', dest='cascade', default='data/haarcascade_frontalface_alt.xml', help='haar cascade')
	parser.add_option('--show_results', action='store_true', dest='show_results', default=False,\
			  help='show results of sweep, requires numpy and matplotlib')
	parser.add_option('--save_results', action='store_true', dest='save_results', default=False,\
			  help='save results of sweep dump as pdfs, requires numpy and matplotlib')
	parser.add_option('--codegen_implementation', dest='codegen_impl', default='gen_detect_faces',\
			  help='override the codegen implementation')
	(options, args) = parser.parse_args()

	result_file = options.output
	cascade_filename = options.cascade
	block_size = (int(options.block_size), int(options.block_size))

	if options.show_results or options.save_results:
		plot_res(result_file, options.save_results, not options.show_results)
		exit(0)

	cascade = parse_haar.parse_haar_xml(cascade_filename)
	# support for alternative codegen implementations
	codegen_implementation = gen_code.gen_detect_faces
	if options.codegen_impl:
		impl_name = options.codegen_impl
		try:
			# try to fetch name from
			codegen_implementation = getattr(gen_code, impl_name)
			print 'using implementation %s'%impl_name
		except Exception, e:
			print 'could not load custom implementation %s'%impl_name
			print 'error', str(e)

			print 'available implementations:'
			print '\n'.join(x for x in dir(gen_code) if 'gen_detect_faces' in x)
			exit(1)

	results = {}
	for cascade_scale in [1, 2, 3, 4]:
		scaled_cascade = scale_cascade(cascade, cascade_scale)
		results2 = {}
		for b in [22, 32, 48, 64, 96, 128]:
			print '#'*40
			print 'scale =', cascade_scale, 'blocksize=', b
			if scaled_cascade.size[0] > b: 
				print 'skip'
				continue
			block_size = (b, b)
			analysers = run_codegen(scaled_cascade, block_size, codegen_implementation)
			analysers_dict = {}
			for a in analysers:
				analysers_dict[str(a)] = a
			results2[b] = analysers_dict
		results[cascade_scale] = results2
	f = open(result_file, 'w')
	pickle.dump(results, f)
	f.close()


