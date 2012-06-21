''' Parsing of filters and filter representation. '''

import xml.dom.minidom as dom

class Filter(object):
	''' Represents a sparse filter. '''
	def __init__(self, coefficients, rows, cols):
		self.coefficients = coefficients
		self.rows = rows
		self.cols = cols
		self.mask = [[False for _ in xrange(self.cols)] for _ in xrange(self.rows)]
		self._calc_mask()
			
	def _calc_mask(self):
		# shift shape over mask size (= filter size) and check if all
		# shifted coeff point lie within the mask
		for i in xrange(self.rows):
			for j in xrange(self.cols):
				all_points_in = True
				for x, y, _ in self.coefficients:
					ii = i + y
					jj = j + x
					if not (ii >= 0 and ii < self.rows and jj >= 0 and jj < self.cols):
						all_points_in = False
				self.mask[i][j] = all_points_in	
					
				
	def size(self):
		return (self.rows, self.cols)

class Filterbank(object):
	''' Represents a sparse filterbank. '''
	def __init__(self, filters = []):
		self.filters = filters

	@classmethod
	def load(cls, filename):
		src = dom.parse(filename)
		storage = get_node(src, 'opencv_storage')
	
		filters = []
		filterNodes = [x for x in storage.childNodes if get_type_id(x) == 'opencv-matrix']
		for i, filterNode in enumerate(filterNodes):
			rows = int(get_text(get_node(filterNode, 'rows')))
			cols = int(get_text(get_node(filterNode, 'cols')))
			data = get_text(get_node(filterNode, 'data'))
			data = [float(x) for x in data.split(' ') if x]
			assert rows*cols == len(data)

			# now filter non-null coeffs + position
			coeffs = [(x, y, data[y*cols+x]) for y in xrange(rows) for x in xrange(cols)]
			coeffs = [x for x in coeffs if x[-1] != 0]
			print coeffs
			if len(coeffs) != 4:
				print 'warning: only %i coefficients, filter %i ignored'%(len(coeffs), i)
				continue
			filters.append(Filter(coeffs, rows, cols))
		return Filterbank(filters)		




# find classifier
def get_type_id(node):
	type_id = ''
	try:
		type_id = node.getAttributeNode('type_id').value
	except:
		pass
	return type_id

def get_nodes(parent, name):
	return [x for x in parent.childNodes if x.nodeName == name]

def get_node(parent, name):
	x = [x for x in parent.childNodes if x.nodeName == name]
	if len(x) > 1: raise Exception('More then one child node')	
	return x[0]
def get_text(parent):
	nodes = parent.childNodes
	if len(nodes) > 1: raise Exception('More then one child node')	
	return str(nodes[0].nodeValue)


if __name__ == '__main__':
	import sys

	if len(sys.argv) < 2:
		print 'usage: %s filterbank.xml'%sys.argv[0]
		exit(1)

	filterbank = Filterbank.load(sys.argv[1])
