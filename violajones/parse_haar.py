import xml.dom.minidom as dom

class HaarClassifier(object):
	__slots__ = ['size', 'stages']
	def __init__(self, size, stages):
		self.size = size
		self.stages = stages
	def __str__(self):
		return 'Haar Classifier size %ix%i stages %i'%(self.size[0], self.size[1], len(self.stages))

class HaarStage(object):
	__slots__ = ['features', 'stage_threshold', 'parent']
	def __init__(self, features, stage_threshold, parent):
		self.features = features
		self.stage_threshold = stage_threshold
		self.parent = parent
	def __str__(self):
		return 'Haar Stage features %i stage_threshold %f parent %i'%(len(self.features), self.stage_threshold, self.parent)

class HaarFeature(object):
	__slots__ = ['shapes', 'tilted', 'threshold', 'left_val', 'right_val']
	def __init__(self, shapes, tilted, threshold, left_val, right_val):
		self.shapes = shapes
		self.tilted = tilted
		self.threshold = threshold
		self.left_val = left_val
		self.right_val = right_val
	def __str__(self):
		return 'Haar feature %s threshold %f left %f right %f'%(str(self.tilted), self.threshold, self.left_val, self.right_val)

# note:
# there are a lot of unnecessary text nodes in the original xml
# maybe it would be better to strip them before doing the actual processing

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

def parse_haar_xml(filename):
	src = dom.parse(filename)
	storage = get_node(src, 'opencv_storage')
	

	haar = [x for x in storage.childNodes if get_type_id(x) == 'opencv-haar-classifier'][0]

	# now walk the haar classifier
	sizeNode = get_node(haar, 'size')
	sizeText = sizeNode.childNodes[0].nodeValue
	size = [int(x) for x in sizeText.split(' ')]

	stagesNode = get_node(haar, 'stages')
	stages = []
	for sn_id, sn in enumerate(get_nodes(stagesNode, '_')):
		features = []
		trees_node = get_node(sn, 'trees')
		for tree_node in get_nodes(trees_node, '_'):
			# features always seem to be within the root node
			root_node = get_node(tree_node, '_')

			# parse feature
			feature_node = get_node(root_node, 'feature')
			rects_node = get_node(feature_node, 'rects')
			rects = []
			for rect_node in get_nodes(rects_node, '_'):
				rect = get_text(rect_node).split()
				x, y, w, h = [int(x) for x in rect[0:4]]
				coeff = float(rect[4])
				rect_res = ((x, y, w, h), coeff)
				rects.append(rect_res)
			tilted =  True if int(get_text(get_node(feature_node, 'tilted'))) else False

			threshold =  float(get_text(get_node(root_node, 'threshold')))
			left_val =  float(get_text(get_node(root_node, 'left_val')))
			right_val =  float(get_text(get_node(root_node, 'right_val')))

			feature = HaarFeature(rects, tilted, threshold, left_val, right_val)
			features.append(feature)

		stage_threshold = float(get_text(get_node(sn, 'stage_threshold')))
		parent = int(get_text(get_node(sn, 'parent')))
		haar_stage = HaarStage(features, stage_threshold, parent)
		stages.append(haar_stage)
	return HaarClassifier(size, stages)


if __name__ == '__main__':
	haar_classifier = parse_haar_xml('data/haarcascade_frontalface_alt.xml')
	print str(haar_classifier)
	for s in haar_classifier.stages:
		print '\t' + str(s)
		for f in s.features:
			print '\t\t' + str(f)

