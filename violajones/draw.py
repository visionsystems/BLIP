from blip.support.svgfig import *

def draw_grid(fig, size, blocksize):
	if not fig: fig = Fig()
	step = size/blocksize
	for i in range(step+1):
		fig = Fig(Line(0, i*blocksize, size, i*blocksize), fig)
		fig = Fig(Line(i*blocksize, 0, i*blocksize, size), fig)
	return fig


def write_svg(fig, size, filename):
	c = canvas(fig.SVG().xml(), height='100%',width='100%', viewBox='0 0 %i %i'%(size+1,size+1), \
	style="stroke-linejoin:round; stroke:black; stroke-width:0.2pt; text-anchor:middle; fill:none")
	c.save(filename) 

def draw_shape(fig, pos, shape):
	px, py = pos
	shape, coeff = shape
	x, y, w, h = shape
	lbx = x + px
	lby = y + py + h
	rtx = x + px + w
	rty = y + py
	cfill = "blue" if coeff >= 0 else "red"
	r = Rect(min(lbx, rtx), min(lby, rty), max(lbx, rtx), max(lby, rty))
	r.attr['fill']=cfill
	r.attr['fill_opacity']='50%'
	r.attr['stroke'] = 'gray'
	fig = Fig(fig, r)
	return fig

def draw_shapes(fig, pos, shapes):
	if len(shapes) != 0:
		fig = draw_shape(fig, pos, shapes[0])
		fig = draw_shapes(fig, pos, shapes[1:])
	return fig

def write_filter(filename, f, block_size, proc_size):
	size = block_size * proc_size

	fig = Fig()
	fig = draw_shapes(fig, f.pos, f.shapes)
	fig = draw_grid(fig, size, block_size)

	write_svg(fig, size, filename) 


if __name__ == '__main__':

	size = 20
	blocksize = 4
	position = (1, 1)
	shapes = [((0, 0, 3, 3), -1), ((0, 3, 3, 2), -1), ((3, 0, 2, 3), -1), ((3, 3, 2, 2), -1)]

	fig = Fig()
	fig = draw_shapes(fig, position, shapes)
	fig = draw_grid(fig, size, blocksize)

	write_svg(fig, size, 'test.svg')

	
