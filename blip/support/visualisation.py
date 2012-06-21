def draw_point_color(image, x, y):
	height, width = len(image), len(image[0])/3
	if x < 0 or x >= width or y < 0 or y >=height: 
		return
	image[y][3*x+0] = 255
	image[y][3*x+1] = 0
	image[y][3*x+2] = 0 

def draw_square_color(image, rect):
	x, y, w, h = rect
	for i in range(w):
		draw_point_color(image, x+i, y    )
		draw_point_color(image, x+i, y+h-1)
	for i in range(h):
		draw_point_color(image, x    , y+i)
		draw_point_color(image, x+w-1, y+i)

def draw_faces(image, detected_faces):
	res = []
	for r in image:
		row = []
		for p in r:
			# create color image
			for i in range(3): row.append(p)
		res.append(row)
	for d in detected_faces:
		draw_square_color(res, d)

	return res 

def show_im(name, image, channels, static=False, scale_im=False):
	try:
		import cv
	except:
		print 'no valid opencv found'
		return

	if (scale_im):
		maxv = 1 if type(image[0][0]) == float else 255
		vmin = min(min(image))
		vmax = max(max(image))
		print 'vmin', vmin, 'vmax', vmax
		scale = float(vmax-vmin) if vmax != vmin else 1.
		print 'scale', scale
		image = [[maxv*float(x-vmin)/scale for x in y] for y in image]
		print 'minmax after ', min(min(image)), max(max(image))

	opencv_im = imageio.to_opencv_mat(image, channels)
	print 'opencv rows cols', opencv_im.rows, opencv_im.cols

	cv.NamedWindow(name, 1)
	cv.ShowImage(name, opencv_im)
	cv.WaitKey(0 if static else 10)
	if static:
		cv.DestroyWindow(name)

def plot_surface3D(x, y, values, plotlabel, xlabel, ylabel, show_plot, save_plot):
	try:
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
		from matplotlib.backends.backend_pdf import PdfPages
		from mpl_toolkits.mplot3d import Axes3D
		import numpy as np
	except Exception, e:
		print 'could not import one of the required modules'
		print 'error: ', str(e)
		return

	X, Y = np.meshgrid(x, y)
	# convert to float to make sure that the pdf backend can handle the plt
	Z = np.array([[float(v) for v in row] for row in values])
	print Z

	fig = plt.figure()
	fig.set_label(plotlabel)
	fig.canvas.manager.set_window_title(plotlabel)
	ax = Axes3D(fig)
	surf=ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0)
	ax.w_xaxis.set_major_locator(ticker.FixedLocator(x))
	ax.set_xlabel(xlabel)
	ax.w_yaxis.set_major_locator(ticker.FixedLocator(y))
	ax.set_ylabel(ylabel)

	if save_plot:
		try:
			p = PdfPages(plotlabel + '.pdf')
			p.savefig()
			p.close()
		except Exception, e:
			print 'could not save graph to pdf file'
			print 'error:', str(e)

	# finally show plot
	if show_plot:
		plt.show()
