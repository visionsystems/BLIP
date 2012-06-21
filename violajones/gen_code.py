from __future__ import with_statement
from blip.simulator.opcodes import *
from blip.code.trace_optimiser import Optimiser, ImmediatePass, PeepholePass, MemoryPass
from blip.code.codegen import Code, scoped_alloc
from blip.support import visualisation

import violajones.parse_haar
from violajones.parse_haar import HaarClassifier, HaarStage, HaarFeature 
import violajones.draw

class InvalidBlocksizeException(Exception): pass
class IllegalInstructionException(Exception): pass

def default_argument_setup(cascade_filename='data/haarcascade_frontalface_alt.xml'):
	from violajones import parse_haar
	# cascade
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	return {'haar_classifier':cascade}

def split_shape_across_blocks(shape, pos, block_size):
	''' Split rectangles according to a block grid for a certain position.''' 
	# XXX refactor this code as there is a lot of code duplication
	try:
		if (block_size[0] != block_size[1]):
			raise InvalidBlocksizeException('split_filter only excepts square blocksizes')
		block_size = block_size[0]
	except TypeError: pass
	px, py = pos

	new_shapes = []
	fx, fy, fw, fh = shape

	# split if necessary in the X direction
	lbx = fx + px
	rtx = fx + px + fw - 1
	if lbx > rtx: raise Exception('lbx > rtx')
	lbx_b = lbx/block_size
	rtx_b = rtx/block_size
	if lbx_b == rtx_b:
		new_shapes.append(shape)
	else:
		# split block
		x = lbx
		current_block = lbx_b
		while x < rtx:
			sx = x
			ex = min((current_block+1)*block_size - 1, rtx)
			ns = ((fx if x == lbx else sx-px), fy, ex-sx + 1, fh)
			new_shapes.append(ns)
			x = ex + 1
			current_block = x/block_size

	final_shapes = []
	# split if necessary in the Y direction
	for s in new_shapes:
		sfx, sfy, sfw, sfh = s
		lby = sfy + py + sfh - 1
		rty = sfy + py
		if lby < rty: raise Exception('lby < rty')
		lby_b = lby/block_size
		rty_b = rty/block_size
		if lby_b == rty_b:
			final_shapes.append(s)
		else:
			y = rty
			current_block = rty_b
			while y < lby:
				sy = y
				ey = min((current_block+1)*block_size - 1, lby)
				ns = (sfx, (sfy if y == rty else sy - py), sfw, ey-sy + 1)
				final_shapes.append(ns)
				y = ey + 1
				current_block = y/block_size
	return final_shapes

def split_filter(feature, pos, block_size):
	shapes = []
	for shape, coeff in feature.shapes:
		split_shapes = split_shape_across_blocks(shape, pos, block_size)
		for s in split_shapes:
			shapes.append((s, coeff))
	return HaarFeature(shapes, feature.tilted, feature.threshold, feature.left_val, feature.right_val)


# code generation
def gen_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, block_size):
	''' Generate instructions integral image of the image and squared image calculation.'''
	width, height = block_size
	with scoped_alloc(code, 2) as (acc, tmp):
		for i in xrange(height):
			for j in xrange(width):
				ptr = width*i + j
				# r2: acc
				# r3: prev addr
				# r4: tmp val
				if j > 0:
					#int_im[i][j] += (float(image[i][j]) + float(int_im[i][j-1]))
					yield MemRImm(tmp, src_ptr + ptr)           # tmp = image[i][j]
					yield MemRImm(acc, integral_ptr + ptr -1)        # acc = int_im[i][j-1]
					yield Add(acc, acc, tmp) # acc = int_im[i][j-1] + image[i][j]
					yield MemWImm(integral_ptr + ptr, acc)				

					#sq_int_im[i][j] += (float(image[i][j]*float_image[i][j]) + float(sq_int_im[i][j-1]))
					yield Mul(tmp, tmp, tmp) # tmp = image[i][j] * image[i][j]
					yield MemRImm(acc, sq_integral_ptr + ptr -1)           # acc = sq_int_im[i][j-1]
					yield Add(acc, acc, tmp) # acc = (image[i][j]*image[i][j]) + sq_int_im[i][j-1]

					yield MemWImm(sq_integral_ptr + ptr, acc)

				else:
					#int_im[i][j] = float(image[i][j])
					yield MemRImm(acc, src_ptr + ptr)
					yield MemWImm(integral_ptr + ptr, acc)

					#sq_int_im[i][j] = float(image[i][j]*image[i][j])
					yield Mul(acc, acc, acc)
					yield MemWImm(sq_integral_ptr + ptr, acc)

		for j in xrange(width):
			for i in xrange(height):
				if i > 0:
					#int_im[i][j] += float(int_im[i-1][j])
					int_ptr_i_j = integral_ptr + i*width + j
					yield MemRImm(acc, int_ptr_i_j)
					int_ptr_im1_j = integral_ptr + (i-1)*width + j
					yield MemRImm(tmp, int_ptr_im1_j)

					yield Add(acc, acc, tmp)
					yield MemWImm(int_ptr_i_j, acc)

					#sq_int_im[i][j] += float(sq_int_im[i-1][j])
					sq_int_ptr_i_j = sq_integral_ptr + i*width + j
					yield MemRImm(acc, sq_int_ptr_i_j)
					sq_int_ptr_im1_j = sq_integral_ptr + (i-1)*width + j
					yield MemRImm(tmp, sq_int_ptr_im1_j)

					yield Add(acc, acc, tmp)
					yield MemWImm(sq_int_ptr_i_j, acc)


def gen_full_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, pe_array_size, block_size):
	width, height = block_size
	pe_width, pe_height = pe_array_size
	for x in gen_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, block_size):
		yield x
	for buffer_ptr in [integral_ptr, sq_integral_ptr]:
		with scoped_alloc(code, 1) as acc:
			# horizontal propagation
			for row in xrange(height):
				yield MemRImm(code.out, buffer_ptr+(row+1)*width-1)
				for bid in xrange(pe_width-1):
					for x in xrange(width):
						ptr = buffer_ptr + row*width + x
						yield MemRImm(acc, ptr)
						yield Add(acc, acc, code.west)
						yield MemWImm(ptr, acc)
					yield Mov(code.out, code.west)

			# vertical propagation
			for col in xrange(width):
				yield MemRImm(code.out, buffer_ptr + width*(height-1) + col)
				for bid in xrange(pe_height-1):
					for y in xrange(height):
						ptr = buffer_ptr + y*width + col
						yield MemRImm(acc, ptr)
						yield Add(acc, acc, code.north)
						yield MemWImm(ptr, acc)
					yield Mov(code.out, code.north)

def gen_equalize_hist(code, block_size):
	yield Nop() # XXX not implemented yet
			
def gen_integral_sum(code, out_reg, position, shape, ptr, block_size):
	''' Gen integral sum code.
	this code assumes that each shape is in a single block
	maximum one block away from the originating block

	note that in contrast with the python implementation,
	a block has the ranges: x[0,w[, y[0,h[
	so width and height of the shape need to be incremented by one
	to be compatible with the violajones sum function
	'''
	px, py = position
	x, y, w, h = shape
	width, height = block_size
	xx = px + x
	yy = py + y
	# val_sum [r4], tmp [r5]
	# to handle values outside the block range:
	# first detect cases and adapt the xx,yy coordinates
	# calculate the value as usual
	# copy the value to the correct block
	copy_from_right = False
	copy_from_below = False
	if not ((xx+w-1) < width):
		copy_from_right = True
		xx -= width
	if not ((yy+h-1) < height):
		copy_from_below = True
		yy -= height

	# v1 = im[yy    ][xx    ]
	# v2 = im[yy    ][xx+w-1]
	# v3 = im[yy+h-1][xx    ]
	# v4 = im[yy+h-1][xx+w-1]
	# val_sum =  v1 - v2 - v3 + v4

	with scoped_alloc(code, 1) as tmp:
		yield MemRImm(out_reg, ptr + yy * width + xx) # r = v1
		yield MemRImm(tmp, ptr +  yy    * width +   (xx+w-1)) # v2
		yield Sub(out_reg, out_reg, tmp) # r = v1 - v2
		yield MemRImm(tmp, ptr + (yy+h-1) * width +  xx     ) # v3
		yield Sub(out_reg, out_reg, tmp) # r = v1 - v2 - v3
		yield MemRImm(tmp, ptr + (yy+h-1) * width + (xx+w-1)) # v4
		yield Add(out_reg, out_reg, tmp) # r = v1 - v2 - v3 + v4

	# now handle the shapes out of PE block
	if copy_from_right and copy_from_below:
		yield Mov(code.out, out_reg)
		yield Mov(code.out, code.east)
		yield Mov(out_reg, code.south)
	elif copy_from_right:
		yield Mov(code.out, out_reg)
		yield Mov(out_reg, code.east)
	elif copy_from_below:
		yield Mov(code.out, out_reg)
		yield Mov(out_reg, code.south)

def gen_calc_variance(code, out_reg, position, integral_ptr, sq_integral_ptr, haar_size, block_size):
	''' Variance calculation.
	'''
	# calculte split shape
	width, height = block_size
	haar_width, haar_height = haar_size
	
	# in order to produce same results as the python VJ implementation, x,y: [0, w], [0, h]
	shape = (0, 0, haar_width + 1, haar_height + 1) 
	shapes = split_shape_across_blocks(shape, position, block_size)

	with scoped_alloc(code, 2) as (int_acc, sq_acc):
		# int_acc: integral sum accum
		yield Xor(int_acc, int_acc, int_acc)
		# sq_acc: square integral sum accum
		yield Xor(sq_acc, sq_acc, sq_acc)
		for i, s in enumerate(shapes):
			# calculate sum of int_im 
			with scoped_alloc(code, 1) as sum_out:
				for x in gen_integral_sum(code, sum_out, position, s, integral_ptr, block_size):
					if i > 0: code.tag_com_overhead_instr(x)
					yield x
				yield Add(int_acc, int_acc, sum_out)

			# calculate sum of sq_int_im
			with scoped_alloc(code, 1) as sum_out:
				for x in gen_integral_sum(code, sum_out, position, s, sq_integral_ptr, block_size):
					if i > 0: code.tag_com_overhead_instr(x)
					yield x
				yield Add(sq_acc, sq_acc, sum_out)

		# calculate variance
		with scoped_alloc(code, 1) as area_r:
			yield Imm(area_r, haar_width*haar_height) # area
			yield Mul(sq_acc, sq_acc, area_r)     # sq_acc = sq_integral_sum*(haar_width*haar_height) 
		yield Mul(int_acc, int_acc, int_acc)    # int_acc = integral_sum^2
		yield Sub(int_acc, sq_acc, int_acc)    # int_acc = sq_integral_sum*(haar_width*haar_height) - integral_sum^2 
		with scoped_alloc(code, 1) as const_0:
			yield Imm(const_0, 0.)
			yield Cmp(int_acc, const_0)                # comp int_acc - 0
		yield Sqrt(out_reg, int_acc, cond='GT')   # r7 = sqrt(sq_integral_sum*(haar_width*haar_height) - integral_sum^2)
		yield Imm(out_reg, 1., cond='LE')          # if int_acc <= 0: variance = 1

def gen_calc_variance_fullintegral(code, out_reg, position, integral_ptr, sq_integral_ptr, haar_size, block_size):
	''' Variance calculation for full integral image.
	'''
	# calculate split shape
	width, height = block_size
	haar_width, haar_height = haar_size
	
	# in order to produce same results as the python VJ implementation, x,y: [0, w], [0, h]
	shape = (0, 0, haar_width + 1, haar_height + 1) 

	with scoped_alloc(code, 1) as tmp:
		# calculate sum of int_im 
		# out_reg = val_sum = integral_sum(integral, xx, yy, w, h)
		for x in gen_fullintegral_sum(code, out_reg, position, shape, integral_ptr, block_size):
			yield x

		# calculate sum of sq_int_im
		# tmp = val_sum = integral_sum(sq_integral, xx, yy, w, h)
		for x in gen_fullintegral_sum(code, tmp, position, shape, sq_integral_ptr, block_size):
			yield x

		# calculate variance
		with scoped_alloc(code, 1) as area_r:
			yield Imm(area_r, haar_width*haar_height) # area
			yield Mul(tmp, tmp, area_r)     # sq_acc = sq_integral_sum*(haar_width*haar_height) 
		yield Mul(out_reg, out_reg, out_reg)    # out_reg = integral_sum^2
		yield Sub(out_reg, tmp, out_reg)    # out_reg = sq_integral_sum*(haar_width*haar_height) - integral_sum^2 
		with scoped_alloc(code, 1) as const_0:
			yield Imm(const_0, 0.)
			yield Cmp(out_reg, const_0)                # comp out_reg - 0
		yield Sqrt(out_reg, out_reg, cond='GT')   # out_reg = sqrt(sq_integral_sum*(haar_width*haar_height) - integral_sum^2)
		yield Imm(out_reg, 1, cond='LE')     # if out_reg <= 0: variance = 1

def all_points_in_n_blocks(points, block_size, n):
	''' Test if all memory accesses lie in at most n blocks. '''
	bwidth, bheight = block_size
	bp = [(x//bwidth, y//bheight) for x,y in points]
	return len(set(bp)) == n
def all_points_in_same_block(points, block_size):
	''' Test if all memory accesses lie in at most 1 block. '''
	return all_points_in_n_blocks(points, block_size, 1)

def all_points_in_two_neighbouring_blocks(points, block_size):
	''' Test if points are in two neigbouring blocks. '''
	bwidth, bheight = block_size
	bp = [(x//bwidth, y//bheight) for x,y in points]
	bp_set = set(bp)
	if not len(bp_set) == 2: return False
	s_a, s_b = bp_set
	neigbours = bool(abs(s_a[0]-s_b[0]) == 1) ^ bool(abs(s_a[1]- s_b[1]) == 1)
	return neigbours

def load_mem_value(code, ptr, pos, reg, block_size):
	''' Load memory value, handles copying from other blocks. '''
	x, y = pos
	width, height = block_size
	copy_from_right = False
	copy_from_below = False
	if not (x < width):
		copy_from_right = True
		x -= width
	if not (y < height):
		copy_from_below = True
		y -= height

	if copy_from_below or copy_from_right:
		yield MemRImm(code.out, ptr + y*width + x)
	else:
		yield MemRImm(reg, ptr + y*width + x)

	if copy_from_right and copy_from_below:
		x = Mov(code.out, code.east)
		y = Mov(reg, code.south)
		code.tag_com_overhead_instr(x)
		code.tag_com_overhead_instr(y)
		yield x
		yield y
	elif copy_from_right:
		x = Mov(reg, code.east)
		code.tag_com_overhead_instr(x)
		yield x
	elif copy_from_below:
		x = Mov(reg, code.south)
		code.tag_com_overhead_instr(x)
		yield x

def gen_fullintegral_sum2_2(code, dest_reg, ptr, points, block_size):
	width, height = block_size
	bp = [(x//width, y//height) for x, y in points]
	bp_set = sorted(set(bp))
	assert(len(bp_set) == 2)
	bp_a, bp_b = bp_set
	if abs(bp_a[0] - bp_b[0]) == 1:
		assert(bp_a[1] == bp_b[1])
		base_row = (bp_a[1] == 0) # bp_a[1] == bp_b[1] is first row?

		mpoints = [(x, y-height) for x, y in points] if not base_row else points
		p_v4 = (mpoints[3][0]-width, mpoints[3][1])
		p_v3 =  mpoints[2]
		p_v2 = (mpoints[1][0]-width, mpoints[1][1])
		p_v1 =  mpoints[0]
		assert(p_v4[0] < width and p_v4[1] < height)
		assert(p_v2[0] < width and p_v2[1] < height)
		assert(p_v2[0] < width and p_v2[1] < height)
		assert(p_v1[0] < width and p_v1[1] < height)

		with scoped_alloc(code, 1) as tmp:
			yield MemRImm(tmp, ptr + p_v4[1]*width + p_v4[0])
			yield MemRImm(dest_reg , ptr + p_v2[1]*width + p_v2[0])
			yield Sub(code.out, tmp, dest_reg)
			yield MemRImm(tmp, ptr + p_v1[1]*width + p_v1[0])
			yield Add(dest_reg, code.east, tmp)
			yield MemRImm(tmp, ptr + p_v3[1]*width + p_v3[0])
			if base_row:
				yield Sub(dest_reg, dest_reg, tmp)
			else:
				yield Sub(code.out, dest_reg, tmp)
				x = Mov(dest_reg, code.south)
				code.tag_com_overhead_instr(x)
				yield x

	elif abs(bp_a[1] - bp_b[1]) == 1:
		assert(bp_a[0] == bp_b[0])
		base_col =  (bp_a[0] == 0) # bp_a[0] == bp_b[0] is first col?

		mpoints = [(x-width, y) for x, y in points] if not base_col else points
		p_v4 = (mpoints[3][0], mpoints[3][1]-height)
		p_v3 = (mpoints[2][0], mpoints[2][1]-height)
		p_v2 =  mpoints[1]
		p_v1 =  mpoints[0]
		assert(p_v4[0] < width and p_v4[1] < height)
		assert(p_v3[0] < width and p_v3[1] < height)
		assert(p_v2[0] < width and p_v2[1] < height)
		assert(p_v1[0] < width and p_v1[1] < height)

		with scoped_alloc(code, 1) as tmp:
			yield MemRImm(tmp, ptr + p_v4[1]*width + p_v4[0])
			yield MemRImm(dest_reg , ptr + p_v3[1]*width + p_v3[0])
			yield Sub(code.out, tmp, dest_reg)
			yield MemRImm(tmp, ptr + p_v1[1]*width + p_v1[0])
			yield Add(dest_reg, code.south, tmp)
			yield MemRImm(tmp, ptr + p_v2[1]*width + p_v2[0])
			if base_col:
				yield Sub(dest_reg, dest_reg, tmp)
			else:
				yield Sub(code.out, dest_reg, tmp)
				x = Mov(dest_reg, code.east)
				code.tag_com_overhead_instr(x)
				yield x
	else:
		assert(False) # impossible case

def gen_fullintegral_sum(code, dest_reg, position, shape, ptr, block_size):
	''' Gen integral sum over full integral image.

	This code assumes that each data point is in a block
	maximum one block away from the originating block.
	result: dest_reg

	Note that in contrast with the python implementation,
	a block has the ranges: x[0,w[, y[0,h[
	so width and height of the shape need to be incremented by one
	to be compatible with the violajones sum function.
	'''
	width, height = block_size
	px, py = position
	x, y, w, h = shape
	xx = px + x
	yy = py + y
	# val_sum [dest_reg]

	points =  ((xx, yy), (xx+w-1, yy), (xx, yy+h-1), (xx+w-1, yy+h-1))
	# if all points are in the same neighboring block
	# calculate sum in that block and copy result over
	all_same_block = all_points_in_same_block(points, block_size)
	two_neighbouring_blocks = False
	block_copy_from_right = False
	block_copy_from_below = False
	if all_same_block:
		if not (xx < width):
			block_copy_from_right = True
			xx -= width
		if not (yy < height):
			block_copy_from_below = True
			yy -= height
	else:
		two_neighbouring_blocks = all_points_in_two_neighbouring_blocks(points, block_size)

	if not two_neighbouring_blocks:
		# v1 = im[yy    ][xx    ]
		# v2 = im[yy    ][xx+w-1]
		# v3 = im[yy+h-1][xx    ]
		# v4 = im[yy+h-1][xx+w-1]
		# val_sum =  v1 - v2 - v3 + v4
		with scoped_alloc(code, 1) as tmp:
			for instr in load_mem_value(code, ptr, (xx, yy), dest_reg, block_size): yield instr
			for instr in load_mem_value(code, ptr, (xx+w-1, yy), tmp, block_size): yield instr
			yield Sub(dest_reg, dest_reg, tmp) # r = v1 - v2
			for instr in load_mem_value(code, ptr, (xx, yy+h-1), tmp, block_size): yield instr
			yield Sub(dest_reg, dest_reg, tmp) # r = v1 - v2 - v3
			for instr in load_mem_value(code, ptr, (xx+w-1, yy+h-1), tmp, block_size): yield instr

			if block_copy_from_right or block_copy_from_below:
				yield Add(code.out, dest_reg, tmp) # r = v1 - v2 - v3 + v4
			else:
				yield Add(dest_reg, dest_reg, tmp) # r = v1 - v2 - v3 + v4

		# now handle the shapes out of PE block
		if all_same_block:
			if block_copy_from_right and block_copy_from_below:
				x = Mov(code.out, code.east)
				y = Mov(dest_reg, code.south)
				code.tag_com_overhead_instr(x)
				code.tag_com_overhead_instr(y)
				yield x
				yield y
			elif block_copy_from_right:
				x = Mov(dest_reg, code.east)
				code.tag_com_overhead_instr(x)
				yield x
			elif block_copy_from_below:
				x = Mov(dest_reg, code.south)
				code.tag_com_overhead_instr(x)
				yield x
	else:
		for instr in gen_fullintegral_sum2_2(code, dest_reg, ptr, points, block_size):
			yield instr

def gen_detect_faces_sp(code, position, in_ptr, out_ptr, args, block_size):
	''' Run detector at single pixel position. '''
	haar_classifier = args['haar_classifier']
	sq_integral_ptr = args['sq_integral_ptr']
	integral_ptr = in_ptr
	res_ptr = out_ptr

	width, height = block_size

	with scoped_alloc(code, 2) as (var_norm, fail_cnt):
		# var = get_variance(j, i, int_im, int_sq_im, haar_width, haar_height)
		for x in gen_calc_variance(code, var_norm, position, integral_ptr, sq_integral_ptr, haar_classifier.size, block_size):
			yield x
		
		# var_norm = 1./var
		yield Inv(var_norm, var_norm) # var_norm = var_norm = 1/variance

		# nodetect = False
		# stage fail counter: fail_cnt 
		yield Xor(fail_cnt, fail_cnt, fail_cnt)
		for stage in haar_classifier.stages:
			tag = Nop()
			tag_instr(tag, 'loop_start')
			yield tag
			with scoped_alloc(code, 1) as stage_val:
				yield Xor(stage_val, stage_val, stage_val)
				for feat in stage.features:
					with scoped_alloc(code, 1) as feat_val:
						yield Xor(feat_val, feat_val, feat_val)
						for ishape, shapecoeff in enumerate(feat.shapes):
							shape, coeff = shapecoeff

							# fix to produce same result as python VJ
							# add 1 to width and height as these are inclusive bounds in python impl
							shape = (shape[0], shape[1], shape[2]+1, shape[3]+1)
							
							# accumulate block parts
							mod_shapes = split_shape_across_blocks(shape, position, block_size)
							with scoped_alloc(code, 1) as shape_acc:
								yield Xor(shape_acc, shape_acc, shape_acc) # accum for different parts of shape [shape_acc]
								for mod_shape in mod_shapes:
									with scoped_alloc(code, 1) as val_sum:
										# val_sum = integral_sum(integral, xx, yy, w, h)
										for x in gen_integral_sum(code, val_sum, position, mod_shape, integral_ptr, block_size):
											if ishape > 0: code.tag_com_overhead_instr(x)
											yield x
										yield Add(shape_acc, val_sum, shape_acc)

								# shape_acc = val_sum*coeff*var_norm
								yield Mul(shape_acc, shape_acc, var_norm)
								with scoped_alloc(code, 1) as coeff_r:
									yield Imm(coeff_r, coeff)
									yield Mul(shape_acc, shape_acc, coeff_r)

								# feat_val += shape_acc
								yield Add(feat_val, feat_val, shape_acc)
							 
						# left val belongs to under threshold condition
						# value +=  feat.left_val if feat_val < feat.threshold else feat.right_val
						with scoped_alloc(code, 1) as th_r:
							yield Imm(th_r, feat.threshold)
							yield Cmp(feat_val, th_r)
						with scoped_alloc(code, 1) as val_r:
							yield Imm(val_r, feat.right_val)   # imm = right_val   
							yield Imm(val_r, feat.left_val, 'LT') # if feat_val < mod_feat.threshold: imm = left_val
							yield Add(stage_val, stage_val, val_r)
#
#				# if accumulated value is too small
#				# no face will be detected so go to next window
#				if value < stage.stage_threshold: 
#					nodetect = True
#					break
				# this is not possible in this SIMD implementation, all stages will be processed anyway
				# it is however necessary to signal the first stage failure in a case of rejection
				# so for a first version, we can just increment a failure counter 
				# each time a stage fails for a certain position 
				with scoped_alloc(code, 1) as stage_threshold_r:
					yield Imm(stage_threshold_r, stage.stage_threshold)
					yield Cmp(stage_val, stage_threshold_r)
				with scoped_alloc(code, 1) as const_1:
					yield Imm(const_1, 1)
					yield Add(fail_cnt, fail_cnt, const_1, 'LT')
				j, i = position
				yield MemWImm(res_ptr + i*width + j, fail_cnt)

def gen_detect_faces(code, block_size, args):
	''' main VJ codegen function
	Every PE element calculates the values in the interval [-block_size/2, block_size/2 [
	(in each dimension)
	'''
	haar_classifier = args['haar_classifier']
	width, height = block_size
	haar_width, haar_height = haar_classifier.size

	# make sure that integral image values only need to be fetched
	# from neigbouring PEs, otherwise the code doesn't work
	assert(haar_width < width and haar_height < height) 

	# all functions need to pass the yields by reyielding the results
	for x in gen_equalize_hist(code, block_size):
		yield x
	
	page_size = width*height
	src_ptr = 0
	res_ptr = page_size
	integral_ptr = 2*page_size
	sq_integral_ptr = 3*page_size

	# all functions need to pass the yields by reyielding the results
	for x in gen_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, block_size):
		yield x

	# scan image with VJ window
	# note: for now each block calculates all the positions in range
	# this means results in a lot of communication overhead, but this
	# is acceptable for a first version
	old_progress = 0

	# XXX change these loop over the block by map global->pixel skeleton
	args = {'haar_classifier':haar_classifier, 'sq_integral_ptr':sq_integral_ptr}
	for i in xrange(height):
		for j in xrange(width):
			position = (j, i)
			for x in gen_detect_faces_sp(code, position, integral_ptr, res_ptr, args, block_size):
				yield x

		progress = 100.*float(i)/(height)
		if progress - old_progress > 5.:
			old_progress = progress
			print 'progress: %.2f'%progress

def gen_detect_faces_fullintegral_sp(code, position, in_ptr, out_ptr, args, block_size):
	''' Run detector at single pixel position, fullintegral version. '''
	haar_classifier = args['haar_classifier']
	sq_integral_ptr = args['sq_integral_ptr']
	integral_ptr = in_ptr
	res_ptr = out_ptr

	width, height = block_size

	with scoped_alloc(code, 2) as (inv_var_norm, fail_cnt):
		with scoped_alloc(code, 1) as var_norm:
			# var = get_variance(j, i, int_im, int_sq_im, haar_width, haar_height)
			for x in gen_calc_variance_fullintegral(code, var_norm, position, integral_ptr, sq_integral_ptr, haar_classifier.size, block_size):
				yield x
			yield Inv(inv_var_norm, var_norm) # inv_var_norm = 1/variance

		yield Xor(fail_cnt, fail_cnt, fail_cnt)
		for stage in haar_classifier.stages:
			tag = Nop()
			tag_instr(tag, 'loop_start')
			yield tag
			with scoped_alloc(code, 1) as stage_value:
				for ifeat, feat in enumerate(stage.features):
					with scoped_alloc(code, 1) as feat_value:
						for ishape, shapecoeff in enumerate(feat.shapes):
							shape, coeff = shapecoeff

							# fix to produce same result as python VJ
							# add 1 to width and height as these are inclusive bounds in python impl
							shape = (shape[0], shape[1], shape[2]+1, shape[3]+1)
							
							# accumulate block parts
							# r4 = val_sum = integral_sum(integral, xx, yy, w, h)
							with scoped_alloc(code, 1) as val_sum:
								for x in gen_fullintegral_sum(code, val_sum, position, shape, integral_ptr, block_size):
									yield x

								# r4 = val_sum*coeff/var_norm
								yield Mul(val_sum, val_sum, inv_var_norm)

								if coeff != -1.:
									with scoped_alloc(code, 1) as coeff_reg:
										yield Imm(coeff_reg, coeff)
										if ishape == 0:
											yield Mul(feat_value, val_sum, coeff_reg)
										else:
											yield Mul(val_sum, val_sum, coeff_reg)
											yield Add(feat_value, feat_value, val_sum)
								else:
									if ishape == 0:
										yield Neg(feat_value, val_sum)
									else:
										yield Sub(feat_value, feat_value, val_sum)
							 
						# left val belongs to under threshold condition
						# value +=  feat.left_val if feat_val < feat.threshold else feat.right_val
						with scoped_alloc(code, 1) as th_r:
							yield Imm(th_r, feat.threshold)
							yield Cmp(feat_value, th_r)
						with scoped_alloc(code, 1) as val_r:
							yield Imm(val_r, feat.right_val)   # imm = right_val   
							yield Imm(val_r, feat.left_val, 'LT') # if feat_value < mod_feat.threshold: imm = left_val
							if ifeat == 0:
								yield Mov(stage_value, val_r)
							else:
								yield Add(stage_value, stage_value, val_r)

				with scoped_alloc(code, 1) as stage_th_r:
					yield Imm(stage_th_r, stage.stage_threshold)
					yield Cmp(stage_value, stage_th_r)
				with scoped_alloc(code, 1) as const_1:
					yield Imm(const_1, 1)
					yield Add(fail_cnt, fail_cnt, const_1, 'LT')
				j, i = position
				yield MemWImm(res_ptr + i*width + j, fail_cnt)

def gen_detect_faces_fullintegral(code, block_size, args):
	''' Main VJ codegen function.

	Every PE element calculates the values in the interval [-block_size/2, block_size/2 [
	(in each dimension)
	This version uses an integral image across the entire image.
	'''
	pe_dim = args['pe_dim']
	haar_classifier = args['haar_classifier']
	width, height = block_size
	haar_width, haar_height = haar_classifier.size

	# make sure that integral image values only need to be fetched
	# from neigbouring PEs, otherwise the code doesn't work
	assert(haar_width < width and haar_height < height) 

	# all functions need to pass the yields by reyielding the results
	for x in gen_equalize_hist(code, block_size):
		yield x
	
	page_size = width*height
	src_ptr = 0
	res_ptr = page_size
	integral_ptr = 2*page_size
	sq_integral_ptr = 3*page_size

	# all functions need to pass the yields by reyielding the results
	for x in gen_full_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, pe_dim, block_size):
		# result: only 55000 instructions for 32x32
		#tag_instr(x, 'section:full_integral_image')
		yield x

	old_progress = 0
	args = {'haar_classifier':haar_classifier, 'sq_integral_ptr':sq_integral_ptr}
	for i in xrange(height):
		for j in xrange(width):
			pos = (j, i)
			for x in gen_detect_faces_fullintegral_sp(code, pos, integral_ptr, res_ptr, args, block_size):
				yield x

		progress = 100.*float(i)/(height)
		if progress - old_progress > 5.:
			old_progress = progress
			print 'progress: %.2f'%progress


def gen_detect_faces_stage_outer(code, block_size):
	''' Main VJ codegen function.

	Every PE element calculates the values in the interval [-block_size/2, block_size/2 [
	(in each dimension)
	This is variation on the normal codegen, looping over stages in the outer loop
	instead of over pixels. 
	Note: the src memory bank is used to store normalisation factors.
	'''
	## register usage:
	## r0: stage fail counter
	## r1: accum for different parts of splitted shape
	## r2: value
	## r3: feat_val
	## r4: val_sum
	## r5: tmp
	## r6:
	## r7: var_norm

	haar_classifier = args['haar_classifier']
	width, height = block_size
	haar_width, haar_height = haar_classifier.size

	# make sure that integral image values only need to be fetched
	# from neigbouring PEs, otherwise the code doesn't work
	assert(haar_width < width and haar_height < height)

	# all functions need to pass the yields by reyielding the results
	for x in gen_equalize_hist(code, block_size):
		yield x

	page_size = width*height
	src_ptr = 0
	res_ptr = page_size
	integral_ptr = 2*page_size
	sq_integral_ptr = 3*page_size

	# all functions need to pass the yields by reyielding the results
	for x in gen_integral_image(code, src_ptr, integral_ptr, sq_integral_ptr, block_size):
		yield x

	# scan image with VJ window
	# note: for now each block calculates all the positions in range
	# this means results in a lot of communication overhead, but this
	# is acceptable for a first version
	old_progress = 0

	# first write the normalisation values and store them in the src memory bank
	for i in xrange(height):
		for j in xrange(width):
			position = (j, i)

			# var = get_variance(j, i, int_im, int_sq_im, haar_width, haar_height)
			# gen_calc variance uses r1, r2 and r7 (result)
			for x in gen_calc_variance(code, position, integral_ptr, sq_integral_ptr, haar_classifier.size, block_size):
				yield x

			# var_norm = 1./var
			yield Inv(code.r(7), code.r(7)) # r7 = var_norm = 1/variance
			yield MemWImm(src_ptr + i*width + j, code.r(7))

	# loop over stages instead of pixel in outer loop
	for stage_nr, stage in enumerate(haar_classifier.stages):
		# load previous fail counter???
		for i in xrange(height):
			for j in xrange(width):
				position = (j, i)
				# if stage == 0, clear counter, otherwise load previous
				if stage_nr == 0:
					yield Xor(code.r(0), code.r(0), code.r(0))
				else:
					yield MemRImm(code.r(0), res_ptr + i*width + j)
				# load previously calculated normalisation value from output
				# into r7
				yield MemRImm(code.r(7), src_ptr + i*width + j)

				# value = 0. [r2]
				yield Xor(code.r(2), code.r(2), code.r(2))
				for feat in stage.features:
					# feat_val = 0. [r3]
					yield Xor(code.r(3), code.r(3), code.r(3))
					for ishape, shapecoeff in enumerate(feat.shapes):
						shape, coeff = shapecoeff

						# fix to produce same result as python VJ
						# add 1 to width and height as these are inclusive bounds in python impl
						shape = (shape[0], shape[1], shape[2]+1, shape[3]+1)

						# accumulate block parts
						mod_shapes = split_shape_across_blocks(shape, position, block_size)
						yield Xor(code.r(1), code.r(1), code.r(1)) # accum for different parts of shape [r1]
						for mod_shape in mod_shapes:

							# r4 = val_sum = integral_sum(integral, xx, yy, w, h)
							# gen_integral_sum uses r4(result) and r5
							for x in gen_integral_sum(code, position, mod_shape, integral_ptr, block_size):
								if ishape > 0: code.tag_com_overhead_instr(x)
								yield x
							yield Add(code.r(1), code.r(4), code.r(1))

						# r4 = val_sum*coeff*var_norm
						yield Mul(code.r(4), code.r(1), code.r(7))
						yield Imm(coeff)
						yield Mul(code.r(4), code.r(4), code.imm)

						# feat_val += r4
						yield Add(code.r(3), code.r(3), code.r(4))

					# left val belongs to under threshold condition
					# value +=  feat.left_val if feat_val < feat.threshold else feat.right_val
					yield Imm(feat.threshold)
					yield Cmp(code.r(3), code.imm)
					yield Imm(feat.right_val)   # imm = right_val
					yield Imm(feat.left_val, 'LT') # if r3 < mod_feat.threshold: imm = left_val
					yield Add(code.r(2), code.r(2), code.imm)
#
#				# if accumulated value is too small
#				# no face will be detected so go to next window
#				if value < stage.stage_threshold:
#					nodetect = True
#					break
				# this is not possible in this SIMD implementation, all stages will be processed anyway
				# it is however necessary to signal the first stage failure in a case of rejection
				# so for a first version, we can just increment a failure counter
				# each time a stage fails for a certain position
				yield Imm(stage.stage_threshold)
				yield Cmp(code.r(2), code.imm)
				yield Imm(1)
				yield Add(code.r(0), code.r(0), code.imm, 'LT')
				yield MemWImm(res_ptr + i*width + j, code.r(0))

#			if not nodetect:
#				detection = (j, i, haar_width, haar_height)
#				print detection
#				detections.append(detection)

		progress = 100.*float(stage_nr)/len(haar_classifier.stages)
		if progress - old_progress > 5.:
			old_progress = progress
			print 'progress: %.2f'%progress
#	return detections
	

def convert_pixelmap_to_detections(pixelmap, violajones_size):
	height, width = len(pixelmap), len(pixelmap[0])

	detections = []
	for i in xrange(height):
		for j in xrange(width):
			if pixelmap[i][j] == 0:
				detection = (j, i, violajones_size[0], violajones_size[1])
				detections.append(detection)
	return detections

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

gen_detect_faces_opt = optimiser_wrapper(gen_detect_faces)
gen_detect_faces_fullintegral_opt = optimiser_wrapper(gen_detect_faces_fullintegral)


def run_detector(block_size, implementation, image_filename, cascade_filename, res_filename_prefix):
	from blip.simulator import interpreter
	from blip.support import imageio
	import violajones.reference

	# first load the cascade
	cascade = violajones.parse_haar.parse_haar_xml(cascade_filename)
	print cascade

	image = imageio.read(image_filename)
	if not image: raise Exception('image %s not found or not supported'%image_filename)

	print 'XXX histogram equalisation is not implemented yet, use violajones impl'
	print '    before executing simulator'
	image = violajones.reference.equalizeHist(image)
	im_size = len(image[0]), len(image)

	pe_dim = [s//b for s,b in zip(im_size, block_size)]

	args = {'haar_classifier':cascade, 'pe_dim':pe_dim}
	# now execute the codegen
	code = Code()
	code.set_generator(implementation, block_size, args)
	#print '# instructions: %i'%(code.instr_size())

	sim = interpreter.Interpreter(code, image, block_size, 4)
	sim.run()

	detections_pixmap = sim.gen_output_image(1) # result is saved in first buffer

	# convert the number of rejections in the stages to detections
	detections = convert_pixelmap_to_detections(detections_pixmap, cascade.size)
	print 'detections:', detections
	detections_im = visualisation.draw_faces(image, detections)

	imageio.write(res_filename_prefix + '_pixmap.png', detections_pixmap, 1)
	imageio.write(res_filename_prefix + '.png', detections_im, 3)


if __name__ == '__main__':
	from blip.code.codegen import load_codegen
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-b', '--block_size', dest='block_size', default=64, help='PE block size')
	parser.add_option('-s', '--src_image', dest='src_image', default='data/vakgroep128_64.png', help='source image')
	parser.add_option('-o', '--output', dest='output', default='blip_detections', help='output prefix')
	parser.add_option('-c', '--cascade', dest='cascade', default='data/haarcascade_frontalface_alt.xml', help='haar cascade')
	parser.add_option('--codegen_implementation', dest='codegen_impl', default='gen_code.gen_detect_faces',\
			  help='override the codegen implementation')
	(options, args) = parser.parse_args()

	block_size = (int(options.block_size), int(options.block_size))
	cascade_filename = options.cascade
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
		run_detector(block_size, implementation, image_filename, cascade_filename, res_filename)
	except Exception, e:
		import pdb
		import sys
		pdb.post_mortem(sys.exc_traceback)
		raise e
	
def split_filter_haar_example(cascade_filename):
	cascade = parse_haar.parse_haar_xml(cascade_filename)
	print cascade

	# now apply it for the first cascade at some position
	position = (10, 6)
	block_size = 24

	stage = cascade.stages[0]
	for feature in stage.features:
		res = split_filter(feature, position, block_size)
		print res
		print '\n'.join([str(x) for x in feature.shapes])
		print '->'
		print '\n'.join([str(x) for x in res.shapes])

