import sys
# add source dir
sys.path.insert(0, '..')
sys.path.insert(0, '../tools')

import random
from itertools import izip

from blip.blipcode import *
from blip.simulator.interpreter import Interpreter
from blip.code.codegen import Code

from tester import compare_vectors, compare_images, run_tests
from tester import get_test_options, parse_test_options

# tests
def test_bbs():

    # setttings
    width, height = 9, 9 # size only effects execution speed so keep it small
    block_size = (width, height)
    nr_images = 5
    th = 20
    alpha = 0.9

    class RefBBS(object):
        def __init__(self, alpha, th, im_size):
            self.im_size = im_size
            self.alpha = alpha
            self.th = th
            self.background = [0. for x in xrange(im_size)]
        def bbs(self, im_p, b_p):
            return self.alpha*im_p + (1-self.alpha)*b_p
        def process(self, image):
            self.background = [self.bbs(im, b) for im, b in izip(image, self.background)]
            return [255. if abs(im-b) > self.th else 0. for im, b in izip(image, self.background)]
    def run_ref(images, th, alpha):
        image0 = images[0]
        im_size = len(image0)
        refBBS = RefBBS(alpha, th, im_size)
        return [refBBS.process(im) for im in images]
 
    def run_test(images, th, alpha, block_size):
        image0 = images[0]
        im_size = len(image0)
        bwidth, bheight = block_size
        assert(im_size == bwidth * bheight) # only one pe

        code = Code()
        code.set_generator(gen_bbs, block_size, {'th':th, 'alpha':alpha})

        output = []
        sim = None
        for im in images:
            # interpreter expects 2D array
            im_tr = [[im[i*width + j] for j in xrange(bwidth)] for i in xrange(bheight)]
            if not sim:
                sim = Interpreter(code, im_tr, block_size)
            else:
                # restart code gen
                sim.reset()
                # set new image
                sim.set_src_image(im_tr)
            sim.run()
            im_out = sim.gen_output_image(1)
            # convert to 1D vector
            im_out_1D = []
            for row in im_out:
                for v in row:
                    im_out_1D.append(v)
            output.append(im_out_1D)
        return output

    images = [[random.randint(0, 255) for x in xrange(width*height)] for y in xrange(nr_images)]
    ref_output = run_ref(images, th, alpha)
    test_output = run_test(images, th, alpha, block_size)

    correct = all(compare_vectors(x,y) < 0.01 for x, y in izip(ref_output, test_output))
    assert correct

def _test_image_function_single_pe(codegen, args, ref_implementation):
    # setttings
    width, height = 9, 9 # size only effects execution speed so keep it small
    block_size = (width, height)

    def run_test(image, args, block_size):
        im_size = len(image[0]), len(image)
        bwidth, bheight = block_size
        assert(im_size == block_size) # only one pe

        code = Code()
        code.set_generator(codegen, block_size, args)

        sim = Interpreter(code, image, block_size)
        sim.run()
        output = sim.gen_output_image(1, False)
        return output

    image = [[random.randint(0, 255) for x in xrange(width)] for y in xrange(height)]
    ref_output = ref_implementation(image, args) 
    test_output = run_test(image, args, block_size)

    if not (compare_images(ref_output, test_output) < 0.01):
	print 'ref'
        print '\n'.join(str(y) for y in ref_output)
	print 'test'
        print '\n'.join(str(y) for y in test_output)
        return False
    return True

def test_threshold():
    th =  160
    def th_ref(image, args):
	th = args['th']
        return [[(255 if x > th else 0) for x in y] for y in image]
    assert _test_image_function_single_pe(gen_threshold, {'th':th}, th_ref)

def test_threshold2():
    th =  160
    def th_ref(image, args):
	th = args['th']
        return [[(255 if x > th else 0) for x in y] for y in image]
    assert  _test_image_function_single_pe(gen_threshold_2, {'th':th, 'disable_optimization':True}, th_ref)

def test_copy_to_out():
    def ref(image, args):
        return [[x for x in y] for y in image]
    assert  _test_image_function_single_pe(gen_copy_to_out, {}, ref)

def test_const_gray():
    def ref(image, args):
        return [[128 for x in y] for y in image]
    return  _test_image_function_single_pe(gen_gray_image_code, {}, ref)

def test_gen_invert_im():
    def ref(image, args):
        return [[255-x for x in y] for y in image]
    assert  _test_image_function_single_pe(gen_invert_im, {}, ref)

def all_test(options = {}):
    tests = [\
        test_bbs,\
        test_threshold,\
        test_threshold2,\
        test_copy_to_out,\
	test_const_gray,\
	test_gen_invert_im,\
    ]
    return run_tests(tests, options)

if __name__ == '__main__':
    opt_parser = get_test_options()
    test_options = parse_test_options(opt_parser)
    if not all_test(test_options): exit(1)

