from __future__ import with_statement
import sys
import traceback
import pdb
from itertools import izip
from optparse import OptionParser

def skip_test(f):
	''' Decorator to mark tests to skip, wrapping pytest functionality if available. '''
	f.__skip__ = True
	try:
		from pytest import skip
		def new_f(*args):
			skip('skip this test')
			return f(*args)
		new_f.__name__ = f.__name__
		new_f.__skip__ = True
		return new_f
	except:
		return f

def compare_images(im_a, im_b):
	h_a, w_a = len(im_a), len(im_a[0])
	h_b, w_b = len(im_b), len(im_b[0])
	if h_a != h_b or w_a != w_b: raise Exception('sizes of both image must be equal')
	err = 0.
	for i in xrange(h_a):
		for j in xrange(w_a):
			v_a = im_a[i][j]
			v_b = im_b[i][j]
			err += abs(v_a - v_b)
	return err
def compare_vectors(v_a, v_b):
        if len(v_a) != len(v_b): raise Exception('sizes not equal')
        return sum(abs(a-b) for a,b in izip(v_a, v_b))

def get_test_options(optparser = None):
	if not optparser:
		optparser = OptionParser()
	optparser.add_option('--skip', dest='skip', default='', help='tests to skip')
	optparser.add_option('--runonly', dest='run_only', default='', help='select tests')
	optparser.add_option('-b', '--batch', action='store_true', dest='do_batch', default=False, help='batch mode')
	return optparser

def parse_test_options(opt_parser):
	test_options = {}
	(options, args) = opt_parser.parse_args()
	test_options['batch'] = options.do_batch
	if options.skip and options.run_only:
		raise Exception('illegal arguments: can\'t use both skip and run_only')

	if options.skip:
		skipped_tests = [x.strip() for x in options.skip.split(',') if x.strip()]
		test_options['skip'] = skipped_tests
	elif options.run_only:
		run_only = [x.strip() for x in options.run_only.split(',') if x.strip()]
		test_options['run_only'] = run_only

	return test_options

def run_tests(tests, options = {}):
    batch = options['batch'] if 'batch' in options else False
    skiplist = options['skip'] if 'skip' in options else []
    run_only = options['run_only'] if 'run_only' in options else []
    print run_only
    tests = [t for t in tests if t.__name__ in run_only] if run_only else tests
    print '-'*40
    print 'running tests...'
    succeeded = 0
    skipped = 0
    for test in tests:
        print '-'*40
        print test.__name__
        if test.__name__ in skiplist or getattr(test, '__skip__', False):
            print '[S]  test skipped'
            skipped += 1
            continue
        succes = True 
        try:
            test()
        except Exception, e:
            succes = False
            print '>>> ' + str(e) + ' <<<'
            if not batch:
                traceback.print_tb(sys.exc_traceback)
                pdb.post_mortem(sys.exc_traceback)
        if succes:
            succeeded+= 1
        else:
            print '[F]  test %s failed'%test.__name__

    print
    print '-'*40
    print '%i tests of total %i tests succeeded, %i skipped'%(succeeded, len(tests), skipped)
    return succeeded == len(tests)

# instruction matcher
# based on code by Antonio Cuni (Pypy)
class InvalidMatch(Exception): pass

class OpMatcher(object):
	def __init__(self, ops):
		self.ops = ops
		self.alpha_map = {}

	@classmethod
	def parse_ops(cls, src):
		ops = [cls.parse_op(line) for line in src.splitlines()]
		return [op for op in ops if op is not None]

	@classmethod
	def parse_op(cls, line):
		# strip comment
		if '#' in line:
			line = line[:line.index('#')]
		line = line.strip()
		if not line:
			return None
		if line == '...':
			return line

		cond = None
		opname = None
		argstr = ''
		if '{' in line:
			opname, _, condarg = line.partition('{')
			cond, _, argstr = condarg.partition('}')
			cond = cond.strip()
		else:
			opname, _, argstr = line.partition(' ')
		args = argstr.strip()
		args = [x.strip() for x in args.split(',') if x.strip()]
		return (opname.strip(), args, cond)

	@classmethod
	def instr_args(cls, instr):
		''' Hack to convert from attributes to arg array '''
		if instr.opcode() == 'imm':
			return  [str(instr.dest), str(instr.value)]
		else:
			args = []
			try:
				args.append(instr.dest)
			except AttributeError: pass

			# use src2 to detect the instruction format
			src2 = None
			try:
				src2 = instr.src2
			except AttributeError: pass
	
			if src2:
				try:
					args.append(instr.src1)
					args.append(instr.src2)
				except AttributeError: pass
			else:
				try:
					args.append(instr.src)
				except AttributeError: pass
			return [str(x) for x in args]

	@classmethod
	def is_const(cls, v1):
		return isinstance(v1, (float, int))

	def match_var(self, v1, exp_v2):
		assert v1 != '_'
		if exp_v2 == '_':
			return True
		if self.is_const(v1) or self.is_const(exp_v2):
			return v1 == exp_v2
		if v1 not in self.alpha_map:
			self.alpha_map[v1] = exp_v2
		# XXX debug
		res = self.alpha_map[v1] == exp_v2
		#print 'match var', v1, exp_v2, self.alpha_map[v1], res
		return res

	def _assert(self, cond, message):
		if not cond:
			raise InvalidMatch(message)

	def match_op(self, op, op_cnt, (exp_opname,  exp_args, exp_cond), exp_op_cnt):
		opname = op.opcode().lower()
		exp_opname = exp_opname.lower()
		def error_str(msg):
			return '%s --> %i: %s <-> %i: %s'%(msg, op_cnt, opname, exp_op_cnt, exp_opname)
		self._assert(opname == exp_opname, error_str('opcode mismatch'))
		self._assert(op.cond() == exp_cond, error_str("condition mismatch %s and %s"%(op.cond(), exp_cond)))
		args = self.instr_args(op)
		self._assert(len(args) == len(exp_args), error_str("wrong number of arguments, %i and %i"%(len(args), len(exp_args))))
		for arg, exp_arg in zip(args, exp_args):
			self._assert(self.match_var(arg, exp_arg), error_str("variable mismatch %s and %s"%(str(arg), str(exp_arg))))

	def _next_op(self, iter_ops, assert_raises=False):
		try:
			i, op = iter_ops.next()
		except StopIteration:
			self._assert(assert_raises, "not enough operations")
			return
		else:
			self._assert(not assert_raises, "operation list too long")
			return i, op

	def match_until(self, until_op, until_op_cnt, iter_ops):
		while True:
			op_cnt, op = self._next_op(iter_ops)
			try:
				# try to match the op, but be sure not to modify the
				# alpha-renaming map in case the match does not work
				alpha_map = self.alpha_map.copy()
				self.match_op(op, op_cnt, until_op, until_op_cnt)
			except InvalidMatch:
				# it did not match: rollback the alpha_map, and just skip this
				# operation
				self.alpha_map = alpha_map
			else:
				# it matched! The '...' operator ends here
				return op_cnt, op

	def match_loop(self, expected_ops):
		"""
		A note about partial matching: the '...' operator is non-greedy,
		i.e. it matches all the operations until it finds one that matches
		what is after the '...'
		"""
		iter_exp_ops = enumerate(expected_ops)
		iter_ops = enumerate(self.ops)
		for exp_i, exp_op in iter_exp_ops:
			if exp_op == '...':
				# loop until we find an operation which matches
				exp_i, exp_op = iter_exp_ops.next()
				i, op = self.match_until(exp_op, exp_i, iter_ops)
			else:
				i, op = self._next_op(iter_ops)
			self.match_op(op, i, exp_op, exp_i)
		# make sure we exhausted iter_ops
		self._next_op(iter_ops, assert_raises=True)

	def match(self, expected_src):
		expected_ops = self.parse_ops(expected_src)
		try:
			self.match_loop(expected_ops)
		except InvalidMatch, e:
			#raise # uncomment this and use py.test --pdb for better debugging
			return False, str(e)
		else:
			return True, ''

def parse_match_error(err):
	op_i = op_exp_i = -1
	err_str = err
	try:
		if '-->' in err and '<->' in err:
			err_str, _, mismatched = err.partition('-->')
			op, _, exp_op = mismatched.partition('<->')
			op_i, _, opname = op.partition(':')
			op_i = int(op_i)
			op_exp_i, _, exp_opname = exp_op.partition(':')
			op_exp_i = int(op_exp_i)
	except:
		pass
	return err_str, op_i, op_exp_i 

def match_code(code, pattern):
	matched, err = OpMatcher(code).match(pattern)
	if not matched:
		err_str, op_i, op_exp_i = parse_match_error(err)
		print 'code doesn\'t match: ' + err_str
		print'new trace:'
		print '\n'.join(('> ' if i == op_i else '  ') + str(x) for i, x in enumerate(code))
		print
		print
		print 'pattern'
		print '\n'.join(('> ' if i == op_exp_i else '  ') + x for i, x in enumerate(pattern.strip().split('\n')))
		return False
	return True

if __name__ == '__main__':
	from optparse import OptionParser
	parser = OptionParser()
	(options, args) = parser.parse_args()

