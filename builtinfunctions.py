from BlipCompiler import BuiltinFunction

class GetGroupId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		# actually get_group_id should have a parameters to select the dimension
		# for now, assume a single dimension
		BuiltinFunction.__init__(self, 'get_group_id', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('grid', None, dest.name)
class GetGlobalId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'get_global_id', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		arg = self.func_args[0]
		yield ('glid', None, dest.name, arg.name)
class GetLocalId(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'get_local_id', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		arg = self.func_args[0]
		yield ('lid', None, dest.name, arg.name)
class LoadWest(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadWest', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'west')
class LoadEast(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadEast', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'east')
class LoadNorth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadNorth', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'north')
class LoadSouth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'loadSouth', 0, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		yield ('mov', None, dest.name, 'south')
class SendToOutput(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'sendOut', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
# slightly more efficient implementation of communication, combine send/receive
class TransferFromWest(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromWest', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'west')
class TransferFromEast(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromEast', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'east')
class TransferFromNorth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromNorth', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'north')
class TransferFromSouth(BuiltinFunction):
	def __init__(self, func_args, **kwargs):
		BuiltinFunction.__init__(self, 'transferFromSouth', 1, func_args, **kwargs)
	def gen_code(self, codegen, dest):
		src = self.func_args[0].name
		yield ('mov', None, 'out', src)
		yield ('mov', None, dest.name, 'south')
