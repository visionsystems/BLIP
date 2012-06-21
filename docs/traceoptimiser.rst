***************
Trace optimiser
***************
In order to optimise segments of assembly code, we can use the trace optimiser.
The generated assembly code is chopped up into trace fragments, all optimisers
are run on this fragment and the optimised code is emitted again.

=========
Principle
=========
The input for the trace optimiser is a generated instruction stream and the alloc/free
information for the registers. Next, each pass is applied to the trace fragment,
changing the instructions locally and updating the alloc/free information.
One important limitation is that conditional values are not handled right now.

================
Optimiser passes
================
In the following section the optimiser passes are described.

Immediate value optimiser
-------------------------
This optimiser tries to keep regularly used immediate values inside free registers
instead of reloading them multiple times. The most difficult part is to keep consistency
between trace fragments, making sure that at the end of a fragment all values are
in the same registers as in the unoptimised version.

Peephole optimiser
------------------
The peephole optimiser tries to optimise small instruction sequences using pattern matching.
Right now only nops are removed that do not contain any tags.

Memory optimiser
----------------
This optimiser tries to keep regularly read memory values inside free registers
instead of reloading them multiple times. This pass should be extended with store sinking.

=====
Notes
=====
It should be possible to unify the optimisations of the compiler and the trace optimiser.
If the instruction stream -> SSA representation can be improved both would use the same format.

