.. Blip documentation master file, created by
   sphinx-quickstart on Tue Jan 25 11:04:30 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Blip's documentation!
================================
This documentation describes the Blip processor architecture 
and a high level simulator for this architecture.

For more information on the Blip architecture, see :doc:`blip architecture <bliparch>`.
The software implementation can be found here: :doc:`blip simulator <blipsim>`.
A description of possible tasks is given here: :doc:`blip tasks <tasks>`.

There are two ways to write code for blip, using the meta assembler and
using the compiler modeled on OpenCl. For the first approach, the automatic optimiser
is described in :doc:`trace optimiser <traceoptimiser>`, the compiler is discribed here:
:doc: `blip compiler <compiler>`.

 
.. toctree::
   :maxdepth: 2

   bliparch
   blipsim
   compiler
   traceoptimiser
   tasks

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

