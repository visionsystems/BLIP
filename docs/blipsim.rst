*********
Simulator
*********

Blip simulator
==============

opcodes
-------
Instruction definition and processor paramereters.

.. automodule:: blip.simulator.opcodes
   :members:

interpreter
-----------
High level interpreter for Blip. Example usage::

    block_size = (64, 64)
    args = {'th':100} # threshold value

    code = Code()
    code.set_generator(blipcode.gen_threshold, block_size, args)

    image = imageio.read(imageFileName)

    interpreter = Interpreter(code, image, block_size)
    interpreter.run()

    out_image = interpreter.gen_output_image()

.. automodule:: blip.simulator.interpreter
   :members:

blipcode
--------
Code generators, mostly image processing.

.. automodule:: blip.blipcode
   :members:

codegen
-------
Code management.

.. automodule:: blip.code.codegen
   :members:

trace_optimise
--------------

.. automodule:: blip.code.trace_optimiser
   :members:


Viola Jones implementation
==========================
Implementation of Viola Jones face detector on the Blip simulator.
Eventually this should be moved into a seperate project.

reference
----------
VJ reference implementation in pure Python.

.. automodule:: violajones.reference
   :members:

parse_haar
----------
Parsing of OpenCV haar cascade XML files.

.. automodule:: violajones.parse_haar
   :members:

gen_code
--------------
Main entry point for running VJ on top of Blip.

.. automodule:: violajones.gen_code
   :members:

draw
----
Draw VJ output, mostly used for debugging shape splitting

.. automodule:: violajones.draw
   :members:


Analysis
========

analysis
--------
Main entry point for Blip analysis.

.. automodule:: blip.analysis.analysis
   :members:

sweep_analysis
--------------
Batch analysis for parameter sweeps.

.. automodule:: blip.analysis.sweep_analysis
   :members:


Support modules
===============

imageio
-------
Wrapper around OpenCV or png.py image handling.

.. automodule:: blip.support.imageio
   :members:

png
----------
Pure Python PNG image handling.

:doc:`PNG documentation <pngdoc>`

svgfig
------
Pure Python SVG image generation.

:doc:`svgfig documentation <svgfigdoc>`


.. toctree::
   :maxdepth: 2

   pngdoc
   svgfigdoc


