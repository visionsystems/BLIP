************
Architecture
************

Architecture overview
=====================
The blip architecture is a combination of an array of processing elements (PE)
operating in a SIMD fashion, controlled by a central processor (CP). 
Each processing element stores a certain block of the total input image.
This processing element is responsible for calculating the output value for
each pixel within its own block. Data from a neigbouring PE can be fetched by
means of explicit communication with the four-connected neigbours.
The central processor runs a dynamic code generator, generating code on the fly for all PE's.

Processing element
==================
The following image shows the architecture of a single processing element:

.. image:: pe.*
   :height: 400px

The PE consists of an ALU, 8 registers and some block memory banks.

Instruction set
---------------
The current version of BLIP has an RISC-inspired three operand instruction set
with conditional execution.

Block buffer memory
-------------------
The first block buffer is used to store the initial image data. The second buffer
will contain the output data. This memory block has a dual read port so that
the control processor can read out the result for each processing element.


Array configuration
===================
The processing elements are arranged in a rectangular grid. Each PE has an output
register that is connected to the west, east, north and south ports of its
east, west, south and north neigbours respectively.

Control processor
=================

.. toctree::
   :maxdepth: 2
