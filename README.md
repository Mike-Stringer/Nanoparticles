

<div align="center">
  <img src="https://github.com/Mike-Stringer/Nanoparticles/blob/master/RW.png"><br>
</div>

*Polymers & Nanoparticles*
=============
Written in C++ with CUDA

[Cleaned code](https://github.com/Mike-Stringer/Nanoparticles/blob/master/Nanoparticles/Nanoparticles/PolymerMain.cu)

Nanoparticle Hyperthreading

- Polymers initialized in a grid of a possible 3D node positions

- Polymers ends and hairpin bends can move

- % of nodes are filled with NPs, these can be clumped or individual

- Polymer length can be changed

- Time series is introduced to allow polymer to relax to radius of gyration

- CUDA is leveraged to parallelize random walk many polymers across the matrix of nodes

- batches of loops refer to randomly choosing a point on the polymer, if that point can move (hairpin or end) and if it is not blocked by NPs, then direction of movement is chosen.

Please see [poster](https://github.com/Mike-Stringer/Nanoparticles/blob/master/PolymerDynamicsPoster.pdf) or 
[report](https://github.com/Mike-Stringer/Nanoparticles/blob/master/PolymerDynamics_2014_MichaelStringer.pdf) for more details!   
	  
	  
