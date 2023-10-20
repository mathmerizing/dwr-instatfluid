# README #

This README documents whatever steps are necessary to get the application
  DTM++.Project/dwr/dwr-instatfluid
up and running.

### What is this repository for? ###

* Space-time adaptive instationary incompressible Navier-Stokes solver

### How do I get set up? ###

* Dependencies
deal.II v9.3.0 at least, installed via candi, cf. https://github.com/dealii/candi

* Configuration
```
cmake .
make
```

* Run (single process)
```
./dwr-instatfluid
```


### Who do I talk to? ###

* Principial Author
    * Dr.-Ing. Dipl.-Ing. Uwe Köcher (koecher@hsu-hamburg.de, dtmproject@uwe.koecher.cc)
    * Julian Roth (roth@ifam.uni-hannover.de)
    * Jan Philipp Thiele (thiele@ifam.uni-hannover.de)
* Contributors
    * Marius P. Bruchhaeuser (bruchhaeuser@hsu-hamburg.de)

Remark. DTM++ is free software.

If you write scientific publication using results obtained by reusing parts
of DTM++, or specifically DTM++/dwr-navier_stokes, especially by reusing the
datastructures, algorithms and/or supporting parameter/data input/output
classes, you are willing to cite the following three publications:

- J. Roth, J.P. Thiele, U. Köcher and T. Wick. "Tensor-Product Space-Time Goal-Oriented Error Control and Adaptivity With Partition-of-Unity Dual-Weighted Residuals for Nonstationary Flow Problems" Computational Methods in Applied Mathematics, 2023. https://doi.org/10.1515/cmam-2022-0200

- U. Koecher, M.P. Bruchhaeuser, M. Bause: "Efficient and scalable
  data structures and algorithms for goal-oriented adaptivity of space-time
  FEM codes", SoftwareX 10(July-December):1-6, 100239, 2019.

- U. Koecher: "Variational space-time methods for the elastic wave equation
  and the diffusion equation", Ph.D. thesis,
  Department of Mechanical Engineering of the Helmut-Schmidt-University,
  University of the German Federal Armed Forces Hamburg, Germany, p. 1-188,
  urn:nbn:de:gbv:705-opus-31129, 2015. Open access via:
  http://edoc.sub.uni-hamburg.de/hsu/volltexte/2015/3112/


### License ###
Copyright (C) 2012-2023 by Uwe Köcher, Julian Roth, Jan Philipp Thiele and contributors

This file is part of DTM++.

DTM++ is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

DTM++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public License
along with DTM++. If not, see <http://www.gnu.org/licenses/>.
Please see the file
	./LICENSE
for details.
