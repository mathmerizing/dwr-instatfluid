/**
 * @file   ParameterHandler.cc
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * @author Julian Roth (JR)
 * 
 * @Date 2022-04-26, high/low order problem, JR
 * @Date 2022-01-17, Added Newton parameters, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-06, stokes, UK
 * @date 2018-09-14, unified to other DTM programs, UK
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2017-02-06, UK
 */

/*  Copyright (C) 2012-2022 by Uwe Koecher and contributors                   */
/*                                                                            */
/*  This file is part of DTM++.                                               */
/*                                                                            */
/*  DTM++ is free software: you can redistribute it and/or modify             */
/*  it under the terms of the GNU Lesser General Public License as            */
/*  published by the Free Software Foundation, either                         */
/*  version 3 of the License, or (at your option) any later version.          */
/*                                                                            */
/*  DTM++ is distributed in the hope that it will be useful,                  */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             */
/*  GNU Lesser General Public License for more details.                       */
/*                                                                            */
/*  You should have received a copy of the GNU Lesser General Public License  */
/*  along with DTM++.   If not, see <http://www.gnu.org/licenses/>.           */

// PROJECT includes
#include <fluid/parameters/ParameterHandler.hh>

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes

namespace fluid {

ParameterHandler::
ParameterHandler() {
	declare_entry(
		"dim",
		"2",
		dealii::Patterns::Integer(),
		"dim"
	);

    declare_entry(
            "problem",
            "Stokes",
            dealii::Patterns::Anything()
    );

    declare_entry(
		"primal only",
		"false",
		dealii::Patterns::Bool(),
		"run only primal problem (true) or primal + dual problem + error estimator (false)"
	);

	enter_subsection("Fluid Problem Specification"); {
		declare_entry(
			"symmetric stress",
			"false",
			dealii::Patterns::Bool(),
			"determines whether the symmetric stress tensor should be used"
		);

		declare_entry(
			"primal projection type",
			"none",
			dealii::Patterns::Anything(),
			"determines which projection should be used for div-correction"
		);

		declare_entry(
			"dual projection type",
			"none",
			dealii::Patterns::Anything(),
			"determines which projection should be used for div-correction"
		);

		declare_entry(
			"primal order",
			"low",
			dealii::Patterns::Anything()
		);

		declare_entry(
			"dual order",
			"high",
			dealii::Patterns::Anything()
		);

		// LOW order problem
		// fe convection (low)
		declare_entry(
			"low convection space type",
			"cG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low convection space type support points",
			"canonical",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low convection p",
			"2",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"low convection time type",
			"dG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low convection time type support points",
			"Gauss",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low convection r",
			"1",
			dealii::Patterns::Integer()
		);
		
		// fe pressure (low)
		declare_entry(
			"low pressure space type",
			"cG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low pressure space type support points",
			"canonical",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low pressure p",
			"1",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"low pressure time type",
			"dG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low pressure time type support points",
			"Gauss",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"low pressure r",
			"1",
			dealii::Patterns::Integer()
		);
		
		// HIGH order problem
		// fe convection (high)
		declare_entry(
			"high convection space type",
			"cG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high convection space type support points",
			"canonical",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high convection p",
			"4",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"high convection time type",
			"dG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high convection time type support points",
			"Gauss",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high convection r",
			"2",
			dealii::Patterns::Integer()
		);
		
		// fe pressure (high)
		declare_entry(
			"high pressure space type",
			"cG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high pressure space type support points",
			"canonical",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high pressure p",
			"2",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"high pressure time type",
			"dG",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high pressure time type support points",
			"Gauss",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"high pressure r",
			"2",
			dealii::Patterns::Integer()
		);
	}
	leave_subsection();
	
	enter_subsection("Mesh Specification"); {
		declare_entry(
			"use mesh input file",
			"false",
			dealii::Patterns::Bool(),
			"determines whether to use an input file or a deal.II GridGenerator"
		);
		
		declare_entry(
			"mesh input filename",
			"./input/.empty",
			dealii::Patterns::Anything(),
			"filename of the mesh which can be read in with dealii::GridIn"
		);
		
		declare_entry(
			"TriaGenerator",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"TriaGenerator Options",
			"invalid",
			dealii::Patterns::Anything()
		);
	}
	leave_subsection();
	
	enter_subsection("Fluid Mesh Specification"); {
		declare_entry(
			"Grid Class",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"Grid Class Options",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"global refinement",
			"0",
			dealii::Patterns::Integer(),
			"Global refinements of the intial mesh"
		);
	}
	leave_subsection();
	
	enter_subsection("Fluid Time Integration"); {
		declare_entry(
			"initial time",
			"0.",
			dealii::Patterns::Double(),
			"initial time t0"
		);
		
		declare_entry(
			"final time",
			"0.",
			dealii::Patterns::Double(),
			"final time T"
		);
		
		declare_entry(
			"time step size",
			"1e-2",
			dealii::Patterns::Double(),
			"initial time step size"
		);
		
		declare_entry(
			"global refinement",
			"0",
			dealii::Patterns::Integer(),
			"Global refinements of the intial time mesh"
		);

		declare_entry(
			"maximum number of intervals per slab",
			"1",
			dealii::Patterns::Integer(),
			"Defines when a slab should be split into two slabs"
		);
	}
	leave_subsection();
	enter_subsection("Reference Values");
		declare_entry(
			"mean drag stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean lift stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean pressure difference stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean vorticity stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean drag navier-stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean lift navier-stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean pressure difference navier-stokes",
			"0.",
			dealii::Patterns::Double()
		);
		declare_entry(
			"mean vorticity navier-stokes",
			"0.",
			dealii::Patterns::Double()
		);
	leave_subsection();
	enter_subsection("Newton");{
		declare_entry(
			"max steps",
			"60",
			dealii::Patterns::Integer()
		);

        declare_entry(
                "lower bound",
                "1e-7",
                dealii::Patterns::Double()
        );

        declare_entry(
                "rebuild parameter",
                "0.1",
                dealii::Patterns::Double()
        );

        declare_entry(
                "line search steps",
                "10",
                dealii::Patterns::Integer()
        );

        declare_entry(
                "line search damping",
                "0.6",
                dealii::Patterns::Double()
        );
	}
	leave_subsection();

	
	enter_subsection("DWR"); {
		declare_entry(
			"calculate functionals",
			"",
			dealii::Patterns::Anything()
		);


		declare_entry(
			"goal type",
			"mean_drag",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"goal weight function",
			"ConstantFunction",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"goal weight options",
			"1.0",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"loops",
			"2",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"solver control in use",
			"false",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"solver control reduction mode",
			"true",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"solver control max iterations",
			"5",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"solver control tolerance",
			"1e-10",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"solver control reduction",
			"1e-8",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen spacetime strategy",
			"global",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"refine and coarsen space strategy",
			"global",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"refine and coarsen space top fraction1",
			"1.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space top fraction2",
			"0.5",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space bottom fraction",
			"0.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space max growth factor n_active_cells",
			"4",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"refine and coarsen space Schwegler theta1",
			"1.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space Schwegler theta2",
			"0.0",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen space riwi alpha",
			"1.1",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"refine and coarsen time strategy",
			"global",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"refine and coarsen time top fraction",
			"1.0",
			dealii::Patterns::Double()
		);

		declare_entry(
			"replace linearization points",
			"false",
			dealii::Patterns::Bool()
		);

		declare_entry(
			"replace weights",
			"false",
			dealii::Patterns::Bool()
		);
	}
	leave_subsection();
	
	enter_subsection("Parameter Specification"); {
		declare_entry(
			"viscosity function",
			"ConstantFunction",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"viscosity options",
			"1.0",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"fluid force function",
			"ZeroTensorFunction",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"fluid force options",
			"",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"fluid force assembler quadrature auto mode",
			"true",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"fluid force assembler quadrature points",
			"0",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"convection initial value function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection initial value options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"pressure initial value function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"pressure initial value options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"convection dirichlet boundary function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection dirichlet boundary options",
			"",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection dirichlet assembler quadrature auto mode",
			"false",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"convection dirichlet assembler quadrature points",
			"0",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"convection neumann boundary function",
			"ZeroTensorFunction",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection neumann boundary options",
			"",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection neumann assembler quadrature auto mode",
			"false",
			dealii::Patterns::Bool()
		);
		
		declare_entry(
			"convection neumann assembler quadrature points",
			"0",
			dealii::Patterns::Integer()
		);
		
		
		declare_entry(
			"convection exact solution function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"convection exact solution options",
			"",
			dealii::Patterns::Anything()
		);
		
		
		declare_entry(
			"pressure exact solution function",
			"invalid",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"pressure exact solution options",
			"",
			dealii::Patterns::Anything()
		);
	}
	leave_subsection();
	
	enter_subsection("Output Quantities"); {
		declare_entry(
			"primal data output dwr loop",
			"all",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"primal data output trigger type",
			"fixed",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"primal data output trigger time",
			"-1.",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"primal data output patches auto mode",
			"true",
			dealii::Patterns::Bool(),
			"primal data output patches auto mode => using p data output patches"
		);
		
		declare_entry(
			"primal data output patches",
			"1",
			dealii::Patterns::Integer()
		);
		
		declare_entry(
			"dual data output dwr loop",
			"all",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dual data output trigger type",
			"fixed",
			dealii::Patterns::Anything()
		);
		
		declare_entry(
			"dual data output trigger time",
			"-1.",
			dealii::Patterns::Double()
		);
		
		declare_entry(
			"dual data output patches auto mode",
			"true",
			dealii::Patterns::Bool(),
			"dual data output patches auto mode => using q data output patches"
		);
		
		declare_entry(
			"dual data output patches",
			"1",
			dealii::Patterns::Integer()
		);

		declare_entry(
			"error estimator data output dwr loop",
			"all",
			dealii::Patterns::Anything()
		);

		declare_entry(
			"error estimator data output trigger type",
			"fixed",
			dealii::Patterns::Anything()
		);

		declare_entry(
			"error estimator data output trigger time",
			"-1.",
			dealii::Patterns::Double()
		);

		declare_entry(
			"error estimator data output patches auto mode",
			"true",
			dealii::Patterns::Bool(),
			"error estimator data output patches auto mode => using 1 data output patch"
		);

		declare_entry(
			"error estimator data output patches",
			"1",
			dealii::Patterns::Integer()
		);
	}
	leave_subsection();
}

} // namespace
