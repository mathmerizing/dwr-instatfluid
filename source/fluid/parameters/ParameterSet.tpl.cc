/**
 * @file   ParameterSet.cc
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-17, Added Newton parameters, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-06, stokes, UK
 * @date 2018-09-14, unified to other DTM programs, UK
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, UK
 * @date 2017-09-11, UK
 * @date 2017-02-07, UK
 *
 * @brief Keeps all parsed input parameters in a struct.
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher                                    */
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
#include <fluid/parameters/ParameterSet.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>

// C++ includes
#include <limits>

namespace fluid {

ParameterSet::
ParameterSet(
	std::shared_ptr< dealii::ParameterHandler > handler) {
	Assert(handler.use_count(), dealii::ExcNotInitialized());
	
	dim = static_cast<unsigned int> (handler->get_integer("dim"));
	
	handler->enter_subsection("Fluid Problem Specification"); {
		fe.symmetric_stress = handler->get_bool(
			"symmetric stress"
		);

		fe.primal.convection.space_type = handler->get(
			"primal convection space type"
		);
		fe.primal.convection.space_type_support_points = handler->get(
			"primal convection space type support points"
		);
		fe.primal.convection.p = static_cast<unsigned int> (
			handler->get_integer("primal convection p")
		);
		
		fe.primal.convection.time_type = handler->get(
			"primal convection time type"
		);
		fe.primal.convection.time_type_support_points = handler->get(
			"primal convection time type support points"
		);
		fe.primal.convection.r = static_cast<unsigned int> (
			handler->get_integer("primal convection r")
		);
		
		
		fe.primal.pressure.space_type = handler->get(
			"primal pressure space type"
		);
		fe.primal.pressure.space_type_support_points = handler->get(
			"primal pressure space type support points"
		);
		fe.primal.pressure.p = static_cast<unsigned int> (
			handler->get_integer("primal pressure p")
		);
		
		fe.primal.pressure.time_type = handler->get(
			"primal pressure time type"
		);
		fe.primal.pressure.time_type_support_points = handler->get(
			"primal pressure time type support points"
		);
		fe.primal.pressure.r = static_cast<unsigned int> (
			handler->get_integer("primal pressure r")
		);
		
		
		// dual
		fe.dual.convection.space_type = handler->get(
			"dual convection space type"
		);
		fe.dual.convection.space_type_support_points = handler->get(
			"dual convection space type support points"
		);
		fe.dual.convection.p = static_cast<unsigned int> (
			handler->get_integer("dual convection p")
		);
		
		fe.dual.convection.time_type = handler->get(
			"dual convection time type"
		);
		fe.dual.convection.time_type_support_points = handler->get(
			"dual convection time type support points"
		);
		fe.dual.convection.r = static_cast<unsigned int> (
			handler->get_integer("dual convection r")
		);
		
		
		fe.dual.pressure.space_type = handler->get(
			"dual pressure space type"
		);
		fe.dual.pressure.space_type_support_points = handler->get(
			"dual pressure space type support points"
		);
		fe.dual.pressure.p = static_cast<unsigned int> (
			handler->get_integer("dual pressure p")
		);
		
		fe.dual.pressure.time_type = handler->get(
			"dual pressure time type"
		);
		fe.dual.pressure.time_type_support_points = handler->get(
			"dual pressure time type support points"
		);
		fe.dual.pressure.r = static_cast<unsigned int> (
			handler->get_integer("dual pressure r")
		);
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Mesh Specification"); {
		use_mesh_input_file = handler->get_bool("use mesh input file");
		mesh_input_filename = handler->get("mesh input filename");
		
		TriaGenerator = handler->get("TriaGenerator");
		TriaGenerator_Options = handler->get("TriaGenerator Options");
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Fluid Mesh Specification"); {
		space.boundary.fluid.Grid_Class = handler->get("Grid Class");
		space.boundary.fluid.Grid_Class_Options = handler->get(
			"Grid Class Options"
		);
		
		space.mesh.fluid.global_refinement = static_cast<unsigned int> (
			handler->get_integer("global refinement")
		);
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Fluid Time Integration"); {
		time.fluid.t0 = handler->get_double("initial time");
		time.fluid.T = handler->get_double("final time");
		time.fluid.tau_n = handler->get_double("time step size");
		time.fluid.initial_time_tria_refinement = handler->get_double(
			"global refinement"
		);
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Newton");{
		newton.max_steps = handler->get_integer("max steps");
		newton.lower_bound = handler->get_double("lower bound");
		newton.rebuild = handler->get_double("rebuild parameter");
		newton.line_search_steps = handler->get_integer("line search steps");
		newton.line_search_damping = handler->get_double("line search damping");
	}
	handler->leave_subsection();

	handler->enter_subsection("DWR"); {
		dwr.goal.type = handler->get("goal type");
		dwr.goal.weight_function = handler->get("goal weight function");
		dwr.goal.weight_options = handler->get("goal weight options");
		
		dwr.solver_control.in_use = handler->get_bool("solver control in use");
		if (dwr.solver_control.in_use) {
			dwr.solver_control.reduction_mode = handler->get_bool(
				"solver control reduction mode"
			);
			
			dwr.solver_control.max_iterations = static_cast<unsigned int> (
				handler->get_integer("solver control max iterations")
			);
			dwr.loops = dwr.solver_control.max_iterations;
			
			dwr.solver_control.tolerance = handler->get_double(
				"solver control tolerance"
			);
			
			dwr.solver_control.reduction = handler->get_double(
				"solver control reduction"
			);
		}
		else {
			dwr.loops = static_cast<unsigned int> (handler->get_integer("loops"));
		}
		
		
		dwr.refine_and_coarsen.space.strategy = handler->get(
			"refine and coarsen space strategy"
		);
		
		dwr.refine_and_coarsen.space.top_fraction1 = handler->get_double(
			"refine and coarsen space top fraction1"
		);
		
		dwr.refine_and_coarsen.space.top_fraction2 = handler->get_double(
			"refine and coarsen space top fraction2"
		);
		
		dwr.refine_and_coarsen.space.bottom_fraction = handler->get_double(
			"refine and coarsen space bottom fraction"
		);
		
		dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells =
			static_cast<unsigned int> (handler->get_integer(
			"refine and coarsen space max growth factor n_active_cells"
		));
		
		dwr.refine_and_coarsen.space.theta1 = handler->get_double(
			"refine and coarsen space Schwegler theta1"
		);
		
		dwr.refine_and_coarsen.space.theta2 = handler->get_double(
			"refine and coarsen space Schwegler theta2"
		);
		
		
		dwr.refine_and_coarsen.time.strategy = handler->get(
			"refine and coarsen time strategy"
		);
		
		dwr.refine_and_coarsen.time.top_fraction = handler->get_double(
			"refine and coarsen time top fraction"
		);
	}
	handler->leave_subsection();
	
	handler->enter_subsection("Parameter Specification"); {
		viscosity_function = handler->get(
			"viscosity function"
		);
		
		viscosity_options = handler->get(
			"viscosity options"
		);
		
		// fluid force
		fluid.force_function = handler->get(
			"fluid force function"
		);
		
		fluid.force_options = handler->get(
			"fluid force options"
		);
		
		fluid.force_assembler_n_quadrature_points = static_cast<unsigned int> (
			handler->get_integer(
				"fluid force assembler quadrature points"
			)
		);
		if (handler->get_bool("fluid force assembler quadrature auto mode")) {
			fluid.force_assembler_n_quadrature_points +=
				fe.primal.convection.p + 1;
		}
		
		// convection initial value functions
		convection.initial_value_function = handler->get(
			"convection initial value function"
		);
		
		convection.initial_value_options = handler->get(
			"convection initial value options"
		);
		
		// convection functions
		convection.dirichlet_boundary_function = handler->get(
			"convection dirichlet boundary function"
		);
		
		convection.dirichlet_boundary_options = handler->get(
			"convection dirichlet boundary options"
		);
		
		convection.dirichlet_assembler_n_quadrature_points =
			static_cast<unsigned int> (handler->get_integer(
				"convection dirichlet assembler quadrature points"
			)
		);
		if (handler->get_bool("convection dirichlet assembler quadrature auto mode")) {
			convection.dirichlet_assembler_n_quadrature_points +=
				fe.primal.convection.p + 1;
		}
		
		
		convection.neumann_boundary_function = handler->get(
			"convection neumann boundary function"
		);
		
		convection.neumann_boundary_options = handler->get(
			"convection neumann boundary options"
		);
		
		convection.neumann_assembler_n_quadrature_points =
			static_cast<unsigned int> (handler->get_integer(
				"convection neumann assembler quadrature points"
			)
		);
		if (handler->get_bool("convection neumann assembler quadrature auto mode")) {
			convection.neumann_assembler_n_quadrature_points +=
				fe.primal.convection.p + 1;
		}
		
		
		convection.exact_solution_function = handler->get(
			"convection exact solution function"
		);
		
		convection.exact_solution_options = handler->get(
			"convection exact solution options"
		);
		
		// pressure initial value functions
		pressure.initial_value_function = handler->get(
			"pressure initial value function"
		);
		
		pressure.initial_value_options = handler->get(
			"pressure initial value options"
		);
		
		// pressure functions
		pressure.exact_solution_function = handler->get(
			"pressure exact solution function"
		);
		
		pressure.exact_solution_options = handler->get(
			"pressure exact solution options"
		);
	}
	handler->leave_subsection();
	
	
	handler->enter_subsection("Output Quantities"); {
		data_output.primal.dwr_loop = handler->get("primal data output dwr loop");
		
		data_output.primal.trigger_type = handler->get("primal data output trigger type");
		data_output.primal.trigger = handler->get_double("primal data output trigger time");
		
		if (handler->get_bool("primal data output patches auto mode")) {
			data_output.primal.patches =
				std::max(
					std::max(fe.primal.convection.p, fe.primal.pressure.p),
					static_cast<unsigned int> (1)
				);
		}
		else {
			data_output.primal.patches = static_cast<unsigned int> (
				handler->get_integer("primal data output patches")
			);
		}
		
		data_output.dual.dwr_loop = handler->get("dual data output dwr loop");
		
		data_output.dual.trigger_type = handler->get("dual data output trigger type");
		data_output.dual.trigger = handler->get_double("dual data output trigger time");
		
		if (handler->get_bool("dual data output patches auto mode")) {
			data_output.dual.patches =
				std::max(
					std::max(fe.dual.convection.p, fe.dual.pressure.p),
					static_cast<unsigned int> (1)
				);
		}
		else {
			data_output.dual.patches = static_cast<unsigned int> (
				handler->get_integer("dual data output patches")
			);
		}
	}
	handler->leave_subsection();
}

} // namespace
