/**
 * @file Fluid.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-04-25, high/low order, JR
 * @Date 2022-01-19, Newtonsolver working for Stokes, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-22, ST hanging nodes, UK
 * @date 2021-11-09, dG(r) and multiple time dofs, UK
 * @date 2021-11-05, dynamics for stokes, JR, UK
 * @date 2021-10-30, force assembler, MPB, UK
 * @date 2019-11-07, space-time, quasi-stationary Stokes, UK
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
#include <fluid/Fluid.tpl.hh>

#include <fluid/QRightBox.tpl.hh>

#include <fluid/grid/Grid_Selector.tpl.hh>

#include <fluid/Viscosity/Viscosity_Selector.tpl.hh>
#include <fluid/Force/Force_Selector.tpl.hh>
#include <fluid/ConvectionDirichletBoundary/ConvectionDirichletBoundary_Selector.tpl.hh>

#include <fluid/types/boundary_id.hh>

// primal
#include <fluid/assembler/ST_FluidAssembly.tpl.hh>

// #include <fluid/assembler/ST_ForceAssembly.tpl.hh>
// template <int dim>
// using ForceAssembler = force::spacetime::Operator::Assembler<dim>;

#include <fluid/assembler/ST_InitialValueAssembly.tpl.hh>
template <int dim>
using IVAssembler = initialvalue::spacetime::Operator::Assembler<dim>;

#include <fluid/assembler/ST_FluidRHSAssembly.tpl.hh>
template <int dim>
using NewtonRHSAssembler = fluidrhs::spacetime::Operator::Assembler<dim>;

// dual
#include <fluid/assembler/ST_Dual_FluidAssembly.tpl.hh>

#include <fluid/assembler/ST_Dual_MeanDragAssembly.tpl.hh>
template <int dim>
using Je_MeanDrag_Assembler = goal::spacetime::Operator::Assembler<dim>;

#include <fluid/assembler/ST_Dual_FinalValueAssembly.tpl.hh>
template <int dim>
using FVDualAssembler = finalvalue::spacetime::dual::Operator::Assembler<dim>;

// divergence free projection
#include <fluid/assembler/ST_DivFreeProjectionAssembly.tpl.hh>
template <int dim>
using ProjectionAssembler = projection::spacetime::Operator::Assembler<dim>;
#include <fluid/assembler/ST_DivFreeProjectionRHSAssembly.tpl.hh>
template <int dim>
using ProjectionRHSAssembler = projectionrhs::spacetime::Operator::Assembler<dim>;

// dual divergence free projection
//#include <fluid/assembler/ST_Dual_DivFreeProjectionAssembly.tpl.hh>
//template <int dim>
//using DualProjectionAssembler = projection::spacetime::dual::Operator::Assembler<dim>;
//#include <fluid/assembler/ST_Dual_DivFreeProjectionRHSAssembly.tpl.hh>
//template <int dim>
//using DualProjectionRHSAssembler = projectionrhs::spacetime::dual::Operator::Assembler<dim>;


// DEAL.II includes
#include <deal.II/base/types.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

namespace fluid {

template<int dim>
void
Fluid<dim>::
set_input_parameters(
	std::shared_ptr< dealii::ParameterHandler > parameter_handler) {
	Assert(parameter_handler.use_count(), dealii::ExcNotInitialized());
	
	parameter_set = std::make_shared< fluid::ParameterSet > (
		parameter_handler
	);
}


template<int dim>
void
Fluid<dim>::
run() {
	// check
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	// check problem we want to solve
	if ( parameter_set->problem.compare("Stokes") == 0)
	{
			DTM::pout
					<< "solving the linear Stokes equations"
					<< std::endl;
	}
	else if ( parameter_set->problem.compare("Navier-Stokes") == 0 )
	{
			DTM::pout
					<< "solving the quasilinear Navier-Stokes equations\n"
					<< std::endl;
	}
	else{
			AssertThrow(
					false,
					dealii::ExcMessage(
							"unknown problem please choose Stokes or Navier-Stokes"
					)
			);
	}
	// check whether we are using the symmetric stress tensor
	DTM::pout << "symmetric stress: " << (parameter_set->fe.symmetric_stress ? "true" : "false") << std::endl;

	// check primal space discretisation
	if ((parameter_set->fe.primal.convection.space_type.compare("cG") == 0) &&
		(parameter_set->fe.primal.pressure.space_type.compare("cG") == 0) ) {
		DTM::pout
			<< "primal space discretisation convection-pressure:"
			<< std::endl
			<< "\t[ "
			// convection
			<< "cG("
			<< parameter_set->fe.primal.convection.p
			<< ")-Q_"
			<< parameter_set->fe.primal.convection.space_type_support_points
			<< ", "
			// pressure
			<< "cG("
			<< parameter_set->fe.primal.pressure.p
			<< ")-Q_"
			<< parameter_set->fe.primal.pressure.space_type_support_points
			<< " ]^T"
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"primal space discretisation unknown"
			)
		);
	}

	// check primal time discretisation
	if ((parameter_set->fe.primal.convection.time_type.compare("dG") == 0) &&
		(parameter_set->fe.primal.pressure.time_type.compare("dG") == 0) ) {
		DTM::pout
			<< "primal time discretisation convection-pressure:"
			<< std::endl
			<< "\t[ "
			// convection
			<< "dG("
			<< parameter_set->fe.primal.convection.r
			<< ")-Q_"
			<< parameter_set->fe.primal.convection.time_type_support_points
			<< ", "
			// pressure
			<< "dG("
			<< parameter_set->fe.primal.pressure.r
			<< ")-Q_"
			<< parameter_set->fe.primal.pressure.time_type_support_points
			<< " ]^T"
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"primal time discretisation unknown"
			)
		);
	}
	
	// check dual space discretisation
	if ((parameter_set->fe.dual.convection.space_type.compare("cG") == 0) &&
		(parameter_set->fe.dual.pressure.space_type.compare("cG") == 0) ) {
		DTM::pout
			<< "dual space discretisation convection-pressure:"
			<< std::endl
			<< "\t[ "
			// convection
			<< "cG("
			<< parameter_set->fe.dual.convection.p
			<< ")-Q_"
			<< parameter_set->fe.dual.convection.space_type_support_points
			<< ", "
			// pressure
			<< "cG("
			<< parameter_set->fe.dual.pressure.p
			<< ")-Q_"
			<< parameter_set->fe.dual.pressure.space_type_support_points
			<< " ]^T"
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"dual space discretisation unknown"
			)
		);
	}

	// check dual time discretisation
	if ((parameter_set->fe.dual.convection.time_type.compare("dG") == 0) &&
		(parameter_set->fe.dual.pressure.time_type.compare("dG") == 0) ) {
		DTM::pout
			<< "dual time discretisation convection-pressure:"
			<< std::endl
			<< "\t[ "
			// convection
			<< "dG("
			<< parameter_set->fe.dual.convection.r
			<< ")-Q_"
			<< parameter_set->fe.dual.convection.time_type_support_points
			<< ", "
			// pressure
			<< "dG("
			<< parameter_set->fe.dual.pressure.r
			<< ")-Q_"
			<< parameter_set->fe.dual.pressure.time_type_support_points
			<< " ]^T"
			<< std::endl;
	}
	else {
		AssertThrow(
			false,
			dealii::ExcMessage(
				"dual time discretisation unknown"
			)
		);
	}

	// determine setw value for dwr loop number of data output filename
	setw_value_dwr_loops = static_cast<unsigned int>(
		std::floor(std::log10(parameter_set->dwr.loops))+1
	);
	
	init_functions();
	init_newton_parameters();
	init_grid();
	
	////////////////////////////////////////////
	// set primal and dual to low or high
	//
	auto slab(grid->slabs.begin());
	auto ends(grid->slabs.end());

	for (; slab != ends; ++slab) {
		// primal = low / high
		if ( !parameter_set->fe.primal_order.compare("low") )
		{
			slab->space.primal.fe_info = slab->space.low.fe_info;
			slab->time.primal.fe_info = slab->time.low.fe_info;
		}
		else if ( !parameter_set->fe.primal_order.compare("high") )
		{
			slab->space.primal.fe_info = slab->space.high.fe_info;
			slab->time.primal.fe_info = slab->time.high.fe_info;
		}
		else
		{
			AssertThrow(false, dealii::ExcMessage("primal_order needs to be 'low' or 'high'."));
		}

		// dual = low / high / high-time
		if ( !parameter_set->fe.dual_order.compare("low") )
		{
			slab->space.dual.fe_info = slab->space.low.fe_info;
			slab->time.dual.fe_info = slab->time.low.fe_info;
		}
		else if ( !parameter_set->fe.dual_order.compare("high") )
		{
			slab->space.dual.fe_info = slab->space.high.fe_info;
			slab->time.dual.fe_info = slab->time.high.fe_info;
		}
		else if ( !parameter_set->fe.dual_order.compare("high-time") )
		{
			slab->space.dual.fe_info = slab->space.low.fe_info;
			slab->time.dual.fe_info = slab->time.high.fe_info;
		}
		else
		{
			AssertThrow(false, dealii::ExcMessage("dual_order needs to be 'low' or 'high' or 'high-time'."));
		}
	} // end for-loop slab

	////////////////////////////////////////////////////////////////////////////
	// adaptivity loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl;
	
	unsigned int dwr_loop{1};
	unsigned int max_dwr_loop{parameter_set->dwr.solver_control.max_iterations};
	do {
		if (dwr_loop > 1) {
			if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("global_space"))
				grid->refine_global(1, 0); // refine grid in space
			else if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("global_time"))
			{
				// splitting each slab into two slabs
				auto slab{grid->slabs.begin()};
				auto ends{grid->slabs.end()};
				for (; slab != ends; ++slab)
					grid->refine_slab_in_time(slab);
			}
			else if	(!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("global"))
			{
				// refine grid in space
				grid->refine_global(1, 0);

				// splitting each slab into two slabs
				auto slab{grid->slabs.begin()};
				auto ends{grid->slabs.end()};
				for (; slab != ends; ++slab)
					grid->refine_slab_in_time(slab);
			}
			else if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive"))
				refine_and_coarsen_space_time_grid(dwr_loop-1); // do adaptive space-time mesh refinements and coarsenings
			else
				// invalid refinement strategy
				AssertThrow(false, dealii::ExcNotImplemented());

			grid->set_manifolds();
		}
		
		DTM::pout
			<< "***************************************************************"
			<< "*****************" << std::endl
			<< "adaptivity loop = " << dwr_loop << std::endl;
		
		grid->set_boundary_indicators();
		
		// primal problem:
		primal_reinit_storage();
		primal_init_data_output();
		primal_do_forward_TMS(dwr_loop, false);

		// dual problem
		dual_reinit_storage();
		dual_init_data_output();
		{
			// error indicators
			eta_reinit_storage();
			eta_init_data_output();
		}
		dual_do_backward_TMS(dwr_loop, false);
		{
			dual_sort_xdmf_by_time(dwr_loop);
			eta_sort_xdmf_by_time(dwr_loop);
		}

		// error estimation
		compute_effectivity_index();

		// compute the number of primal and dual space-time dofs
		unsigned long int n_primal_st_dofs = 0;
		unsigned long int n_dual_st_dofs   = 0;

		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (; slab != ends; ++slab)
		{
			n_primal_st_dofs += slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs();
			n_dual_st_dofs += slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs();
		}

		DTM::pout << "\n#DoFs(primal; Space-Time) = " << n_primal_st_dofs
				  << "\n#DoFs(dual; Space-Time)   = " << n_dual_st_dofs
				  << std::endl;

//		if (estimated_error < TOL_DWR)
//			break;

		++dwr_loop;
	} while (dwr_loop <= max_dwr_loop);

	// TODO: create convergence table
}


////////////////////////////////////////////////////////////////////////////////
// protected member functions (internal use only)
//

template<int dim>
void
Fluid<dim>::
init_grid() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	////////////////////////////////////////////////////////////////////////////
	// init grid from input parameter file spec.
	//
	{
		fluid::grid::Selector<dim> selector;
		selector.create_grid(
			parameter_set,
			grid
		);
		
		Assert(grid.use_count(), dealii::ExcNotInitialized());
	}
	
	////////////////////////////////////////////////////////////////////////////
	// initialize slabs of grid
	//
	
	Assert(parameter_set->fe.primal.convection.p, dealii::ExcInvalidState());
	Assert(parameter_set->fe.primal.pressure.p, dealii::ExcInvalidState());
	
	Assert((parameter_set->time.fluid.t0 >= 0.), dealii::ExcInvalidState());
	Assert(
		(parameter_set->time.fluid.t0 < parameter_set->time.fluid.T),
		dealii::ExcInvalidState()
	);
	Assert((parameter_set->time.fluid.tau_n > 0.), dealii::ExcInvalidState());
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	grid->initialize_slabs();
	
	grid->generate();
	grid->set_manifolds();
	
	grid->refine_global(
		parameter_set->space.mesh.fluid.global_refinement,
		parameter_set->time.fluid.initial_time_tria_refinement
	);
	
	DTM::pout
		<< "grid: number of slabs = " << grid->slabs.size()
		<< std::endl;
}


template<int dim>
void
Fluid<dim>::
init_functions() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	// viscosity function:
	{
		fluid::viscosity::Selector<dim> selector;
		selector.create_function(
			parameter_set->viscosity_function,
			parameter_set->viscosity_options,
			function.viscosity
		);
		
		Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
	}
	
	// fluid force function:
	{
		fluid::force::Selector<dim> selector;
		selector.create_function(
			parameter_set->fluid.force_function,
			parameter_set->fluid.force_options,
			function.fluid.force
		);
		
		Assert(function.fluid.force.use_count(), dealii::ExcNotInitialized());
	}
	
	// convection dirichlet boundary function:
	{
		convection::dirichlet::Selector<dim> selector;
		selector.create_function(
			parameter_set->convection.dirichlet_boundary_function,
			parameter_set->convection.dirichlet_boundary_options,
			function.convection.dirichlet
		);
		
		Assert(function.convection.dirichlet.use_count(), dealii::ExcNotInitialized());
	}
}


template<int dim>
void
Fluid<dim>::
init_newton_parameters() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	newton.max_steps = parameter_set->newton.max_steps;
	newton.lower_bound = parameter_set->newton.lower_bound;
	newton.rebuild = parameter_set->newton.rebuild;
	newton.line_search_steps = parameter_set->newton.line_search_steps;
	newton.line_search_damping = parameter_set->newton.line_search_damping;


	DTM::pout << "\n* Newton parameters found:"
			  << "\n\tmaximum # of steps = " << newton.max_steps
			  << "\n\twanted tolerance = " << newton.lower_bound
			  << "\n\trebuild matrix if residual quotient > " << newton.rebuild
			  << "\n\tLine Search parameters:"
			  << "\n\t\tmaximum # of steps = " << newton.line_search_steps
			  << "\n\t\tdamping factor = " << newton.line_search_damping
			  << std::endl << std::endl;
}
////////////////////////////////////////////////////////////////////////////////
// primal problem
//

template<int dim>
void
Fluid<dim>::
primal_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data:
	// NOTE: * primal space: time dG(r) method (all dofs in 1 vector)
	//       * primal solution dof vectors: u
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	
	primal.storage.u =
		std::make_shared< DTM::types::storage_data_vectors<1> > ();
	
	primal.storage.u->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);

	primal.storage.um =
		std::make_shared< DTM::types::storage_data_vectors<1> > ();

	primal.storage.um->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);
}

template<int dim>
void
Fluid<dim>::
primal_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &x,
	const typename DTM::types::storage_data_vectors<1>::iterator &xm
) {
	for (unsigned int j{0}; j < x->x.size(); ++j) {
		// x
		x->x[j] = std::make_shared< dealii::Vector<double> > ();

		Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

		Assert(
			slab->space.primal.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		Assert(
			slab->time.primal.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		x->x[j]->reinit(
			slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
		);

		// xm
		xm->x[j] = std::make_shared< dealii::Vector<double> > ();

		xm->x[j]->reinit(
			slab->space.primal.fe_info->dof->n_dofs()
		);
	}
}

template<int dim>
void
Fluid<dim>::
primal_assemble_system(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double > > u
) {
	// ASSEMBLY SPACE-TIME OPERATOR MATRIX /////////////////////////////////////
	Assert(
		slab->spacetime.primal.sp.use_count(),
		dealii::ExcNotInitialized()
	);
	
	primal.L = std::make_shared< dealii::SparseMatrix<double> > ();
	primal.L->reinit(*slab->spacetime.primal.sp);
	*primal.L = 0;
	
	{
		fluid::spacetime::Operator::
		Assembler<dim> assembler;

		assembler.set_symmetric_stress(parameter_set->fe.symmetric_stress);
		
		Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
		assembler.set_functions(
			function.viscosity
		);

		assembler.set_time_quad_type((
				!parameter_set->fe.primal_order.compare("low") ?
						parameter_set->fe.low.convection.time_type_support_points :
						parameter_set->fe.high.convection.time_type_support_points
		));

		Assert(primal.L.use_count(), dealii::ExcNotInitialized());
		assembler.assemble(
			primal.L,
			slab,
			u,
			(parameter_set->problem.compare("Navier-Stokes") == 0)
		);
		
	}

	// FOR DEBUGGING PRINT MATRIX:
	{
		int slab_number = 0;
		auto tmp_slab = grid->slabs.begin();
		while (slab != tmp_slab)
		{
			slab_number++;
			tmp_slab++;
		}

		std::ostringstream filename;
		filename << "primal_matrix_" << std::setw(3) << std::setfill('0') << slab_number << ".txt";
		std::ofstream out(filename.str().c_str(), std::ios_base::out);
		primal.L->print(out);
		out.close();
	}

}


template<int dim>
void
Fluid<dim>::
primal_assemble_const_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &um,
	std::map<dealii::types::global_dof_index, double> &boundary_values
) {
	// ASSEMBLY SPACE-TIME OPERATOR: InitialValue VECTOR ///////////////////////
	primal.Mum = std::make_shared< dealii::Vector<double> > ();
	
	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	
	if (parameter_set->fe.divfree_projection != "none")
	{
		bool use_gradient_projection;
		if (parameter_set->fe.divfree_projection == "L2")
		{
			use_gradient_projection = false;
		}
		else if (parameter_set->fe.divfree_projection == "H1")
		{
			use_gradient_projection = true;
		}
		else
		{
			AssertThrow(false, dealii::ExcMessage("divfree_projection needs to be 'none', 'L2' or 'H1'."));
		}

		primal.projection_matrix = std::make_shared< dealii::SparseMatrix<double> > ();
		primal.projection_matrix->reinit(*slab->space.primal.sp_L);
		*primal.projection_matrix = 0;

		primal.projection_rhs = std::make_shared< dealii::Vector<double> > ();
		primal.projection_rhs->reinit(slab->space.primal.fe_info->dof->n_dofs());
		*primal.projection_rhs = 0;

		// use divergence-free projection on primal.um
		{
			DTM::pout << "dwr-instatfluid: divergence free projection (gradient=" << (use_gradient_projection ? "true" : "false") << ")...";

			// 1. assemble projection matrix
			ProjectionAssembler<dim> matrix_assembler;
			matrix_assembler.set_gradient_projection(use_gradient_projection); // TRUE: use H^1_0 projection; FALSE: use L^2 projection

			Assert(primal.projection_matrix.use_count(), dealii::ExcNotInitialized());
			matrix_assembler.assemble(
				primal.projection_matrix,
				slab
			);

			// 2. assemble projection rhs
			ProjectionRHSAssembler<dim> rhs_assembler;
			rhs_assembler.set_gradient_projection(use_gradient_projection); // TRUE: use H^1_0 projection; FALSE: use L^2 projection

			Assert(primal.um.use_count(), dealii::ExcNotInitialized());
			Assert(primal.projection_rhs.use_count(), dealii::ExcNotInitialized());
			rhs_assembler.assemble(
				primal.um,
				primal.projection_rhs,
				slab
			);

			// 3. apply boundary values and hanging nodes to the linear system of the projection
			// NOTE: primal_apply_bc(boundary_values, primal.projection_matrix, primal.um_projected, primal.projection_rhs);
			// The above line doesn't work, because boundary_values is space-time and projection_matrix is only space, hence these are not the correct BC

			{
				std::map<dealii::types::global_dof_index, double> boundary_values;

				auto dirichlet_function =
				std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
					*function.convection.dirichlet,
					0, (dim+1)
				);

				// get all boundary colours on the current triangulation
				std::unordered_set< dealii::types::boundary_id > boundary_colours;
				{
					auto cell = slab->space.tria->begin_active();
					auto endc = slab->space.tria->end();
					for ( ; cell != endc; ++cell) {
					if (cell->at_boundary()) {
						// loop over all faces
						for (unsigned int face_no{0};
							face_no < dealii::GeometryInfo<dim>::faces_per_cell;
							++face_no) {
						if (cell->face(face_no)->at_boundary()) {
							auto face = cell->face(face_no);
							boundary_colours.insert(
								static_cast< unsigned int > (face->boundary_id())
							);
						}}
					}}
				}

				// process all found boundary colours
				for (auto colour : boundary_colours) {
					//////////////////////////////////////
					// component wise convection dirichlet
					//

					auto component_mask_convection =
					std::make_shared< dealii::ComponentMask > (
						(dim+1), false
					);

					switch (dim) {
					case 3:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_convection_c3)) {
							component_mask_convection->set(2, true);
						}

						// NOTE: not to break switch here is indended
						[[fallthrough]];

					case 2:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_convection_c2)) {
							component_mask_convection->set(1, true);
						}

						// NOTE: not to break switch here is indended
						[[fallthrough]];

					case 1:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_convection_c1)) {
							component_mask_convection->set(0, true);
						}

						break;

					default:
						AssertThrow(false, dealii::ExcNotImplemented());
					}

					// create boundary_values as
					// std::map<dealii::types::global_dof_index, double>
					{
						dirichlet_function->set_time(
							slab->t_m
						);

						// pass through time to the actual function since it
						// doesn't work through the wrapper from a tensor function
						function.convection.dirichlet->set_time(
							slab->t_m
						);

						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.primal.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							*dirichlet_function,
							boundary_values_qt,
							*component_mask_convection
						);

						// boundary_values_qt -> boundary_values
						for (auto &el : boundary_values_qt) {
							boundary_values[el.first] = el.second;
						}
					}

					///////////////////////////////////////////////////////////
					// prescribed_no_slip: all convection components are homog.
					//

					for (unsigned int component{0}; component < (dim+1); ++component) {
						component_mask_convection->set(component, false);
					}

					switch (dim) {
					case 3:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_no_slip)) {
							component_mask_convection->set(2, true);
						}

						// NOTE: not to break switch here is indended
						[[fallthrough]];

					case 2:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_no_slip)) {
							component_mask_convection->set(1, true);
						}

						// NOTE: not to break switch here is indended
						[[fallthrough]];

					case 1:
						if (colour & static_cast< dealii::types::boundary_id > (
							fluid::types::space::boundary_id::prescribed_no_slip)) {
							component_mask_convection->set(0, true);
						}

						break;

					default:
						AssertThrow(false, dealii::ExcNotImplemented());
					}

					// create boundary_values as
					// std::map<dealii::types::global_dof_index, double>
					{
						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.primal.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							dealii::ZeroFunction<dim>(dim+1),
							boundary_values_qt,
							*component_mask_convection
						);

						// boundary_values_qt -> boundary_values
						for (auto &el : boundary_values_qt) {
							boundary_values[el.first] = el.second;
						}
					}
				}

				// apply boundary values
				if (boundary_values.size()) {
					////////////////////////////////////////////////////////////////////
					// prepare function header
					std::shared_ptr< dealii::SparseMatrix<double> > A = primal.projection_matrix;
					std::shared_ptr< dealii::Vector<double> > x = primal.um_projected;
					std::shared_ptr< dealii::Vector<double> > b = primal.projection_rhs;

					////////////////////////////////////////////////////////////////////
					// internal checks
					Assert(
						A->m(),
						dealii::ExcMessage(
							"empty operator matrix A (rows: A.m() == 0)"
						)
					);

					Assert(
						A->n(),
						dealii::ExcMessage(
							"empty operator matrix A (cols: A.n() == 0)"
						)
					);

					Assert(
						(A->n() == x->size()),
						dealii::ExcMessage(
							"Internal dimensions error: "
							"A->n() =/= x->size() of linear system A x = b"
						)
					);

					Assert(
						(A->m() == b->size()),
						dealii::ExcMessage(
							"Internal dimensions error: "
							"A->m() =/= b->size() of linear system A x = b"
						)
					);

					////////////////////////////////////////////////////////////////////
					// apply boundary values to operator matrix A, vectors x and b

					// preparation: eliminate constrained rows of operator A completely
					for (auto &boundary_value : boundary_values) {
						// on-the-fly check:
						// validity of the given boundary value constraints
						Assert(
							(boundary_value.first < A->m()),
							dealii::ExcMessage(
								"constraining of dof index (as boundary value) "
								"larger than (A.m()-1) is not valid"
							)
						);

						// eliminate constrained row of operator matrix completely
						auto el_in_row_i{A->begin(boundary_value.first)};
						auto end_el_in_row_i{A->end(boundary_value.first)};

						for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
							el_in_row_i->value() = double(0.);
						}
					}

					double diagonal_scaling_value{1.};

					////////////////////////////////////////////////////////////////////
					// apply boundary values:
					// to the linear system (A x = b)
					// without eliminating the corresponding column entries from
					// the operator matrix A
					//
					for (auto &boundary_value : boundary_values) {
						// set scaled identity operator
						A->set(
							boundary_value.first,  // i
							boundary_value.first,  // i
							diagonal_scaling_value // scaling factor or 1
						);

						// set constrained solution vector component (for iterative lss)
						(*x)[boundary_value.first] = boundary_value.second;

						// set constrained right hand side vector component
						(*b)[boundary_value.first] =
							diagonal_scaling_value * boundary_value.second;
					}
				}
			}

			// condense hanging nodes in projection matrix
			slab->space.primal.fe_info->constraints->condense(*primal.projection_matrix);

			// 4. solve projection linear system for primal.um_projected
			dealii::SparseDirectUMFPACK iA;
			iA.initialize(*primal.projection_matrix);
			iA.vmult(*primal.um_projected, *primal.projection_rhs);

			// distribute hanging node constraints on solution
			slab->space.primal.fe_info->constraints->distribute(
				*primal.um_projected
			);

			um->x[0] = primal.um_projected;

			DTM::pout << " (done)" << std::endl;
		}


		primal.Mum->reinit(
			slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
		);
		*primal.Mum = 0.;

		{
			IVAssembler<dim> assembler;

			DTM::pout << "dwr-instatfluid: assemble space-time slab initial value vector...";
			Assert(primal.um.use_count(), dealii::ExcNotInitialized());
			Assert(primal.Mum.use_count(), dealii::ExcNotInitialized());
			assembler.assemble(
				primal.um_projected,
				primal.Mum,
				slab
			);

			DTM::pout << " (done)" << std::endl;
		}
	}
	else // not using divfree projection
	{
		um->x[0] = primal.um;

		primal.Mum->reinit(
			slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
		);
		*primal.Mum = 0.;

		{
			IVAssembler<dim> assembler;

			DTM::pout << "dwr-instatfluid: assemble space-time slab initial value vector...";
			Assert(primal.um.use_count(), dealii::ExcNotInitialized());
			Assert(primal.Mum.use_count(), dealii::ExcNotInitialized());
			assembler.assemble(
				primal.um,
				primal.Mum,
				slab
			);

			DTM::pout << " (done)" << std::endl;
		}
	}

//	// ASSEMBLY SPACE-TIME OPERATOR: FORCE VECTOR //////////////////////////////
//	primal.f = std::make_shared< dealii::Vector<double> > ();
//
//	Assert(
//		slab->space.primal.fe_info->dof.use_count(),
//		dealii::ExcNotInitialized()
//	);
//	Assert(
//		slab->time.primal.fe_info->dof.use_count(),
//		dealii::ExcNotInitialized()
//	);
//
//	primal.f->reinit(
//		slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
//	);
//	*primal.f = 0.;
//
//	{
//		ForceAssembler<dim> assembler;
//
//		Assert(function.fluid.force.use_count(), dealii::ExcNotInitialized());
//		assembler.set_functions(
//			function.fluid.force
//		);
//
//		DTM::pout << "dwr-fluid: assemble space-time slab force vector...";
//		Assert(primal.f.use_count(), dealii::ExcNotInitialized());
//		assembler.assemble(
//			primal.f,
//			slab
//		);
//
//		DTM::pout << " (done)" << std::endl;
//	}
}


template<int dim>
void
Fluid<dim>::
primal_assemble_and_construct_Newton_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::map<dealii::types::global_dof_index, double> &boundary_values,
	std::shared_ptr<dealii::Vector<double> > u

) {
	// ASSEMBLY SPACE-TIME OPERATOR: Rhs Bilinearform VECTOR ///////////////////////
	primal.Fu = std::make_shared< dealii::Vector<double> > ();

	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	primal.Fu->reinit(
		slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
	);
	*primal.b = 0.;
	*primal.Fu = 0.;
	{
		NewtonRHSAssembler<dim> assembler;

		assembler.set_symmetric_stress(parameter_set->fe.symmetric_stress);

		Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
		assembler.set_functions(
			function.viscosity
		);

		assembler.set_time_quad_type((
				!parameter_set->fe.primal_order.compare("low") ?
						parameter_set->fe.low.convection.time_type_support_points :
						parameter_set->fe.high.convection.time_type_support_points
		));

		Assert(primal.Fu.use_count(), dealii::ExcNotInitialized());
		assembler.assemble(
			primal.Fu,
			slab,
			u,
			( parameter_set->problem.compare("Navier-Stokes") == 0 )
		);
	}

	*primal.b = *primal.Mum;
	primal.b->add(-1., *primal.Fu);
	primal_apply_bc(boundary_values, primal.b);
}

template<int dim>
void
Fluid<dim>::
primal_calculate_boundary_values(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::map<dealii::types::global_dof_index, double> &boundary_values,
	bool zero
) {
	std::shared_ptr< dealii::VectorFunctionFromTensorFunction<dim> > dirichlet_function
		= std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
			*function.convection.dirichlet,
			0, (dim+1)
		);

	// get all boundary colours on the current triangulation
	std::unordered_set< dealii::types::boundary_id > boundary_colours;
	{
		auto cell = slab->space.tria->begin_active();
		auto endc = slab->space.tria->end();
		for ( ; cell != endc; ++cell) {
		if (cell->at_boundary()) {
			// loop over all faces
			for (unsigned int face_no{0};
				face_no < dealii::GeometryInfo<dim>::faces_per_cell;
				++face_no) {
			if (cell->face(face_no)->at_boundary()) {
				auto face = cell->face(face_no);
				boundary_colours.insert(
					static_cast< unsigned int > (face->boundary_id())
				);
			}}
		}}
	}

	// process all found boundary colours
	for (auto colour : boundary_colours) {
		//////////////////////////////////////
		// component wise convection dirichlet
		//

		auto component_mask_convection =
		std::make_shared< dealii::ComponentMask > (
			(dim+1), false
		);

		switch (dim) {
		case 3:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_convection_c3)) {
				component_mask_convection->set(2, true);
			}

			// NOTE: not to break switch here is intended
			[[fallthrough]];

		case 2:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_convection_c2)) {
				component_mask_convection->set(1, true);
			}

			// NOTE: not to break switch here is intended
			[[fallthrough]];

		case 1:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_convection_c1)) {
				component_mask_convection->set(0, true);
			}

			break;

		default:
			AssertThrow(false, dealii::ExcNotImplemented());
		}

		// create boundary_values as
		// std::map<dealii::types::global_dof_index, double>
		{
			std::shared_ptr< dealii::Quadrature<1> > support_points;
			if (!parameter_set->fe.primal_order.compare("low"))
			{
				if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss")) ) {

					support_points =
							std::make_shared< dealii::QGauss<1> > (
									(parameter_set->fe.low.convection.r + 1)
							);
				} else if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss-Lobatto")) ){

					if (parameter_set->fe.low.convection.r < 1){
						support_points =
								std::make_shared< QRightBox<1> > ();
					} else {
						support_points =
								std::make_shared< dealii::QGaussLobatto<1> > (
										(parameter_set->fe.low.convection.r + 1)
								);
					}
				}
			}
			else
			{
				if ( !(parameter_set->
						fe.high.convection.time_type_support_points
						.compare("Gauss")) ) {

					support_points =
							std::make_shared< dealii::QGauss<1> > (
									(parameter_set->fe.high.convection.r + 1)
							);

				} else if ( !(parameter_set->
						fe.high.convection.time_type_support_points
						.compare("Gauss-Lobatto")) ){

					if (parameter_set->fe.high.convection.r < 1){
						support_points =
								std::make_shared< QRightBox<1> > ();

					} else {
						support_points =
								std::make_shared< dealii::QGaussLobatto<1> > (
										(parameter_set->fe.high.convection.r + 1)
								);
					}
				}
			}



			auto cell_time = slab->time.primal.fe_info->dof->begin_active();
			auto endc_time = slab->time.primal.fe_info->dof->end();

			dealii::FEValues<1> time_fe_values(
				*slab->time.primal.fe_info->mapping,
				*slab->time.primal.fe_info->fe,
				*support_points,
				dealii::update_quadrature_points
			);

			for ( ; cell_time != endc_time; ++cell_time) {
				time_fe_values.reinit(cell_time);

				for (unsigned int qt{0}; qt < support_points->size(); ++qt) {
					dirichlet_function->set_time(
						time_fe_values.quadrature_point(qt)[0]
					);

					// pass through time to the actual function since it
					// doesn't work through the wrapper from a tensor function
					function.convection.dirichlet->set_time(
						time_fe_values.quadrature_point(qt)[0]
					);

					std::map<dealii::types::global_dof_index,double>
						boundary_values_qt;

					if ( zero )
					{
						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.primal.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							dealii::ZeroFunction<dim>(dim+1),
							boundary_values_qt,
							*component_mask_convection
						);
					}
					else{
						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.primal.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							*dirichlet_function,
							boundary_values_qt,
							*component_mask_convection
						);
					}
					// boundary_values_qt -> boundary_values
					for (auto &el : boundary_values_qt) {
						dealii::types::global_dof_index idx =
							el.first
							// time offset
							+ slab->space.primal.fe_info->dof->n_dofs() *
								(cell_time->index()
								* slab->time.primal.fe_info->fe->dofs_per_cell)
							// local in time dof
							+ slab->space.primal.fe_info->dof->n_dofs() * qt
						;

						boundary_values[idx] = el.second;
					}
				}
			}
		}

		///////////////////////////////////////////////////////////
		// prescribed_no_slip: all convection components are homog.
		//

		for (unsigned int component{0}; component < (dim+1); ++component) {
			component_mask_convection->set(component, false);
		}

		switch (dim) {
		case 3:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_no_slip)) {
				component_mask_convection->set(2, true);
			}

			// NOTE: not to break switch here is intended
			[[fallthrough]];

		case 2:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_no_slip)) {
				component_mask_convection->set(1, true);
			}

			// NOTE: not to break switch here is intended
			[[fallthrough]];

		case 1:
			if (colour & static_cast< dealii::types::boundary_id > (
				fluid::types::space::boundary_id::prescribed_no_slip)) {
				component_mask_convection->set(0, true);
			}

			break;

		default:
			AssertThrow(false, dealii::ExcNotImplemented());
		}

		// create boundary_values as
		// std::map<dealii::types::global_dof_index, double>
		{
			const dealii::QGauss<1> support_points(
				slab->time.primal.fe_info->fe->tensor_degree()+1
			);

			auto cell_time = slab->time.primal.fe_info->dof->begin_active();
			auto endc_time = slab->time.primal.fe_info->dof->end();

			dealii::FEValues<1> time_fe_values(
				*slab->time.primal.fe_info->mapping,
				*slab->time.primal.fe_info->fe,
				support_points,
				dealii::update_quadrature_points
			);

			for ( ; cell_time != endc_time; ++cell_time) {
				time_fe_values.reinit(cell_time);

				for (unsigned int qt{0}; qt < support_points.size(); ++qt) {
					std::map<dealii::types::global_dof_index,double>
						boundary_values_qt;

					dealii::VectorTools::interpolate_boundary_values (
						*slab->space.primal.fe_info->dof,
						static_cast< dealii::types::boundary_id > (
							colour
						),
						dealii::ZeroFunction<dim>(dim+1),
						boundary_values_qt,
						*component_mask_convection
					);

					// boundary_values_qt -> boundary_values
					for (auto &el : boundary_values_qt) {
						dealii::types::global_dof_index idx =
							el.first
							// time offset
							+ slab->space.primal.fe_info->dof->n_dofs() *
								(cell_time->index()
								* slab->time.primal.fe_info->fe->dofs_per_cell)
							// local in time dof
							+ slab->space.primal.fe_info->dof->n_dofs() * qt
						;

						boundary_values[idx] = el.second;
					}
				}
			} // no slip
		}
	} // for each (boundary) colour
}

template<int dim>
void
Fluid<dim>::
primal_apply_bc(
	std::map<dealii::types::global_dof_index, double> &boundary_values,
	std::shared_ptr< dealii::Vector<double> > x
) {
	////////////////////////////////////////////////////////////////////////
	// MatrixTools::apply_boundary_values (number = double for A,x,b)
	// input: x vector: IR^n
	//
	//        boundary_values: map< dof, double >
	//
	//
	// NOTE: a spurious, but nicely scaled, singular value is introduced
	//       in the operator matrix for each boundary value constraint
	//
	if (boundary_values.size()) {
		////////////////////////////////////////////////////////////////////
		// apply boundary values to vector x
		for (auto &boundary_value : boundary_values) {
			// set constrained solution vector component (for iterative lss)
			(*x)[boundary_value.first] = boundary_value.second;
		}
	}
}

template<int dim>
void
Fluid<dim>::
primal_apply_bc(
	std::map<dealii::types::global_dof_index, double> &boundary_values,
	std::shared_ptr< dealii::SparseMatrix<double> > A,
	std::shared_ptr< dealii::Vector<double> > x,
	std::shared_ptr< dealii::Vector<double> > b
) {
	////////////////////////////////////////////////////////////////////////
	// MatrixTools::apply_boundary_values (number = double for A,x,b)
	// input: A sparse matrix: IR^(m,n)
	//        x vector: IR^n
	//        b vector: IR^m
	//
	//        boundary_values: map< dof, double >
	//
	// NOTE: constrained rows (and columns) are not removed
	//       (not condensed) from the sparsity pattern due to
	//       memory efficiency reasons (this avoids the expensive
	//       reallocation and copying of the operator and vectors)
	//
	// NOTE: a spurious, but nicely scaled, singular value is introduced
	//       in the operator matrix for each boundary value constraint
	//
	if (boundary_values.size()) {
		////////////////////////////////////////////////////////////////////
		// internal checks
		Assert(
			A->m(),
			dealii::ExcMessage(
				"empty operator matrix A (rows: A.m() == 0)"
			)
		);

		Assert(
			A->n(),
			dealii::ExcMessage(
				"empty operator matrix A (cols: A.n() == 0)"
			)
		);

		Assert(
			(A->n() == x->size()),
			dealii::ExcMessage(
				"Internal dimensions error: "
				"A->n() =/= x->size() of linear system A x = b"
			)
		);

		Assert(
			(A->m() == b->size()),
			dealii::ExcMessage(
				"Internal dimensions error: "
				"A->m() =/= b->size() of linear system A x = b"
			)
		);

		////////////////////////////////////////////////////////////////////
		// apply boundary values to operator matrix A, vectors x and b

		// preparation: eliminate constrained rows of operator A completely
		for (auto &boundary_value : boundary_values) {
			// on-the-fly check:
			// validity of the given boundary value constraints
			Assert(
				(boundary_value.first < A->m()),
				dealii::ExcMessage(
					"constraining of dof index (as boundary value) "
					"larger than (A.m()-1) is not valid"
				)
			);

			// eliminate constrained row of operator matrix completely
			auto el_in_row_i{A->begin(boundary_value.first)};
			auto end_el_in_row_i{A->end(boundary_value.first)};

			for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
				el_in_row_i->value() = double(0.);
			}
		}

		////////////////////////////////////////////////////////////////////
		// find (absolute) maximum of all remaining diagonal elements
		// as scaling parameter for the boundary values identity operator
		//
		// NOTE: this is (a bit) more expensive then simply using the
		//       first non-zero diagonal element (for the case that in
		//       a constrained row the diagonal element is zero).
		//       The advantage is, that the spurious singular values of
		//       all boundary value constraints are identical, i.e.
		//       a fixed and nicely scaled value.
		//
		double diagonal_scaling_value{1.}; //0.};

//		for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
//			if (std::abs(A->el(i,i)) > std::abs(diagonal_scaling_value)) {
//				diagonal_scaling_value = A->el(i,i);
//			}
//		}
//
//		if (diagonal_scaling_value == double(0.)) {
//			diagonal_scaling_value = double(1.);
//		}
//
//		Assert(
//			(diagonal_scaling_value != double(0.)),
//			dealii::ExcInternalError()
//		);

		////////////////////////////////////////////////////////////////////
		// apply boundary values:
		// to the linear system (A x = b)
		// without eliminating the corresponding column entries from
		// the operator matrix A
		//
		for (auto &boundary_value : boundary_values) {
			// set scaled identity operator
			A->set(
				boundary_value.first,  // i
				boundary_value.first,  // i
				diagonal_scaling_value // scaling factor or 1
			);

			// set constrained solution vector component (for iterative lss)
			(*x)[boundary_value.first] = boundary_value.second;

			// set constrained right hand side vector component
			(*b)[boundary_value.first] =
				diagonal_scaling_value * boundary_value.second;
		}

//		 			////////////////////////////////////////////////////////////////////
//		 			// eliminate constrained column entries
//		 			//
//		 			// NOTE: this is quite expensive, but helps iterative lss
//		 			//       since the boundary value entries are shifted to the
//		 			//       right hand side.
//		 			//
//		 			// NOTE: there is no symmetry assumption on the sparsity pattern,
//		 			//       which is necessary for space-time operators
//		 			//
//		 			for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
//		 				// if the row i of the operator A is not constrained,
//		 				// check if constrained columns need to be eliminated
//		 				if (boundary_values.find(i) == boundary_values.end()) {
//		 					// row i of A is not constrained
//		 					auto el_in_row_i{A->begin(i)};
//		 					auto end_el_in_row_i{A->end(i)};
//
//		 					// check if a_ij needs to be eliminated
//		 					for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
//		 						// get iterator of a_ij
//		 						auto boundary_value =
//		 							boundary_values.find(el_in_row_i->column());
//
//		 						// if a_ij is constrained
//		 						if (boundary_value != boundary_values.end()) {
//		 							// shift constraint to rhs
//		 							(*b)[i] -=
//		 								el_in_row_i->value()*boundary_value->second;
//
//		 							// eliminate a_ij
//		 							el_in_row_i->value() = 0.;
//		 						}
//		 					}
//		 				}
//		 			}
	}
}

template<int dim>
void
Fluid<dim>::
primal_solve_slab_problem(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const typename DTM::types::storage_data_vectors<1>::iterator &um
) {
	
	primal.b  = std::make_shared< dealii::Vector<double> >();
	primal.du = std::make_shared< dealii::Vector<double> >();
	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	
	primal.b->reinit(
		slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
	);

	primal.du->reinit(
		slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()
	);

	////////////////////////////////////////////////////////////////////////////
	// apply inhomogeneous Dirichlet boundary values
	//

	DTM::pout << "dwr-instatfluid: compute boundary values..." ;

	std::map<dealii::types::global_dof_index, double> initial_bc;
	primal_calculate_boundary_values(slab, initial_bc);
    std::map<dealii::types::global_dof_index, double> zero_bc;
    primal_calculate_boundary_values(slab, zero_bc, true);

    DTM::pout << " (done)" << std::endl;

    DTM::pout << "dwr-instatfluid: apply previous solution as initial Newton guess..." ;

    for (unsigned int i{0} ; i < slab->space.primal.fe_info->dof->n_dofs() ; i++) {
    	for (unsigned int ii{0} ; ii < slab->time.primal.fe_info->dof->n_dofs() ; ii++) {
    		(*u->x[0])[i+slab->space.primal.fe_info->dof->n_dofs()*ii] = (*primal.um)[i];
    	}
    }

    DTM::pout << " (done)" << std::endl;

    primal_apply_bc(initial_bc, u->x[0]);
    slab->spacetime.primal.constraints->distribute(
		*u->x[0]
	);

    // assemble slab problem const rhs
	primal_assemble_const_rhs(slab, um, initial_bc);

	Assert(
		primal.Mum.use_count(),
		dealii::ExcNotInitialized()
	);

    primal_assemble_and_construct_Newton_rhs(slab, zero_bc, u->x[0]);

    DTM::pout << "dwr-instatfluid: starting Newton loop\n"
    		  << "It.\tResidual\tReduction\tRebuild\tLSrch"<< std::endl;

    double newton_residual = primal.b->linfty_norm();
    double old_newton_residual = newton_residual;
    double new_newton_residual;

    unsigned int newton_step = 1;
    unsigned int line_search_step;

    DTM::pout << std::setprecision(5) << "0\t" << newton_residual << std::endl;

    while (newton_residual > newton.lower_bound && newton_step <= newton.max_steps)
    {
    	old_newton_residual = newton_residual;
    	primal_assemble_and_construct_Newton_rhs(slab, zero_bc, u->x[0]);
    	newton_residual = primal.b->linfty_norm();

    	if (newton_residual < newton.lower_bound)
		{
			DTM::pout << "res\t" << newton_residual << std::endl;
			break;
		}

		if (newton_residual/old_newton_residual > newton.rebuild){
			// NOTE: for Stokes without adaptive refinement the system matrix needs only to be inverted on the first slab
			if (!parameter_set->problem.compare("Navier-Stokes") || !parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") || slab->t_m == parameter_set->time.fluid.t0)
			{
				primal_assemble_system(slab, u->x[0]);
			}
			else
			{
				// system matrix doesn't need to be assembled, since it hasn't changed
				// hence also primal.iA stays the same
				primal.L = std::make_shared< dealii::SparseMatrix<double> > ();
				primal.L->reinit(*slab->spacetime.primal.sp);
				*primal.L = 0;
			}

			primal_apply_bc(zero_bc, primal.L, primal.du, primal.b);
//			// printing out the system matrix
//			std::ofstream out("primal_matrix.txt", std::ios_base::out);
//			primal.L->print(out);
//			out.close();
//			exit(9);

			////////////////////////////////////////////////////////////////////////////
			// condense hanging nodes in system matrix, if any
			//
			slab->spacetime.primal.constraints->condense(*primal.L);

			// NOTE: for Stokes without adaptive refinement the system matrix needs only to be inverted on the first slab
			if (!parameter_set->problem.compare("Navier-Stokes") || !parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") || slab->t_m == parameter_set->time.fluid.t0)
				primal.iA.initialize(*primal.L);
		}

		////////////////////////////////////////////////////////////////////////////
		// solve linear system with direct solver
		//
		primal.iA.vmult(*primal.du, *primal.b);

		slab->spacetime.primal.constraints->distribute(
			*primal.du
		);

		for (line_search_step = 0; line_search_step < newton.line_search_steps; line_search_step++) {
			u->x[0]->add(1.0,*primal.du);

			primal_assemble_and_construct_Newton_rhs(slab, zero_bc, u->x[0]);
			slab->spacetime.primal.constraints->distribute(
				*primal.b
			);

			new_newton_residual = primal.b->linfty_norm();

			if (new_newton_residual < newton_residual)
				break;
			else
				u->x[0]->add(-1.0, *primal.du);
			*primal.du *= newton.line_search_damping;
		}

		DTM::pout << std::setprecision(5) << newton_step << "\t"
				  << std::scientific << newton_residual << "\t"
				  << std::scientific << newton_residual/old_newton_residual << "\t";

		if (newton_residual/old_newton_residual > newton.rebuild)
			DTM::pout << "r\t";
		else
			DTM::pout << " \t";

		DTM::pout << line_search_step << "\t" << std::scientific << std::endl;
		if (line_search_step == newton.line_search_steps && std::abs(newton_residual - old_newton_residual) < 1e-15)
		{
			break;
		}
		newton_step++;
    }
}

template<int dim>
void
Fluid<dim>::
primal_subtract_pressure_mean(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double> > x
) {
	std::vector< dealii::types::global_dof_index > dofs_per_component(
			slab->space.primal.fe_info->dof->get_fe_collection().n_components(), 0
	);


	dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
			*slab->space.primal.fe_info->dof,
			true
	);

	// set specific values of dof counts
	dealii::types::global_dof_index N_b; // convection

	// dof count convection: vector-valued primitive FE
	N_b = 0;
	for (unsigned int d{0}; d < dim; ++d) {
		N_b += dofs_per_component[0*dim+d];
	}

	dealii::FEValues<1> fe_values_time(
			*slab->time.primal.fe_info->mapping,
			*slab->time.primal.fe_info->fe,
			dealii::QGauss<1>(parameter_set->fe.primal.convection.r + 1),
			dealii::update_quadrature_points
	);

	std::vector< dealii::types::global_dof_index > local_dof_indices(
			slab->time.primal.fe_info->fe->dofs_per_cell
	);

	dealii::types::global_dof_index n_dofs_space =
			slab->space.primal.fe_info->dof->n_dofs();

	auto cell_time = slab->time.primal.fe_info->dof->begin_active();
	auto endc_time = slab->time.primal.fe_info->dof->end();
	for ( ; cell_time != endc_time ; ++cell_time) {
		fe_values_time.reinit(cell_time);

		cell_time ->get_dof_indices(local_dof_indices);

		for ( unsigned int local_time_dof{0} ;
				local_time_dof < fe_values_time.n_quadrature_points;
				++local_time_dof
		){
			//extract current qp solution
			dealii::Vector<double> uk;
			uk.reinit(n_dofs_space);
			//					uk.clear();
			for (dealii::types::global_dof_index i{0} ; i < n_dofs_space ; i++)
			{
				dealii::types::global_dof_index ii =
						n_dofs_space*(local_dof_indices[local_time_dof])+i;

				uk[i] = x->operator()(ii);
			}

			const double mean_pressure = dealii::VectorTools::compute_mean_value(
					*slab->space.primal.fe_info->dof,
					dealii::QGauss<dim> (parameter_set->fe.primal.convection.p+2),
					uk,
					dim
			);

//			std::cout << "mean pressure correction: " << mean_pressure << std::endl;
			for ( dealii::types::global_dof_index i{N_b} ; i < n_dofs_space ; i++)
			{
				dealii::types::global_dof_index ii =
						n_dofs_space*(local_dof_indices[local_time_dof])+i;


				x->operator()(ii) -= mean_pressure;
			}

		}
	}

}
template<int dim>
void
Fluid<dim>::
primal_do_forward_TMS(
	const unsigned int dwr_loop,
	bool last
) {
	////////////////////////////////////////////////////////////////////////////
	// prepare time marching scheme (TMS) loop
	//
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	// resetting the goal functionals
	error_estimator.goal_functional.fem.mean_drag = 0.;
	error_estimator.goal_functional.fem.mean_lift = 0.;
	
	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to first space-time slab: Omega x I_1
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	
	////////////////////////////////////////////////////////////////////////////
	// storage: init iterators to storage_data_vectors
	//          corresponding to first space-time slab: Omega x I_1
	//
	
	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	auto u = primal.storage.u->begin();
	
	Assert(primal.storage.um.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.um->size(), dealii::ExcNotInitialized());
	auto um = primal.storage.um->begin();

	////////////////////////////////////////////////////////////////////////////
	// interpolate (or project) initial value(s)
	//
	
	////////////////////////////////////////////////////////////////////////////
	// do TMS loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "primal: solving forward TMS problem..." << std::endl
		<< std::endl;
	
	unsigned int n{1};
	while (slab != grid->slabs.end()) {
		DTM::pout
			<< "primal: solving problem on "
			<< "Q_" << n
			<< " = Omega_h x (" << slab->t_m << ", " << slab->t_n << ") "
			<< std::endl;
		
		if ( !parameter_set->fe.primal_order.compare("low") )
		{
			grid->initialize_low_grid_components_on_slab(slab);
			grid->distribute_low_on_slab(slab);
		}
		else if ( !parameter_set->fe.primal_order.compare("high") )
		{
			grid->initialize_high_grid_components_on_slab(slab);
			grid->distribute_high_on_slab(slab);
		}
		else
			AssertThrow(false, dealii::ExcNotImplemented());

		grid->create_sparsity_pattern_primal_on_slab(slab);
		primal_reinit_storage_on_slab(slab, u, um);

		if (slab == grid->slabs.begin()) {
			////////////////////////////////////////////////////////////////////////////
			// interpolate (or project) initial value(s)
			//
			std::shared_ptr< dealii::VectorFunctionFromTensorFunction<dim> > dirichlet_function
					= std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
						*function.convection.dirichlet,
						0, (dim+1)
					);

			dirichlet_function->set_time(0);
			function.convection.dirichlet->set_time(0);

			primal.um = std::make_shared< dealii::Vector<double> > ();
			primal.um->reinit(slab->space.primal.fe_info->dof->n_dofs());
			*primal.um = 0.;

			primal.um_projected = std::make_shared< dealii::Vector<double> > ();
			primal.um_projected->reinit(slab->space.primal.fe_info->dof->n_dofs());
			*primal.um_projected = 0.;

//			Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
//			function.u_0->set_time(slab->t_m);

			Assert((slab != grid->slabs.end()), dealii::ExcInternalError());
			Assert(slab->space.primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			Assert(primal.um.use_count(), dealii::ExcNotInitialized());
			
			dealii::VectorTools::interpolate(
				*slab->space.primal.fe_info->mapping,
				*slab->space.primal.fe_info->dof,
				*dirichlet_function, //*function.u_0,
				*primal.um
			);

			// NOTE: after the first dwr-loop the initial triangulation could have
			//       hanging nodes. Therefore,
			// distribute hanging node constraints to make the result continuous again:
			slab->space.primal.fe_info->constraints->distribute(
				*primal.um
			);
		}
		else {
			// not the first slab: transfer un solution to um solution
			Assert(primal.un.use_count(), dealii::ExcNotInitialized());

			primal.um = std::make_shared< dealii::Vector<double> > ();
			primal.um->reinit(slab->space.primal.fe_info->dof->n_dofs());
			*primal.um = 0.;

			primal.um_projected = std::make_shared< dealii::Vector<double> > ();
			primal.um_projected->reinit(slab->space.primal.fe_info->dof->n_dofs());
			*primal.um_projected = 0.;

			// for n > 1 interpolate between two (different) spatial meshes
			// the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->space.primal.fe_info->dof,
				*primal.un,
				// solution on I_n:
				*slab->space.primal.fe_info->dof,
				*slab->space.primal.fe_info->constraints,
				*primal.um
			);

			slab->space.primal.fe_info->constraints->distribute(
				*primal.um
			);
		}
		
		// solve slab problem (i.e. apply boundary values and solve for u0)
		primal_solve_slab_problem(slab, u, um);
		
		// FOR DEBUGGING: output primal space-time solution vector
		{
			int slab_number = 0;
			auto tmp_slab = grid->slabs.begin();
			while (slab != tmp_slab)
			{
				slab_number++;
				tmp_slab++;
			}

			std::ostringstream filename;
			filename << "u_" << std::setw(3) << std::setfill('0') << slab_number << ".txt";
			std::ofstream out(filename.str().c_str(), std::ios_base::out);
			u->x[0]->print(out,8,true,false);
		}

		////////////////////////////////////////////////////////////////////////
		// do postprocessings on the solution
		//
		
		// evaluate solution u(t_n)
		primal.un = std::make_shared< dealii::Vector<double> > ();
		primal.un->reinit(slab->space.primal.fe_info->dof->n_dofs());
		*primal.un = 0.;

		{
			dealii::FEValues<1> fe_face_values_time(
				*slab->time.primal.fe_info->mapping,
				*slab->time.primal.fe_info->fe,
				dealii::QGaussLobatto<1>(2),
				dealii::update_values
			);

			auto cell_time = slab->time.primal.fe_info->dof->begin_active();
			auto last_cell_time = cell_time;
			auto endc_time = slab->time.primal.fe_info->dof->end();

			for ( ; cell_time != endc_time; ++cell_time) {
				last_cell_time=cell_time;
			}

			cell_time=last_cell_time;
			{
				Assert((cell_time != endc_time), dealii::ExcInternalError());
				fe_face_values_time.reinit(cell_time);

				// evaluate solution for t_n of Q_n
				for (unsigned int jj{0};
					jj < slab->time.primal.fe_info->fe->dofs_per_cell; ++jj)
				for (dealii::types::global_dof_index i{0};
					i < slab->space.primal.fe_info->dof->n_dofs(); ++i) {
					(*primal.un)[i] += (*u->x[0])[
						i
						// time offset
						+ slab->space.primal.fe_info->dof->n_dofs() *
							(cell_time->index() * slab->time.primal.fe_info->fe->dofs_per_cell)
						// local in time dof
						+ slab->space.primal.fe_info->dof->n_dofs() * jj
					] * fe_face_values_time.shape_value(jj,1);
				}
			}
		}

		// output data
		primal_do_data_output(slab, u, dwr_loop, last);

		////////////////////////////////////////////////////////////////////////
		// compute functional values:
		//
		compute_functional_values(u, slab);
		
		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//
		
		primal.L = nullptr;
		primal.b = nullptr;
// 		primal.f = nullptr;
		primal.Mum = nullptr;
		
		grid->clear_primal_on_slab(slab);

		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//

		++n;
		++slab;
		++u;
		++um;

		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "primal: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// output exact error

	// analytical mean drag
	double reference_goal_functional;
	if (parameter_set->problem.compare("Navier-Stokes") == 0)
		reference_goal_functional = error_estimator.goal_functional.reference.NSE.mean_drag;
	else
		reference_goal_functional = error_estimator.goal_functional.reference.Stokes.mean_drag;
	// FEM mean drag
	double fem_goal_functional = error_estimator.goal_functional.fem.mean_drag;

	DTM::pout << "---------------------------" << std::endl;
	DTM::pout << "Mean drag:" << std::endl;
	DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
	DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
	DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
	DTM::pout << "---------------------------" << std::endl << std::endl;

	// analytical mean lift
	if (parameter_set->problem.compare("Navier-Stokes") == 0)
		reference_goal_functional = error_estimator.goal_functional.reference.NSE.mean_lift;
	else
		reference_goal_functional = error_estimator.goal_functional.reference.Stokes.mean_lift;
	// FEM mean lift
	fem_goal_functional = error_estimator.goal_functional.fem.mean_lift;

	DTM::pout << "---------------------------" << std::endl;
	DTM::pout << "Mean lift:" << std::endl;
	DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
	DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
	DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
	DTM::pout << "---------------------------" << std::endl << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// allow garbage collector to clean up memory
	//
	
	primal.um = nullptr;
	primal.un = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
// primal data output
//

template<int dim>
void
Fluid<dim>::
primal_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	// set up which dwr loop(s) are allowed to make data output:
	if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
		return;
	}
	
	// may output data: initialise (mode: all, last or specific dwr loop)
	DTM::pout
		<< "primal solution data output: patches = "
		<< parameter_set->data_output.primal.patches
		<< std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// INIT DATA POSTPROCESSOR
	unsigned int output_quantities(0);
	
	output_quantities |=
		static_cast<unsigned int>(fluid::OutputQuantities::convection);
	
	output_quantities |=
		static_cast<unsigned int>(fluid::OutputQuantities::pressure);
	
	primal.data_postprocessor = std::make_shared<
		fluid::DataPostprocessor<dim> > (
		output_quantities
	);
	
	////////////////////////////////////////////////////////////////////////////
	// INIT DATA OUTPUT
	//
	
	primal.data_output = std::make_shared< DTM::DataOutput<dim> >();
	
	std::vector<std::string> data_field_names;
	for (unsigned int i{0}; i < dim; ++i)
		data_field_names.push_back("convection");
	data_field_names.push_back("pressure");
	
	primal.data_output->set_data_field_names(data_field_names);
	
	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	for (unsigned int i{0}; i < dim; ++i)
		dci_field.push_back(dealii::DataComponentInterpretation::component_is_part_of_vector);
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);
	
	primal.data_output->set_data_component_interpretation_field(dci_field);
	
	primal.data_output->set_data_output_patches(
		parameter_set->data_output.primal.patches
	);
	
	// check if we use a fixed trigger interval, or, do output once on a I_n
	if ( !parameter_set->data_output.primal.trigger_type.compare("fixed") ) {
		primal.data_output_trigger_type_fixed = true;
	}
	else {
		primal.data_output_trigger_type_fixed = false;
	}
	
	// only for fixed
	primal.data_output_trigger = parameter_set->data_output.primal.trigger;
	
	if (primal.data_output_trigger_type_fixed) {
		DTM::pout
			<< "primal solution data output: using fixed mode with trigger = "
			<< primal.data_output_trigger
			<< std::endl;
	}
	else {
		DTM::pout
			<< "primal solution data output: using I_n mode (trigger adapts to I_n automatically)"
			<< std::endl;
	}
	
	primal.data_output_time_value = parameter_set->time.fluid.t0;
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const unsigned int dwr_loop) {
	// triggered output mode
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.primal.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	primal.data_output->set_DoF_data(
		slab->space.primal.fe_info->dof,
		slab->space.primal.fe_info->partitioning_locally_owned_dofs
	);
	
 	auto u_trigger = std::make_shared< dealii::Vector<double> > ();
 	u_trigger->reinit(slab->space.primal.fe_info->dof->n_dofs());
	
	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
	
	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.primal.fe_info->mapping,
			*slab->time.primal.fe_info->fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);
		
		Assert(
			slab->time.primal.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);
		
		auto cell_time = slab->time.primal.fe_info->dof->begin_active();
		auto endc_time = slab->time.primal.fe_info->dof->end();
		
		for ( ; cell_time != endc_time; ++cell_time) {
			fe_face_values_time.reinit(cell_time);
			
			////////////////////////////////////////////////////////////////////
			// construct quadrature for data output,
			// if triggered output time values are inside this time element
			//
			
			auto t_m = fe_face_values_time.quadrature_point(0)[0];
			auto t_n = fe_face_values_time.quadrature_point(1)[0];
			auto tau = t_n-t_m;
			
			std::list<double> output_times;
			
			if (primal.data_output_time_value < t_m) {
				primal.data_output_time_value = t_m;
			}
			
			for ( ; (primal.data_output_time_value <= t_n) ||
				(
					(primal.data_output_time_value > t_n) &&
					(std::abs(primal.data_output_time_value - t_n) < tau*1e-12)
				); ) {
				output_times.push_back(primal.data_output_time_value);
				primal.data_output_time_value += primal.data_output_trigger;
			}
			
			if (output_times.size() && output_times.back() > t_n) {
				output_times.back() = t_n;
			}
			
			if ((output_times.size() > 1) &&
				(output_times.back() == *std::next(output_times.rbegin()))) {
				// remove the last entry, iff doubled
				output_times.pop_back();
			}
			
			// convert container
			if (!output_times.size()) {
				continue;
			}
			
			std::vector< dealii::Point<1> > output_time_points(output_times.size());
			{
				auto time{output_times.begin()};
				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
					double t_trigger{*time};
					output_time_points[q][0] = (t_trigger-t_m)/tau;
				}
				
				if (output_time_points[0][0] < 0) {
					output_time_points[0][0] = 0;
				}
				
				if (output_time_points[output_time_points.size()-1][0] > 1) {
					output_time_points[output_time_points.size()-1][0] = 1;
				}
			}
			
			dealii::Quadrature<1> quad_time(output_time_points);
			
			// create fe values
			dealii::FEValues<1> fe_values_time(
				*slab->time.primal.fe_info->mapping,
				*slab->time.primal.fe_info->fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);
			
			fe_values_time.reinit(cell_time);
			
			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.primal.fe_info->fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);
			
			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*u_trigger = 0.;
 				
 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.primal.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.primal.fe_info->dof->n_dofs(); ++i) {
 					(*u_trigger)[i] += (*u->x[0])[
 						i
 						// time offset
 						+ slab->space.primal.fe_info->dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}
				
//				std::cout
//					<< "output generated for t = "
//					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
//					<< std::endl;
				
				primal.data_output->write_data(
					filename.str(),
					u_trigger,
					primal.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
	
	// check if data for t=T was written
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &u,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.primal.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	primal.data_output->set_DoF_data(
		slab->space.primal.fe_info->dof,
		slab->space.primal.fe_info->partitioning_locally_owned_dofs
	);
	
	auto u_trigger = std::make_shared< dealii::Vector<double> > ();
 	u_trigger->reinit(slab->space.primal.fe_info->dof->n_dofs());
 	
	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
	
	{
		// choose temporal quadrature
		std::shared_ptr< dealii::Quadrature<1> > time_support_points;
		{
			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss")) ) {

				time_support_points =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.primal.convection.r + 1)
				);
			}

			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ) {

				time_support_points =
				std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.primal.convection.r + 1)
				);
			}
		}

		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.primal.fe_info->mapping,
			*slab->time.primal.fe_info->fe,
			*time_support_points,
			dealii::update_values |
			dealii::update_quadrature_points
		);
		
		auto cell_time = slab->time.primal.fe_info->dof->begin_active();
		auto endc_time = slab->time.primal.fe_info->dof->end();
		
		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);
			
			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.primal.fe_info->fe->dofs_per_cell
			);
			
			cell_time->get_dof_indices(local_dof_indices_time);
			
			for (
				unsigned int qt{0};
				qt < fe_values_time.n_quadrature_points;
				++qt) {
 				*u_trigger = 0.;
 				
 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.primal.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.primal.fe_info->dof->n_dofs(); ++i) {
 					(*u_trigger)[i] += (*u->x[0])[
 						i
 						// time offset
 						+ slab->space.primal.fe_info->dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}
				
// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0]
// 					<< std::endl;
				
 				primal.data_output->write_data(
					filename.str(),
					u_trigger,
					primal.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] + (qt == 0) * 1e-5 // t_trigger, slightly shifted for left time cell end point to avoid ParaView errors caused by 2 outputs at same time point
				);

//				// TODO: delete vtk output
//				{
//					// get slab number
//					int slab_number = 0;
//					auto tmp_slab = grid->slabs.begin();
//					while (slab != tmp_slab)
//					{
//						slab_number++;
//						tmp_slab++;
//					}
//
//					std::vector<std::string> solution_names;
//					solution_names.push_back("x_velo");
//					solution_names.push_back("y_velo");
//					solution_names.push_back("p_fluid");
//
//					std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
//						data_component_interpretation(dim + 1, dealii::DataComponentInterpretation::component_is_scalar);
//
//					dealii::DataOut<dim> data_out;
//					data_out.attach_dof_handler(*slab->space.low.fe_info->dof);
//
//					data_out.add_data_vector(*u_trigger, solution_names,
//											 dealii::DataOut<dim>::type_dof_data,
//											 data_component_interpretation);
//
//					data_out.build_patches();
//					data_out.set_flags(
//							dealii::DataOutBase::VtkFlags(
//									slab->t_n,
//									slab_number
//							)
//					);
//
//					// save VTK files
//					const std::string filename =
//						"interpolated_solution-" + dealii::Utilities::int_to_string(slab_number, 6) + ".vtk";
//					std::ofstream output(filename);
//					data_out.write_vtk(output);
//				}
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &x,
	const unsigned int dwr_loop,
	bool last
) {
	if (primal.data_output_trigger <= 0) return;
	
	// set up which dwr loop(s) are allowed to make data output:
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	if ( !parameter_set->data_output.primal.dwr_loop.compare("none") ) {
		return;
	}
	
	if (!parameter_set->data_output.primal.dwr_loop.compare("last")) {
		// output only the last (final) dwr loop
		if (last) {
			primal.data_output_dwr_loop = dwr_loop;
		}
		else {
			return;
		}
	}
	else {
		if (!parameter_set->data_output.primal.dwr_loop.compare("all")) {
			// output all dwr loops
			if (!last) {
				primal.data_output_dwr_loop = dwr_loop;
			}
			else {
				return;
			}
		}
		else {
			// output on a specific dwr loop
			if (!last) {
				primal.data_output_dwr_loop =
					std::stoi(parameter_set->data_output.primal.dwr_loop)-1;
			}
			else {
				return;
			}
		}
	}
	
	if (primal.data_output_dwr_loop < 0)
		return;
	
	if ( static_cast<unsigned int>(primal.data_output_dwr_loop) != dwr_loop )
		return;
	
	if (!primal.data_output_trigger_type_fixed) {
		// I_n output mode (output on natural Q_n support points in time)
		primal_do_data_output_on_slab_Qn_mode(slab, x, dwr_loop);
	}
	else {
		// fixed trigger output mode
		primal_do_data_output_on_slab(slab, x, dwr_loop);
	}
}

////////////////////////////////////////////////////////////////////////////////
// dual problem
//

template<int dim>
void
Fluid<dim>::
dual_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data:
	// NOTE: * dual space: time dG(r) method (all dofs in 1 vector)
	//       * dual solution dof vectors: z
	//

	Assert(grid.use_count(), dealii::ExcNotInitialized());

	dual.storage.z =
		std::make_shared< DTM::types::storage_data_vectors<1> > ();

	dual.storage.z->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);
}


template<int dim>
void
Fluid<dim>::
dual_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &z
) {
	for (unsigned int j{0}; j < z->x.size(); ++j) {
		z->x[j] = std::make_shared< dealii::Vector<double> > ();

		Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

		Assert(
			slab->space.dual.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		Assert(
			slab->time.dual.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		z->x[j]->reinit(
			slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()
		);
	}
}


template<int dim>
void
Fluid<dim>::
dual_assemble_system(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::Vector<double > > u
) {
	// ASSEMBLY SPACE-TIME OPERATOR MATRIX /////////////////////////////////////
	Assert(
		slab->spacetime.dual.sp.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.L = std::make_shared< dealii::SparseMatrix<double> > ();
	dual.L->reinit(*slab->spacetime.dual.sp);
	*dual.L = 0;

	{
		fluid::spacetime::dual::Operator::
		Assembler<dim> dual_assembler;

		dual_assembler.set_symmetric_stress(parameter_set->fe.symmetric_stress);

		Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
		dual_assembler.set_functions(
			function.viscosity
		);

		dual_assembler.set_time_quad_type((
				!parameter_set->fe.dual_order.compare("low") ?
						parameter_set->fe.low.convection.time_type_support_points :
						parameter_set->fe.high.convection.time_type_support_points
		));

		DTM::pout << "dynamic fluid: assemble space-time slab dual operator matrix...";
		Assert(dual.L.use_count(), dealii::ExcNotInitialized());
		dual_assembler.assemble(
			dual.L,
			slab,
			u,
			(parameter_set->problem.compare("Navier-Stokes") == 0)
		);

		DTM::pout << " (done)" << std::endl;
	}

	// FOR DEBUGGING PRINT MATRIX:
	{
		int slab_number = 0;
		auto tmp_slab = grid->slabs.begin();
		while (slab != tmp_slab)
		{
			slab_number++;
			tmp_slab++;
		}

		std::ostringstream filename;
		filename << "dual_matrix_" << std::setw(3) << std::setfill('0') << slab_number << ".txt";
		std::ofstream out(filename.str().c_str(), std::ios_base::out);
		dual.L->print(out);
		out.close();
	}
}


template<int dim>
void
Fluid<dim>::
dual_assemble_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// NOTE: for nonlinear goal functionals, we need also the primal solution u for this function

	// ASSEMBLY SPACE-TIME OPERATOR: FinalValue VECTOR ///////////////////////
	dual.Mzn = std::make_shared< dealii::Vector<double> > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Mzn->reinit(
		slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()
	);
	*dual.Mzn = 0.;

	{
		FVDualAssembler<dim> dual_assembler;

		DTM::pout << "dwr-instatfluid: assemble space-time slab final value vector...";
		Assert(dual.zn.use_count(), dealii::ExcNotInitialized());
		Assert(dual.Mzn.use_count(), dealii::ExcNotInitialized());
		dual_assembler.assemble(
			dual.zn,
			dual.Mzn,
			slab
		);

		DTM::pout << " (done)" << std::endl;
	}

	// ASSEMBLY SPACE-TIME OPERATOR: GOAL FUNCTIONAL VECTOR //////////////////////////////
	dual.Je = std::make_shared< dealii::Vector<double> > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Je->reinit(
		slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()
	);
	*dual.Je = 0.;

	{
		Je_MeanDrag_Assembler<dim> Je_assembler;

		Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
		Je_assembler.set_functions(
			function.viscosity
		);

		DTM::pout << "dwr-instatfluid: assemble space-time slab Je_meandrag vector...";
		Je_assembler.assemble(
			dual.Je,
			slab,
			parameter_set->time.fluid.t0,
			parameter_set->time.fluid.T
		);

		DTM::pout << " (done)" << std::endl;
	}
}


template<int dim>
void
Fluid<dim>::
dual_solve_slab_problem(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &z
) {
	Assert(dual.L.use_count(), dealii::ExcNotInitialized());
	Assert(dual.Mzn.use_count(), dealii::ExcNotInitialized());
	Assert(dual.Je.use_count(), dealii::ExcNotInitialized());

	dual.b = std::make_shared< dealii::Vector<double> > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.b->reinit(
		slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()
	);
	*dual.b = 0.;

	dual.b->add(1., *dual.Mzn);
	dual.b->add(1., *dual.Je);

	if (std::next(slab) == grid->slabs.end())
	{
//		std::cout << "slab->t_n = " << slab->t_n << std::endl;
//		std::cout << "copying debug_L matrix" << std::endl;
		dual.debug_L_no_bc = std::make_shared< dealii::SparseMatrix<double> > ();
//		std::cout << "created pointer" << std::endl;

		dual.debug_sp = std::make_shared< dealii::SparsityPattern > ();
		dual.debug_sp->copy_from(dual.L->get_sparsity_pattern());
//		std::cout << "copied sparsity pattern into dual.debug_sp" << std::endl;
		dual.debug_L_no_bc->reinit(*dual.debug_sp);  //dual.L->get_sparsity_pattern());
////		std::cout << "reinited with sparsity pattern" << std::endl;
		dual.debug_L_no_bc->copy_from(*dual.L);
		//dual.debug_L_no_bc.add(-1.,*dual.L);
////		std::cout << "copied matrix" << std::endl;
//		std::cout << "dual.L->linfty_norm() = " << dual.L->linfty_norm() << std::endl;
//		std::cout << "dual.debug_L_no_bc.linfty_norm() = " << dual.debug_L_no_bc->linfty_norm() << std::endl;
	}

	////////////////////////////////////////////////////////////////////////////
	// apply (in)homogeneous Dirichlet boundary values
	//

	DTM::pout << "dwr-instatfluid: dealii::MatrixTools::apply_boundary_values...";
	{
		std::map<dealii::types::global_dof_index, double> boundary_values;

		// get all boundary colours on the current triangulation
		std::unordered_set< dealii::types::boundary_id > boundary_colours;
		{
			auto cell = slab->space.tria->begin_active();
			auto endc = slab->space.tria->end();
			for ( ; cell != endc; ++cell) {
			if (cell->at_boundary()) {
				// loop over all faces
				for (unsigned int face_no{0};
					face_no < dealii::GeometryInfo<dim>::faces_per_cell;
					++face_no) {
				if (cell->face(face_no)->at_boundary()) {
					auto face = cell->face(face_no);
					boundary_colours.insert(
						static_cast< unsigned int > (face->boundary_id())
					);
				}}
			}}
		}

		// process all found boundary colours
		for (auto colour : boundary_colours) {
			//////////////////////////////////////
			// component wise convection dirichlet
			//

			auto component_mask_convection =
			std::make_shared< dealii::ComponentMask > (
				(dim+1), false
			);

			switch (dim) {
			case 3:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_convection_c3)) {
					component_mask_convection->set(2, true);
				}

				// NOTE: not to break switch here is indended
				[[fallthrough]];

			case 2:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_convection_c2)) {
					component_mask_convection->set(1, true);
				}

				// NOTE: not to break switch here is indended
				[[fallthrough]];

			case 1:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_convection_c1)) {
					component_mask_convection->set(0, true);
				}

				break;

			default:
				AssertThrow(false, dealii::ExcNotImplemented());
			}

			// create boundary_values as
			// std::map<dealii::types::global_dof_index, double>
			{
				// choose temporal quadrature
				std::shared_ptr< dealii::Quadrature<1> > time_support_points;
				{
					if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss")) ) {

						time_support_points =
						std::make_shared< dealii::QGauss<1> > (
							(parameter_set->fe.dual.convection.r + 1)
						);
					}

					if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss-Lobatto")) ) {

						time_support_points =
						std::make_shared< dealii::QGaussLobatto<1> > (
							(parameter_set->fe.dual.convection.r + 1)
						);
					}
				}

//				const dealii::QGauss<1> support_points(
//					slab->time.dual.fe_info->fe->tensor_degree()+1
//				);

				auto cell_time = slab->time.dual.fe_info->dof->begin_active();
				auto endc_time = slab->time.dual.fe_info->dof->end();

				dealii::FEValues<1> time_fe_values(
					*slab->time.dual.fe_info->mapping,
					*slab->time.dual.fe_info->fe,
					*time_support_points,
					dealii::update_quadrature_points
				);

				for ( ; cell_time != endc_time; ++cell_time) {
					time_fe_values.reinit(cell_time);

					for (unsigned int qt{0}; qt < time_support_points->size(); ++qt) {
						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.dual.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							dealii::ZeroFunction<dim>(dim+1),
							boundary_values_qt,
							*component_mask_convection
						);

						// boundary_values_qt -> boundary_values
						for (auto &el : boundary_values_qt) {
							dealii::types::global_dof_index idx =
								el.first
								// time offset
								+ slab->space.dual.fe_info->dof->n_dofs() *
									(cell_time->index()
									* slab->time.dual.fe_info->fe->dofs_per_cell)
								// local in time dof
								+ slab->space.dual.fe_info->dof->n_dofs() * qt
							;

							boundary_values[idx] = el.second;
						}
					}
				}
			}

			///////////////////////////////////////////////////////////
			// prescribed_no_slip: all convection components are homog.
			//

			for (unsigned int component{0}; component < (dim+1); ++component) {
				component_mask_convection->set(component, false);
			}

			switch (dim) {
			case 3:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_no_slip)) {
					component_mask_convection->set(2, true);
				}

				// NOTE: not to break switch here is indended
				[[fallthrough]];

			case 2:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_no_slip)) {
					component_mask_convection->set(1, true);
				}

				// NOTE: not to break switch here is indended
				[[fallthrough]];

			case 1:
				if (colour & static_cast< dealii::types::boundary_id > (
					fluid::types::space::boundary_id::prescribed_no_slip)) {
					component_mask_convection->set(0, true);
				}

				break;

			default:
				AssertThrow(false, dealii::ExcNotImplemented());
			}

			// create boundary_values as
			// std::map<dealii::types::global_dof_index, double>
			{
				// choose temporal quadrature
				std::shared_ptr< dealii::Quadrature<1> > time_support_points;
				{
					if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss")) ) {

						time_support_points =
						std::make_shared< dealii::QGauss<1> > (
							(parameter_set->fe.dual.convection.r + 1)
						);
					}

					if ( !(parameter_set->
						fe.low.convection.time_type_support_points
						.compare("Gauss-Lobatto")) ) {

						time_support_points =
						std::make_shared< dealii::QGaussLobatto<1> > (
							(parameter_set->fe.dual.convection.r + 1)
						);
					}
				}

//				const dealii::QGauss<1> support_points(
//					slab->time.dual.fe_info->fe->tensor_degree()+1
//				);

				auto cell_time = slab->time.dual.fe_info->dof->begin_active();
				auto endc_time = slab->time.dual.fe_info->dof->end();

				dealii::FEValues<1> time_fe_values(
					*slab->time.dual.fe_info->mapping,
					*slab->time.dual.fe_info->fe,
					*time_support_points,
					dealii::update_quadrature_points
				);

				for ( ; cell_time != endc_time; ++cell_time) {
					time_fe_values.reinit(cell_time);

					for (unsigned int qt{0}; qt < time_support_points->size(); ++qt) {
						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.dual.fe_info->dof,
							static_cast< dealii::types::boundary_id > (
								colour
							),
							dealii::ZeroFunction<dim>(dim+1),
							boundary_values_qt,
							*component_mask_convection
						);

						// boundary_values_qt -> boundary_values
						for (auto &el : boundary_values_qt) {
							dealii::types::global_dof_index idx =
								el.first
								// time offset
								+ slab->space.dual.fe_info->dof->n_dofs() *
									(cell_time->index()
									* slab->time.dual.fe_info->fe->dofs_per_cell)
								// local in time dof
								+ slab->space.dual.fe_info->dof->n_dofs() * qt
							;

							boundary_values[idx] = el.second;
						}
					}
				} // no slip
			}
		} // for each (boundary) colour

		////////////////////////////////////////////////////////////////////////
		// MatrixTools::apply_boundary_values (number = double for A,x,b)
		// input: A sparse matrix: IR^(m,n)
		//        x vector: IR^n
		//        b vector: IR^m
		//
		//        boundary_values: map< dof, double >
		//
		// NOTE: constrained rows (and columns) are not removed
		//       (not condensed) from the sparsity pattern due to
		//       memory efficiency reasons (this avoids the expensive
		//       reallocation and copying of the operator and vectors)
		//
		// NOTE: a spurious, but nicely scaled, singular value is introduced
		//       in the operator matrix for each boundary value constraint
		//
		if (boundary_values.size()) {
			////////////////////////////////////////////////////////////////////
			// prepare function header
			std::shared_ptr< dealii::SparseMatrix<double> > A = dual.L;
			std::shared_ptr< dealii::Vector<double> > x = z->x[0];
			std::shared_ptr< dealii::Vector<double> > b = dual.b;

			////////////////////////////////////////////////////////////////////
			// internal checks
			Assert(
				A->m(),
				dealii::ExcMessage(
					"empty operator matrix A (rows: A.m() == 0)"
				)
			);

			Assert(
				A->n(),
				dealii::ExcMessage(
					"empty operator matrix A (cols: A.n() == 0)"
				)
			);

			Assert(
				(A->n() == x->size()),
				dealii::ExcMessage(
					"Internal dimensions error: "
					"A->n() =/= x->size() of linear system A x = b"
				)
			);

			Assert(
				(A->m() == b->size()),
				dealii::ExcMessage(
					"Internal dimensions error: "
					"A->m() =/= b->size() of linear system A x = b"
				)
			);

			////////////////////////////////////////////////////////////////////
			// apply boundary values to operator matrix A, vectors x and b

			// preparation: eliminate constrained rows of operator A completely
			for (auto &boundary_value : boundary_values) {
				// on-the-fly check:
				// validity of the given boundary value constraints
				Assert(
					(boundary_value.first < A->m()),
					dealii::ExcMessage(
						"constraining of dof index (as boundary value) "
						"larger than (A.m()-1) is not valid"
					)
				);

				// eliminate constrained row of operator matrix completely
				auto el_in_row_i{A->begin(boundary_value.first)};
				auto end_el_in_row_i{A->end(boundary_value.first)};

				for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
					el_in_row_i->value() = double(0.);
				}
			}

			// find (absolute) maximum of all remaining diagonal elements
			// as scaling parameter for the boundary values identity operator
			//
			// NOTE: this is (a bit) more expensive then simply using the
			//       first non-zero diagonal element (for the case that in
			//       a constrained row the diagonal element is zero).
			//       The advantage is, that the spurious singular values of
			//       all boundary value constraints are identical, i.e.
			//       a fixed and nicely scaled value.
			//
//			double diagonal_scaling_value{0.};
//
//			for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
//				if (std::abs(A->el(i,i)) > std::abs(diagonal_scaling_value)) {
//					diagonal_scaling_value = A->el(i,i);
//				}
//			}
//
//			if (diagonal_scaling_value == double(0.)) {
//				diagonal_scaling_value = double(1.);
//			}
//
//			Assert(
//				(diagonal_scaling_value != double(0.)),
//				dealii::ExcInternalError()
//			);
			double diagonal_scaling_value{1.};

			////////////////////////////////////////////////////////////////////
			// apply boundary values:
			// to the linear system (A x = b)
			// without eliminating the corresponding column entries from
			// the operator matrix A
			//
			for (auto &boundary_value : boundary_values) {
				// set scaled identity operator
				A->set(
					boundary_value.first,  // i
					boundary_value.first,  // i
					diagonal_scaling_value // scaling factor or 1
				);

				// set constrained solution vector component (for iterative lss)
				(*x)[boundary_value.first] = boundary_value.second;

				// set constrained right hand side vector component
				(*b)[boundary_value.first] =
					diagonal_scaling_value * boundary_value.second;
			}

 			////////////////////////////////////////////////////////////////////
 			// eliminate constrained column entries
 			//
 			// NOTE: this is quite expensive, but helps iterative lss
 			//       since the boundary value entries are shifted to the
 			//       right hand side.
 			//
 			// NOTE: there is no symmetry assumption on the sparsity pattern,
 			//       which is necessary for space-time operators
 			//
 			for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
 				// if the row i of the operator A is not constrained,
 				// check if constrained columns need to be eliminated
 				if (boundary_values.find(i) == boundary_values.end()) {
 					// row i of A is not constrained
 					auto el_in_row_i{A->begin(i)};
 					auto end_el_in_row_i{A->end(i)};

 					// check if a_ij needs to be eliminated
 					for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
 						// get iterator of a_ij
 						auto boundary_value =
 							boundary_values.find(el_in_row_i->column());

 						// if a_ij is constrained
 						if (boundary_value != boundary_values.end()) {
 							// shift constraint to rhs
 							(*b)[i] -=
 								el_in_row_i->value()*boundary_value->second;

 							// eliminate a_ij
 							el_in_row_i->value() = 0.;
 						}
 					}
 				}
 			}
		}
	}
	DTM::pout << " (done)" << std::endl;

	if (std::next(slab) == grid->slabs.end())
	{
////		std::cout << "copying debug_L matrix" << std::endl;
//		//dual.debug_L_no_bc = std::make_shared< dealii::SparseMatrix<double> > ();
////		std::cout << "created pointer" << std::endl;
//		dual.debug_L_bc.reinit(dual.L->get_sparsity_pattern());
//////		std::cout << "reinited with sparsity pattern" << std::endl;
//		//dual.debug_L_bc.copy_from(*dual.L);
//		dual.debug_L_bc.add(1.,*dual.L);
//////		std::cout << "copied matrix" << std::endl;
//		std::cout << "dual.L->linfty_norm() = " << dual.L->linfty_norm() << std::endl;
//		std::cout << "dual.debug_L_bc.linfty_norm() = " << dual.debug_L_bc.linfty_norm() << std::endl;

//		std::ofstream out("dual_matrix.txt", std::ios_base::out);
//		dual.L->print(out);
//		out.close();

		dual.debug_L_bc = std::make_shared< dealii::SparseMatrix<double> > ();
		dual.debug_L_bc->reinit(*dual.debug_sp);
		dual.debug_L_bc->copy_from(*dual.L);
	}

	////////////////////////////////////////////////////////////////////////////
	// condense hanging nodes in system matrix, if any
	//
	slab->spacetime.dual.constraints->condense(*dual.L);

	////////////////////////////////////////////////////////////////////////////
	// solve linear system with direct solver
	//

	DTM::pout << "dwr-instatfluid: setup direct lss and solve...";

	if (!parameter_set->problem.compare("Navier-Stokes") || !parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") || slab->t_n == parameter_set->time.fluid.T)
		dual.iA.initialize(*dual.L);
	dual.iA.vmult(*z->x[0], *dual.b);

	DTM::pout << " (done)" << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// distribute hanging nodes constraints on solution
	//

	slab->spacetime.dual.constraints->distribute(
		*z->x[0]
	);
}


template<int dim>
void
Fluid<dim>::
dual_do_backward_TMS(
	const unsigned int dwr_loop,
	bool last
) {
	////////////////////////////////////////////////////////////////////////////
	// prepare time marching scheme (TMS) loop
	//
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to last space-time slab: Omega x I_N
	//

	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = std::prev(grid->slabs.end());

	////////////////////////////////////////////////////////////////////////////
	// storage: init iterators to storage_data_vectors
	//          corresponding to last space-time slab: Omega x I_N
	//

	Assert(dual.storage.z.use_count(), dealii::ExcNotInitialized());
	Assert(dual.storage.z->size(), dealii::ExcNotInitialized());
	auto z = std::prev(dual.storage.z->end());

	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	auto u = std::prev(primal.storage.u->end());

	Assert(primal.storage.um.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.um->size(), dealii::ExcNotInitialized());
	auto um = std::prev(primal.storage.um->end());

	// error indicators
	Assert(error_estimator.storage.eta_space.use_count(), dealii::ExcNotInitialized());
	Assert(error_estimator.storage.eta_space->size(), dealii::ExcNotInitialized());
	auto eta_space = std::prev(error_estimator.storage.eta_space->end());

	Assert(error_estimator.storage.eta_time.use_count(), dealii::ExcNotInitialized());
	Assert(error_estimator.storage.eta_time->size(), dealii::ExcNotInitialized());
	auto eta_time = std::prev(error_estimator.storage.eta_time->end());

	////////////////////////////////////////////////////////////////////////////
	// interpolate (or project) final condition
	//

	// NOTE: z(T) = 0 for mean drag functional

	////////////////////////////////////////////////////////////////////////////
	// do TMS loop
	//

	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< "dual: solving backward TMS problem..." << std::endl
		<< std::endl;

	Assert(grid.use_count(), dealii::ExcNotInitialized());
	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	unsigned int n{N};

	while (n) {
		DTM::pout
			<< "dual: solving problem on "
			<< "Q_" << n
			<< " = Omega_h x (" << slab->t_m << ", " << slab->t_n << ") "
			<< std::endl;

		// if (primal_order != dual_order): init dual_order components
		if ( parameter_set->fe.primal_order.compare(parameter_set->fe.dual_order) )
		{
			if ( !parameter_set->fe.dual_order.compare("low") )
			{
				grid->initialize_low_grid_components_on_slab(slab);
				grid->distribute_low_on_slab(slab);
			}
			else if ( !parameter_set->fe.dual_order.compare("high") || !parameter_set->fe.dual_order.compare("high-time") )
			{
				grid->initialize_high_grid_components_on_slab(slab);
				grid->distribute_high_on_slab(slab);
			}
			else
				AssertThrow(false, dealii::ExcNotImplemented());
		}
		else // (primal_order == dual_order): reinit the remaining order (high or low), since it is needed for error estimation
		{
			if ( !parameter_set->fe.dual_order.compare("low") )
			{
				// (dual == low): init high for error estimator
				grid->initialize_high_grid_components_on_slab(slab);
				grid->distribute_high_on_slab(slab);
			}
			else if ( !parameter_set->fe.dual_order.compare("high") || !parameter_set->fe.dual_order.compare("high-time") )
			{
				// (dual == high or dual == high-time): init low for error estimator
				grid->initialize_low_grid_components_on_slab(slab);
				grid->distribute_low_on_slab(slab);
			}
			else
				AssertThrow(false, dealii::ExcNotImplemented());
		}

		grid->create_sparsity_pattern_dual_on_slab(slab);
		dual_reinit_storage_on_slab(slab, z);

		{
			// error indicators
			grid->initialize_pu_grid_components_on_slab(slab);
			grid->distribute_pu_on_slab(slab);

			eta_reinit_storage_on_slab(
				slab,
				eta_space,
				eta_time
			);

			error_estimator.pu_dwr = std::make_shared< fluid::cGp_dGr::cGq_dGs::ErrorEstimator<dim> > ();
			// set the important variables for the error estimator
			error_estimator.pu_dwr->init(
				function.viscosity,
				function.convection.dirichlet,
				grid,
				parameter_set->fe.symmetric_stress,
				parameter_set->dwr.replace_linearization_points,
				parameter_set->dwr.replace_weights,
				parameter_set->fe.primal_order,
				parameter_set->fe.dual_order,
				(parameter_set->problem.compare("Navier-Stokes") == 0)
			);
		}


		if (n == N) {
			////////////////////////////////////////////////////////////////////////////
			// interpolate (or project) initial value(s)
			//

			dual.zn = std::make_shared< dealii::Vector<double> > ();
			dual.zn->reinit(slab->space.dual.fe_info->dof->n_dofs());
			*dual.zn = 0.;

			Assert(slab->space.dual.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			Assert(dual.zn.use_count(), dealii::ExcNotInitialized());

			dealii::VectorTools::interpolate(
				*slab->space.dual.fe_info->mapping,
				*slab->space.dual.fe_info->dof,
				dealii::ZeroFunction<dim>(dim+1),
				*dual.zn
			);

			// NOTE: after the first dwr-loop the initial triangulation could have
			//       hanging nodes. Therefore,
			// distribute hanging node constraints to make the result continuous again:
			slab->space.dual.fe_info->constraints->distribute(
				*dual.zn
			);
		}
		else {
			// not the last slab: transfer zm solution to zn solution
			Assert(dual.zm.use_count(), dealii::ExcNotInitialized());

			dual.zn = std::make_shared< dealii::Vector<double> > ();
			dual.zn->reinit(slab->space.dual.fe_info->dof->n_dofs());
			*dual.zn = 0.;

			// for n < N interpolate between two (different) spatial meshes
			// the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n+1}:
				*std::next(slab)->space.dual.fe_info->dof,
				*dual.zm,
				// solution on I_n:
				*slab->space.dual.fe_info->dof,
				*slab->space.dual.fe_info->constraints,
				*dual.zn
			);

			slab->space.dual.fe_info->constraints->distribute(
				*dual.zn
			);
		}

		// assemble slab problem

		// NOTE: for Stokes without adaptive refinement the dual system matrix needs only to be inverted on the last slab
		if (!parameter_set->problem.compare("Navier-Stokes") || !parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") || slab->t_n == parameter_set->time.fluid.T)
		{
			dual_assemble_system(slab, u->x[0]);
		}
		else
		{
			// dual system matrix doesn't need to be assembled, since it hasn't changed
			// hence also dual.iA stays the same
			dual.L = std::make_shared< dealii::SparseMatrix<double> > ();
			dual.L->reinit(*slab->spacetime.dual.sp);
			*dual.L = 0;
		}
		dual_assemble_rhs(slab);

		// solve slab problem (i.e. apply boundary values and solve for z0)
		dual_solve_slab_problem(slab, z);
		// FOR DEBUGGING: output dual space-time solution vector
		{
			int slab_number = 0;
			auto tmp_slab = grid->slabs.begin();
			while (slab != tmp_slab)
			{
				slab_number++;
				tmp_slab++;
			}

			std::ostringstream filename;
			filename << "z_" << std::setw(3) << std::setfill('0') << slab_number << ".txt";
			std::ofstream out(filename.str().c_str(), std::ios_base::out);
			z->x[0]->print(out,8,true,false);
		}
//		std::cout << "Finished dual_solve_slab_problem" << std::endl;

//		// TODO: delete -> debug
//		// checking whether J * U = J(u)
//		auto high_slab_u = std::make_shared< dealii::Vector<double> > ();
//		{
//			high_slab_u->reinit(
//				slab->space.high.fe_info->dof->n_dofs()
//				* slab->time.high.fe_info->dof->n_dofs()
//			);
//			*high_slab_u = 0.;
//
//			auto slab_u_tq  = std::make_shared< dealii::Vector<double> > ();
//			slab_u_tq->reinit(
//				slab->space.low.fe_info->dof->n_dofs()
//			);
//			*slab_u_tq = 0.;
//
//			auto high_slab_u_tq  = std::make_shared< dealii::Vector<double> > ();
//			high_slab_u_tq->reinit(
//				slab->space.high.fe_info->dof->n_dofs()
//			);
//			*high_slab_u_tq = 0.;
//
//			std::vector<double> time_qp;
//			time_qp.push_back(slab->t_m);
//			time_qp.push_back(0.5*(slab->t_m+slab->t_n));
//			time_qp.push_back(slab->t_n);
//
//			unsigned int ii = 0;
//			for (auto t_q : time_qp)//{slab->t_m, 0.5*(slab->t_m+slab->t_n), slab->t_n})
//			{
//				for (dealii::types::global_dof_index i{0}; i < slab->space.low.fe_info->dof->n_dofs(); ++i)
//					if (ii == 0)
//						(*slab_u_tq)[i] = (*u->x[0])[i + slab->space.low.fe_info->dof->n_dofs() * 0];
//					else if (ii == 1)
//						(*slab_u_tq)[i] =  0.5 * (*u->x[0])[i + slab->space.low.fe_info->dof->n_dofs() * 0] + 0.5 * (*u->x[0])[i + slab->space.low.fe_info->dof->n_dofs() * 1];
//					else if (ii == 2)
//						(*slab_u_tq)[i] = (*u->x[0])[i + slab->space.low.fe_info->dof->n_dofs() * 1];
//
//				// use interpolation in space to go from slab_w_tq to high_slab_w_tq
//				dealii::FETools::interpolate(
//					// low solution
//					*slab->space.low.fe_info->dof,
//					*slab_u_tq,
//					// high solution
//					*slab->space.high.fe_info->dof,
//					*slab->space.high.fe_info->constraints,
//					*high_slab_u_tq
//				);
//
//				// write high_slab_w_tq into high_slab_w
//				for (dealii::types::global_dof_index i{0}; i < slab->space.high.fe_info->dof->n_dofs(); ++i)
//					(*high_slab_u)[i + slab->space.high.fe_info->dof->n_dofs() * ii] = (*high_slab_u_tq)[i];
//		//			std::cout << "filled high_slab_u" << std::endl;
//				ii++;
//			}
//		}
//		std::cout << (*dual.Je) * (*high_slab_u) << std::endl;
//		// END of debug

		////////////////////////////////////
		// error estimation with PU-DWR
		//
		// NOTE: to estimate the error we possibly need to extrapolate z in time
		// to extrapolate in time, we also need z(t_m^-) which is from the solution from the last time slab
		// hence we always lag one slab with the error estimation
		//
		Assert(error_estimator.pu_dwr.use_count(), dealii::ExcNotInitialized());
		// evaluate error on next slab
		if (n < N)
		{
			// pass dual.L to error estimator -> only for debugging! TODO: delete this
//			std::cout << "calling estimate with linfty norms: " << dual.debug_L_no_bc->linfty_norm() << std::endl;
			error_estimator.pu_dwr->estimate_on_slab(dual.debug_L_no_bc, dual.debug_L_bc, std::next(slab), std::next(u), std::next(um), std::next(z), std::next(eta_space), std::next(eta_time));
//			std::cout << "finished estimate_on_slab" << std::endl;

			// apply B. Endtmayer's post processing of the error indicators
			// see https://arxiv.org/pdf/1811.07586.pdf (Figure 1)
			for (auto line : std::next(slab)->space.pu.fe_info->constraints->get_lines())
			{
				// spatial error indicators
				for (unsigned int i=0; i<std::pow(2, dim-1); ++i)
					(*std::next(eta_space)->x[0])[line.entries[i].first] += (1. / std::pow(2, dim-1)) * (*std::next(eta_space)->x[0])[line.index];
				(*std::next(eta_space)->x[0])[line.index] = 0.;

				// temporal error indicators
				for (unsigned int i=0; i<std::pow(2, dim-1); ++i)
					(*std::next(eta_time)->x[0])[line.entries[i].first] += (1. / std::pow(2, dim-1)) * (*std::next(eta_time)->x[0])[line.index];
				(*std::next(eta_time)->x[0])[line.index] = 0.;
			}
//			std::cout << "post processed error estimator" << std::endl;
		}
		// evaluate error on first slab
		if (n == 1)
		{
			// pass dual.L to error estimator -> only for debugging! TODO: delete this
			error_estimator.pu_dwr->estimate_on_slab(dual.debug_L_no_bc, dual.debug_L_bc, slab, u, um, z, eta_space, eta_time);

			// apply B. Endtmayer's post processing of the error indicators
			// see https://arxiv.org/pdf/1811.07586.pdf (Figure 1)
			for (auto line : slab->space.pu.fe_info->constraints->get_lines())
			{
				// spatial error indicators
				for (unsigned int i=0; i<std::pow(2, dim-1); ++i)
					(*eta_space->x[0])[line.entries[i].first] += (1. / std::pow(2, dim-1)) * (*eta_space->x[0])[line.index];
				(*eta_space->x[0])[line.index] = 0.;

				// temporal error indicators
				for (unsigned int i=0; i<std::pow(2, dim-1); ++i)
					(*eta_time->x[0])[line.entries[i].first] += (1. / std::pow(2, dim-1)) * (*eta_time->x[0])[line.index];
				(*eta_time->x[0])[line.index] = 0.;
			}
		}

		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//

		// evaluate solution z(t_m)
		dual.zm = std::make_shared< dealii::Vector<double> > ();
		dual.zm->reinit(slab->space.dual.fe_info->dof->n_dofs());
		*dual.zm = 0.;

		{
			dealii::FEValues<1> fe_face_values_time(
				*slab->time.dual.fe_info->mapping,
				*slab->time.dual.fe_info->fe,
				dealii::QGaussLobatto<1>(2),
				dealii::update_values
			);

			auto cell_time = slab->time.dual.fe_info->dof->begin_active();
			auto last_cell_time = cell_time;
			auto endc_time = slab->time.dual.fe_info->dof->end();

			for ( ; cell_time != endc_time; ++cell_time) {
				last_cell_time=cell_time;
			}

			cell_time=last_cell_time;
			{
				Assert((cell_time!=endc_time), dealii::ExcInternalError());
				fe_face_values_time.reinit(cell_time);

				// evaluate solution for t_m of Q_n
				for (unsigned int jj{0};
					jj < slab->time.dual.fe_info->fe->dofs_per_cell; ++jj)
				for (dealii::types::global_dof_index i{0};
					i < slab->space.dual.fe_info->dof->n_dofs(); ++i) {
					(*dual.zm)[i] += (*z->x[0])[
						i
						// time offset
						+ slab->space.dual.fe_info->dof->n_dofs() *
							(cell_time->index() * slab->time.dual.fe_info->fe->dofs_per_cell)
						// local in time dof
						+ slab->space.dual.fe_info->dof->n_dofs() * jj
					] * fe_face_values_time.shape_value(jj,0);
				}
			}
		}

		// output data
		dual_do_data_output(slab, z, dwr_loop, last);
		if (n < N)
			eta_do_data_output(std::next(slab), std::next(eta_space), std::next(eta_time), dwr_loop, last);
		if (n == 1)
			eta_do_data_output(slab, eta_space, eta_time, dwr_loop, last);

		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//

		dual.L = nullptr;
		dual.b = nullptr;

		dual.Mzn = nullptr;
		dual.Je = nullptr;

		grid->clear_dual_on_slab(slab);


		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//

		--n;
		--slab;
		--z;
		--u;
		--um;
		--eta_space;
		--eta_time;

		DTM::pout << std::endl;
	}

	DTM::pout
		<< "dual: backward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;

	////////////////////////////////////////////////////////////////////////////
	// allow garbage collector to clean up memory
	//

	dual.zm = nullptr;
	dual.zn = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// dual data output
//

template<int dim>
void
Fluid<dim>::
dual_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	// set up which dwr loop(s) are allowed to make data output:
	if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
		return;
	}

	// may output data: initialise (mode: all, last or specific dwr loop)
	DTM::pout
		<< "dual solution data output: patches = "
		<< parameter_set->data_output.dual.patches
		<< std::endl;

	////////////////////////////////////////////////////////////////////////////
	// INIT DATA POSTPROCESSOR
	unsigned int output_quantities(0);

	output_quantities |=
		static_cast<unsigned int>(fluid::OutputQuantities::convection);

	output_quantities |=
		static_cast<unsigned int>(fluid::OutputQuantities::pressure);

	dual.data_postprocessor = std::make_shared<
		fluid::DataPostprocessor<dim> > (
		output_quantities
	);

	////////////////////////////////////////////////////////////////////////////
	// INIT DATA OUTPUT
	//

	dual.data_output = std::make_shared< DTM::DataOutput<dim> >();

	std::vector<std::string> data_field_names;
	for (unsigned int i{0}; i < dim; ++i)
		data_field_names.push_back("convection");
	data_field_names.push_back("pressure");

	dual.data_output->set_data_field_names(data_field_names);

	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	for (unsigned int i{0}; i < dim; ++i)
		dci_field.push_back(dealii::DataComponentInterpretation::component_is_part_of_vector);
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

	dual.data_output->set_data_component_interpretation_field(dci_field);

	dual.data_output->set_data_output_patches(
		parameter_set->data_output.dual.patches
	);

	// check if we use a fixed trigger interval, or, do output once on a I_n
	if ( !parameter_set->data_output.dual.trigger_type.compare("fixed") ) {
		dual.data_output_trigger_type_fixed = true;
	}
	else {
		dual.data_output_trigger_type_fixed = false;
	}

	// only for fixed
	dual.data_output_trigger = parameter_set->data_output.dual.trigger;

	if (dual.data_output_trigger_type_fixed) {
		DTM::pout
			<< "dual solution data output: using fixed mode with trigger = "
			<< dual.data_output_trigger
			<< std::endl;
	}
	else {
		DTM::pout
			<< "dual solution data output: using I_n mode (trigger adapts to I_n automatically)"
			<< std::endl;
	}

	dual.data_output_time_value = parameter_set->time.fluid.T;
}


template<int dim>
void
Fluid<dim>::
dual_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &z,
	const unsigned int dwr_loop) {
	// TODO: might need to be debugged; adapted from primal_do_data_output_on_slab()

	// triggered output mode
	Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.dual.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	dual.data_output->set_DoF_data(
		slab->space.dual.fe_info->dof,
		slab->space.dual.fe_info->partitioning_locally_owned_dofs
	);

 	auto z_trigger = std::make_shared< dealii::Vector<double> > ();
 	z_trigger->reinit(
 		slab->space.dual.fe_info->dof->n_dofs()
 	);

	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.dual.fe_info->mapping,
			*slab->time.dual.fe_info->fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);

		Assert(
			slab->time.dual.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		auto cell_time = slab->time.dual.fe_info->dof->begin_active();
		auto endc_time = slab->time.dual.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_face_values_time.reinit(cell_time);

			////////////////////////////////////////////////////////////////////
			// construct quadrature for data output,
			// if triggered output time values are inside this time element
			//

			auto t_m = fe_face_values_time.quadrature_point(0)[0];
			auto t_n = fe_face_values_time.quadrature_point(1)[0];
			auto tau = t_n-t_m;

			std::list<double> output_times;

			if (dual.data_output_time_value < t_m) {
				dual.data_output_time_value = t_m;
			}

			for ( ; (dual.data_output_time_value <= t_n) ||
				(
					(dual.data_output_time_value > t_n) &&
					(std::abs(dual.data_output_time_value - t_n) < tau*1e-12)
				); ) {
				output_times.push_back(dual.data_output_time_value);
				dual.data_output_time_value += dual.data_output_trigger;
			}

			if (output_times.size() && output_times.back() > t_n) {
				output_times.back() = t_n;
			}

			if ((output_times.size() > 1) &&
				(output_times.back() == *std::next(output_times.rbegin()))) {
				// remove the last entry, iff doubled
				output_times.pop_back();
			}

			// convert container
			if (!output_times.size()) {
				continue;
			}

			std::vector< dealii::Point<1> > output_time_points(output_times.size());
			{
				auto time{output_times.begin()};
				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
					double t_trigger{*time};
					output_time_points[q][0] = (t_trigger-t_m)/tau;
				}

				if (output_time_points[0][0] < 0) {
					output_time_points[0][0] = 0;
				}

				if (output_time_points[output_time_points.size()-1][0] > 1) {
					output_time_points[output_time_points.size()-1][0] = 1;
				}
			}

			dealii::Quadrature<1> quad_time(output_time_points);

			// create fe values
			dealii::FEValues<1> fe_values_time(
				*slab->time.dual.fe_info->mapping,
				*slab->time.dual.fe_info->fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);

			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.dual.fe_info->fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);

			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*z_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.dual.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.dual.fe_info->dof->n_dofs(); ++i) {
 					(*z_trigger)[i] += (*z->x[0])[
 						i
 						// time offset
 						+ slab->space.dual.fe_info->dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

//				std::cout
//					<< "dual output generated for t = "
//					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
//					<< std::endl;

				dual.data_output->write_data(
					filename.str(),
					z_trigger,
					dual.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
dual_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &z,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.dual.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	dual.data_output->set_DoF_data(
		slab->space.dual.fe_info->dof,
		slab->space.dual.fe_info->partitioning_locally_owned_dofs
	);

	auto z_trigger = std::make_shared< dealii::Vector<double> > ();
 	z_trigger->reinit(
 		slab->space.dual.fe_info->dof->n_dofs()
 	);

	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// choose temporal quadrature
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time;
		{
			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss")) ) {

				fe_quad_time =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.dual.convection.r + 1)
				);
			}

			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_time =
				std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.dual.convection.r + 1)
				);
			}
		}

		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.dual.fe_info->mapping,
			*slab->time.dual.fe_info->fe,
			*fe_quad_time,
			dealii::update_values |
			dealii::update_quadrature_points
		);

		auto cell_time = slab->time.dual.fe_info->dof->begin_active();
		auto endc_time = slab->time.dual.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.dual.fe_info->fe->dofs_per_cell
			);

			cell_time->get_dof_indices(local_dof_indices_time);

			for (
				unsigned int qt{0};
				qt < fe_values_time.n_quadrature_points;
				++qt) {
 				*z_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.dual.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.dual.fe_info->dof->n_dofs(); ++i) {
 					(*z_trigger)[i] += (*z->x[0])[
 						i
 						// time offset
 						+ slab->space.dual.fe_info->dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0] + (qt == 0) * 1e-5
// 					<< std::endl;

				dual.data_output->write_data(
					filename.str(),
					z_trigger,
					dual.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] + (qt == 0) * 1e-5 // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
dual_do_data_output(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &z,
	const unsigned int dwr_loop,
	bool last
) {
	if (dual.data_output_trigger <= 0) return;

	// set up which dwr loop(s) are allowed to make data output:
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	if ( !parameter_set->data_output.dual.dwr_loop.compare("none") ) {
		return;
	}

	if (!parameter_set->data_output.dual.dwr_loop.compare("last")) {
		// output only the last (final) dwr loop
		if (last) {
			dual.data_output_dwr_loop = dwr_loop;
		}
		else {
			return;
		}
	}
	else {
		if (!parameter_set->data_output.dual.dwr_loop.compare("all")) {
			// output all dwr loops
			if (!last) {
				dual.data_output_dwr_loop = dwr_loop;
			}
			else {
				return;
			}
		}
		else {
			// output on a specific dwr loop
			if (!last) {
				dual.data_output_dwr_loop =
					std::stoi(parameter_set->data_output.dual.dwr_loop)-1;
			}
			else {
				return;
			}
		}
	}

	if (dual.data_output_dwr_loop < 0)
		return;

	if ( static_cast<unsigned int>(dual.data_output_dwr_loop) != dwr_loop )
		return;

	if (!dual.data_output_trigger_type_fixed) {
		// I_n output mode (output on natural Q_n support points in time)
		dual_do_data_output_on_slab_Qn_mode(slab, z, dwr_loop);
	}
	else {
		// fixed trigger output mode
		dual_do_data_output_on_slab(slab, z, dwr_loop);
	}
}

template<int dim>
void
Fluid<dim>::
dual_sort_xdmf_by_time(
		const unsigned int dwr_loop
) {
	// name of xdmf file to be sorted by time of the snapshots
	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop
		<< ".xdmf";

	// read the current xdmf file and save it line by line in a std::vector
	std::ifstream old_xdmf_file(filename.str());
	if (!old_xdmf_file.good()) // exit this function, if there is no xdmf file
		return;

	std::vector< std::string > old_lines;

	std::string current_line;
	while (getline (old_xdmf_file, current_line))
		old_lines.push_back(current_line);

	old_xdmf_file.close();

	// create a std::vector for the new lines where the grids are sorted in ascending order by the time
	std::vector< std::string > new_lines;

	// the first 5 lines remain unchanged
	for (unsigned int i = 0; i <= 4; ++i)
		new_lines.push_back(old_lines[i]);

	//////////////////////////////////////////
	// reorder the grids by time
	//
	std::vector< std::vector< std::string > > grids;
	std::vector< std::string > current_grid;

	// each grid consists of 23 lines
	for (unsigned int i = 5; i < old_lines.size()-3; ++i)
	{
		current_grid.push_back(old_lines[i]);
		if (current_grid.size() == 23)
		{
			grids.push_back(current_grid);
			current_grid.clear();
		}
	}

	// get time of grid
    auto get_time = [](std::vector< std::string > grid)
    {
    	Assert(grid.size() == 23, dealii::ExcInvalidState());
        // the second line contains the time
    	std::string line = grid[1];
    	// get the number between the double quotes
        std::string delimiter = "\"";
        std::string token = line.substr(line.find(delimiter)+1, line.size());
        token = token.substr(0, token.find(delimiter));
        return std::stod(token);
    };

    // sort the grids by time
    std::sort(std::begin(grids),
              std::end(grids),
              [&](std::vector< std::string > & a, std::vector< std::string > & b)
	{
    	return get_time(a) < get_time(b);
	});

    // append sorted grid to new_lines
    for (auto grid : grids)
    {
    	for (auto line : grid)
    	{
    		new_lines.push_back(line);
    	}
    }

	// the last 3 lines remain unchanged
	for (unsigned int i = old_lines.size()-3; i < old_lines.size(); ++i)
	{
		new_lines.push_back(old_lines[i]);
	}

	// write to file
	std::ofstream new_xdmf_file;
	new_xdmf_file.open(filename.str());
	for (auto line : new_lines)
		new_xdmf_file << line << std::endl;
	new_xdmf_file.close();
}


////////////////////////////////////////////////////////////////////////////////
// functional values
//
template<int dim>
void
Fluid<dim>::
compute_functional_values(
		const typename DTM::types::storage_data_vectors<1>::iterator &u,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	Assert(dim==2, dealii::ExcNotImplemented());
	
	// un := u at quadrature point, i.e. un = u(.,t_n)
	std::shared_ptr< dealii::Vector<double> > un = std::make_shared< dealii::Vector<double> >(
			slab->space.primal.fe_info->dof->n_dofs()
	);

	// create vectors with functional values at quadrature points
	// format: (time, value)
	std::vector< std::tuple< double, double > > pressure_values;
	std::vector< std::tuple< double, double > > drag_values;
	std::vector< std::tuple< double, double > > lift_values;

	auto cell_time = slab->time.primal.fe_info->dof->begin_active();
	auto endc_time = slab->time.primal.fe_info->dof->end();

	std::shared_ptr< dealii::Quadrature<1> > quad_time;
	{
		if ( !(parameter_set->
			fe.low.convection.time_type_support_points
			.compare("Gauss")) ) {
			quad_time =
			std::make_shared< dealii::QGauss<1> > (
				(parameter_set->fe.primal.convection.r + 1)
			);
		} else if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ){
			if (parameter_set->fe.low.convection.r < 1){
				quad_time = std::make_shared< QRightBox<1> > ();
			} else {
				quad_time =
						std::make_shared< dealii::QGaussLobatto<1> > (
								(parameter_set->fe.low.convection.r + 1)
						);
			}
		}
	}

	dealii::FEValues<1> fe_values_time(
		*slab->time.primal.fe_info->mapping,
		*slab->time.primal.fe_info->fe,
		*quad_time,
		dealii::update_quadrature_points | dealii::update_JxW_values
	);

	std::vector< dealii::types::global_dof_index > time_local_dof_indices(
		slab->time.primal.fe_info->fe->dofs_per_cell
	);

	for ( ; cell_time != endc_time; ++cell_time) {
		fe_values_time.reinit(cell_time);
		cell_time->get_dof_indices(time_local_dof_indices);

		for (unsigned int ii{0}; ii < slab->time.primal.fe_info->fe->dofs_per_cell; ++ii)
		{
			un->reinit(slab->space.primal.fe_info->dof->n_dofs());

			double tn = fe_values_time.quadrature_point(ii)[0];

			// compute solution un at time tn
			for (dealii::types::global_dof_index i{0}; i < slab->space.primal.fe_info->dof->n_dofs(); ++i)
			{
				(*un)[i] += (*u->x[0])[
					i
					// time offset
					+ slab->space.primal.fe_info->dof->n_dofs() *
						(cell_time->index() * slab->time.primal.fe_info->fe->dofs_per_cell)
					// local in time dof
					+ slab->space.primal.fe_info->dof->n_dofs() * ii
				];
			}

			////////////////////////////////////
			// compute functional values of un

			////////////////////////////////////
			// pressure
			dealii::Point<dim> M;

			if (dim==2) {
				M[0] = 0.15;
				M[1] = 0.20;
			}

			double pressure_front = compute_pressure(
				M,
				un,
				slab
			); // pressure - left  point on circle

			if (dim==2) {
				M[0] = 0.25;
				M[1] = 0.20;
			}

			double pressure_back = compute_pressure(
				M,
				un,
				slab
			); // pressure - right point on circle

			// save pressure difference to the vector pressure_values
			double pressure_diff = pressure_front - pressure_back;
			pressure_values.push_back(std::make_tuple(tn, pressure_diff));

			////////////////////////////////////
			// drag and lift

			// Compute drag and lift via line integral
			dealii::Tensor<1, dim> drag_lift_value;
			compute_drag_lift_tensor(
				un,
				slab,
				drag_lift_value
			);
			drag_values.push_back(std::make_tuple(tn, drag_lift_value[0]));
			lift_values.push_back(std::make_tuple(tn, drag_lift_value[1]));

			// update space-time goal functionals
			double scaling = 1. / (parameter_set->time.fluid.T - parameter_set->time.fluid.t0);
			error_estimator.goal_functional.fem.mean_drag += scaling * drag_lift_value[0] * fe_values_time.JxW(ii);
			error_estimator.goal_functional.fem.mean_lift += scaling * drag_lift_value[1] * fe_values_time.JxW(ii);
		}
	}

	//////////////////////////////////////////////////////
	// output goal functionals

	DTM::pout
			<< "------------------"
			<< std::endl;

	// pressure
	DTM::pout << "Pressure difference:" << std::endl;

	std::ofstream p_out;
	// append instead pressure difference to text file (ios_base::app):
	p_out.open("pressure.log", std::ios_base::app);

	for (auto &item : pressure_values)
	{
		DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
		DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

		// save to txt file
		p_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
	}
	p_out.close();

	// drag
	DTM::pout << "Face drag:" << std::endl;

	std::ofstream d_out;
	d_out.open("drag.log", std::ios_base::app);

	for (auto &item : drag_values)
	{
		DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
		DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

		// save to txt file
		d_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
	}
	d_out.close();

	// lift
	DTM::pout << "Face lift:" << std::endl;

	std::ofstream l_out;
	l_out.open("lift.log", std::ios_base::app);

	for (auto &item : lift_values)
	{
		DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
		DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

		// save to txt file
		l_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
	}
	l_out.close();
}

template<int dim>
double
Fluid<dim>::
compute_pressure(
	dealii::Point<dim> x,
	std::shared_ptr< dealii::Vector<double> > un,
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// evaluate the fe system solution as x_h for a given point x
	dealii::Vector<double> x_h(dim + 1);
	
	dealii::VectorTools::point_value(
		*slab->space.primal.fe_info->dof,
		*un, // input dof vector at t_n
		x, // evaluation point
		x_h
	);
	
	// return the pressure component of the fe solution evaluation
	return x_h[dim];
}

template<int dim>
void
Fluid<dim>::
compute_drag_lift_tensor(
		std::shared_ptr< dealii::Vector<double> > un,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		dealii::Tensor<1, dim> &drag_lift_value
) {
	const dealii::QGauss<dim - 1> face_quadrature_formula(
		std::max(
			std::max(
				slab->space.primal.fe_info->fe->base_element(0).base_element(0).tensor_degree(),
				slab->space.primal.fe_info->fe->base_element(0).base_element(1).tensor_degree()
			),
			static_cast<unsigned int> (1)
		) + 4
	);
	dealii::FEFaceValues<dim> fe_face_values(*slab->space.primal.fe_info->fe, face_quadrature_formula,
									 dealii::update_values | dealii::update_gradients | dealii::update_normal_vectors |
										 dealii::update_JxW_values | dealii::update_quadrature_points);

	const unsigned int dofs_per_cell = slab->space.primal.fe_info->fe->dofs_per_cell;
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	std::vector<unsigned int> local_dof_indices(dofs_per_cell);
	std::vector<dealii::Vector<double>> face_solution_values(n_face_q_points,
													 dealii::Vector<double>(dim + 1));

	std::vector<std::vector<dealii::Tensor<1, dim>>>
		face_solution_grads(n_face_q_points, std::vector<dealii::Tensor<1, dim>>(dim + 1));

	typename dealii::DoFHandler<dim>::active_cell_iterator
		cell = slab->space.primal.fe_info->dof->begin_active(),
		endc = slab->space.primal.fe_info->dof->end();

	for (; cell != endc; ++cell)
	{
	  // First, we are going to compute the forces that
	  // act on the cylinder. We notice that only the fluid
	  // equations are defined here.
	  for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
		if (cell->face(face)->at_boundary() &&
			cell->face(face)->boundary_id() == fluid::types::space::boundary_id::prescribed_obstacle)
		{
		  fe_face_values.reinit(cell, face);
		  fe_face_values.get_function_values(*un, face_solution_values);
		  fe_face_values.get_function_gradients(*un, face_solution_grads);

		  for (unsigned int q = 0; q < n_face_q_points; ++q)
		  {

			dealii::Tensor<2, dim> pI;
			pI.clear(); // reset all values to zero
			for (unsigned int l = 0; l < dim; l++)
			  pI[l][l] = face_solution_values[q](dim);

			dealii::Tensor<2, dim> grad_v;
			for (unsigned int l = 0; l < dim; l++)
			  for (unsigned int m = 0; m < dim; m++)
				grad_v[l][m] = face_solution_grads[q][l][m];

			double _viscosity = function.viscosity->value(
				fe_face_values.quadrature_point(q),0
			);

			dealii::Tensor<2, dim> sigma_fluid = -pI + _viscosity * grad_v;
			if (parameter_set->fe.symmetric_stress)
				sigma_fluid += _viscosity * transpose(grad_v);

			drag_lift_value -= sigma_fluid * fe_face_values.normal_vector(q) * fe_face_values.JxW(q);
		  }
		} // end boundary stokes::types::space::boundary_id::prescribed_obstacle for fluid
	} // end cell

	// 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek 1996)
	double max_velocity = function.convection.dirichlet->value(
					dealii::Point<dim>(0.0, 0.205) // maximal velocity in the middle of the boundary
	)[0];

	if (parameter_set->convection.dirichlet_boundary_function.compare("Convection_Parabolic_Inflow_3") == 0)
	{
		if (max_velocity < 0.3 + 1e-9)
				drag_lift_value *= 500.0; // 2D-1
		else
			drag_lift_value *= 20.0; // 2D-2
	}
	else if (parameter_set->convection.dirichlet_boundary_function.compare("Convection_Parabolic_Inflow_3_sin") == 0)
	{
		drag_lift_value *= 20.0; // 2D-3
	}
}

////////////////////////////////////////////////////////////////////////////////
// error estimation and space-time grid adaption
//

template<int dim>
void
Fluid<dim>::
eta_reinit_storage() {
	////////////////////////////////////////////////////////////////////////////
	// init storage containers for vector data:
	// NOTE: * eta space: time dG(0) method
	//       * eta solution dof vectors: \eta
	//

	Assert(grid.use_count(), dealii::ExcNotInitialized());

	error_estimator.storage.eta_space =
		std::make_shared< DTM::types::storage_data_vectors<1> > ();

	error_estimator.storage.eta_space->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);

	error_estimator.storage.eta_time =
		std::make_shared< DTM::types::storage_data_vectors<1> > ();

	error_estimator.storage.eta_time->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);
}

template<int dim>
void
Fluid<dim>::
eta_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta_t
) {
	// spatial error indicators
	for (unsigned int j{0}; j < eta_s->x.size(); ++j) {
		eta_s->x[j] = std::make_shared< dealii::Vector<double> > ();

		Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

		Assert(
			slab->space.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		Assert(
			slab->time.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		eta_s->x[j]->reinit(
			slab->space.pu.fe_info->dof->n_dofs() * slab->time.pu.fe_info->dof->n_dofs()
		);
	}

	// temporal error indicators
	for (unsigned int j{0}; j < eta_t->x.size(); ++j) {
		eta_t->x[j] = std::make_shared< dealii::Vector<double> > ();

		Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

		Assert(
			slab->space.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		Assert(
			slab->time.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		eta_t->x[j]->reinit(
			slab->space.pu.fe_info->dof->n_dofs() * slab->time.pu.fe_info->dof->n_dofs()
		);
	}
}

template<int dim>
void
Fluid<dim>::
compute_effectivity_index() {
	// sum up error estimator
	double value_eta_k = 0.;
	DTM::pout << "eta_k(estimator) = ";
	for (auto &element : *error_estimator.storage.eta_time)
	{
		value_eta_k += std::accumulate(element.x[0]->begin(), element.x[0]->end(), 0.);
		DTM::pout << std::accumulate(element.x[0]->begin(), element.x[0]->end(), 0.) << ",";
	}
	DTM::pout << std::endl;

	double value_eta_h = 0.;
	DTM::pout << "eta_h(estimator) = ";
	for (auto &element : *error_estimator.storage.eta_space)
	{
		value_eta_h += std::accumulate(element.x[0]->begin(), element.x[0]->end(), 0.);
		DTM::pout << std::accumulate(element.x[0]->begin(), element.x[0]->end(), 0.) << ",";
	}
	DTM::pout << std::endl;

	const double value_eta = std::abs(value_eta_k + value_eta_h);

	// true error of FEM simulation in goal functional
	double reference_goal_functional;
	if (parameter_set->problem.compare("Navier-Stokes") == 0)
		reference_goal_functional = error_estimator.goal_functional.reference.NSE.mean_drag;
	else
		reference_goal_functional = error_estimator.goal_functional.reference.Stokes.mean_drag;
	const double fem_goal_functional = error_estimator.goal_functional.fem.mean_drag;
	const double true_error = std::abs(reference_goal_functional - fem_goal_functional);

	// effectivity index
	const double I_eff = value_eta / true_error;

	DTM::pout << "\neta_k              = " << value_eta_k
			  << "\neta_h              = " << value_eta_h
			  << "\neta                = " << value_eta
			  << "\n|J(u) - J(u_{kh})| = " << true_error
			  << "\nI_eff              = " << I_eff
			  << std::endl;
}


template<int dim>
void
Fluid<dim>::
refine_and_coarsen_space_time_grid(
	const unsigned int dwr_loop
) {
	Assert(
			error_estimator.storage.eta_space->size()==grid->slabs.size(),
			dealii::ExcInternalError()
	);
	Assert(
			error_estimator.storage.eta_time->size()==grid->slabs.size(),
			dealii::ExcInternalError()
	);

	const unsigned int N{static_cast<unsigned int>(grid->slabs.size())};
	std::vector<double> eta_k(N);

	double eta_k_global {0.};
	double eta_h_global {0.};

	// compute eta^n on I_n for n=1..N as well as global estimators
	{
		auto eta_it{error_estimator.storage.eta_time->begin()};
		for (unsigned n{0}; n < N; ++n, ++eta_it) {
			Assert(
				(eta_it != error_estimator.storage.eta_time->end()),
				dealii::ExcInternalError()
			);

			double eta_k_K = std::accumulate(
				eta_it->x[0]->begin(),
				eta_it->x[0]->end(),
				0.
			);
			eta_k[n] = std::abs(eta_k_K);
			eta_k_global += eta_k_K;
		}
	}

	// Per definition eta_k[0] is 0 for the primal problem, so just set it to the next time step
	//eta_k[0] = eta_k[1];
	if (eta_k[0] == 0.)
		DTM::pout << "eta_k[0] = " << eta_k[0] << std::endl;

	// TODO: output is slab-wise and NOT timecell-wise

	///////////////////////////////
	// output eta_k for each slab

	std::ostringstream filename;
	filename
		<< "slabwise_eta_k-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop << ".log";

	std::ofstream eta_k_out;
	eta_k_out.open(filename.str());

	unsigned int _i = 0;
	for (auto &slab : grid->slabs)
	{
		// save to txt file
		eta_k_out << slab.t_m << "," << eta_k[_i] << std::endl;
		eta_k_out << slab.t_n << "," << eta_k[_i] << std::endl;
		_i++;
	}
	eta_k_out.close();

	/////////////////////////////
	// output k_n for each slab

	// reset filename
	filename.str("");
	filename.clear();

	filename
			<< "slabwise_k_n-dwr_loop-"
			<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop << ".log";

	std::ofstream k_n_out;
	k_n_out.open(filename.str());

	for (auto &slab : grid->slabs)
	{
		// save to txt file
		k_n_out << slab.t_m << "," << slab.tau_n() << std::endl;
		k_n_out << slab.t_n << "," << slab.tau_n() << std::endl;
	}
	k_n_out.close();

	{
		auto eta_it{error_estimator.storage.eta_space->begin()};
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (unsigned n{0}; n < N; ++n, ++eta_it, ++slab) {
			Assert(
				(eta_it != error_estimator.storage.eta_space->end()),
				dealii::ExcInternalError()
			);

			eta_h_global += parameter_set->time.fluid.T/(N*slab->tau_n())*
				std::accumulate(
						eta_it->x[0]->begin(),
						eta_it->x[0]->end(),
						0.
				);
		}
	}

	// Choose if temporal or spatial discretization should be refined
	// according to Algorithm 4.1 in Schmich & Vexler
	double equilibration_factor{1.0e7};

	///////////////////////////////////////////
	// mark for temporal refinement
	if (std::abs(eta_k_global)*equilibration_factor >= std::abs(eta_h_global))
	{
		Assert(
			((parameter_set->dwr.refine_and_coarsen.time.top_fraction >= 0.) &&
			(parameter_set->dwr.refine_and_coarsen.time.top_fraction <= 1.)),
			dealii::ExcMessage(
				"parameter_set->dwr.refine_and_coarsen.time.top_fraction "
				"must be in [0,1]"
			)
		);

		if (parameter_set->dwr.refine_and_coarsen.time.top_fraction > 0.) {
			std::vector<double> eta_sorted(eta_k);
			std::sort(eta_sorted.begin(), eta_sorted.end(),std::greater<double>());


			double threshold = 0.;
			//do Doerfler marking
			if ( parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_fraction") == 0){
				double D_goal = std::accumulate(
						eta_k.begin(),
						eta_k.end(),
						0.
				) * parameter_set->dwr.refine_and_coarsen.time.top_fraction;

				double D_sum = 0.;
				for ( unsigned int n{0} ; n < N ; n++ )
				{
					D_sum += eta_sorted[n];
					if ( D_sum >= D_goal ){
						threshold = eta_sorted[n];
						n = N;
					}
				}

			} else if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_number") == 0) {
				// check if index for eta_criterium_for_mark_time_refinement is valid
				Assert(static_cast<int>(std::ceil(static_cast<double>(N)
						* parameter_set->dwr.refine_and_coarsen.time.top_fraction)) >= 0,
					dealii::ExcInternalError()
				);

				unsigned int index_for_mark_time_refinement {
					static_cast<unsigned int> (
						static_cast<int>(std::ceil(
							static_cast<double>(N)
							* parameter_set->dwr.refine_and_coarsen.time.top_fraction
						))
					)
				};

				threshold = eta_sorted[ index_for_mark_time_refinement < N ?
											index_for_mark_time_refinement : N-1 ];

			}

			auto slab{grid->slabs.begin()};
			auto ends{grid->slabs.end()};
			for (unsigned int n{0} ; slab != ends; ++slab, ++n) {
				Assert((n < N), dealii::ExcInternalError());

				if (eta_k[n] >= threshold) {
					slab->set_refine_in_time_flag();
					DTM::pout << "Marked slab " << n << " for temporal refinement" << std::endl;
				}
			}
		}
	}

	///////////////////////////////
	// spatial refinement
	if (std::abs(eta_k_global) <= equilibration_factor*std::abs(eta_h_global))
	{
		for (auto &eta_In : *error_estimator.storage.eta_space) {
			for (auto &eta_K : *eta_In.x[0] ) {
				eta_K = std::abs(eta_K);
				Assert(eta_K >= 0., dealii::ExcInternalError());
			}
		}
		unsigned int K_max{0};
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		auto eta_it{error_estimator.storage.eta_space->begin()};
		for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it, ++n) {

			Assert(
				(eta_it != error_estimator.storage.eta_space->end()),
				dealii::ExcInternalError()
			);

			DTM::pout << "\tn = " << n << std::endl;

			const auto n_active_cells_on_slab{slab->space.tria->n_global_active_cells()};
			DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
			K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

			if ( parameter_set->dwr.refine_and_coarsen.space.top_fraction1 == 1.0 )
			{
				slab->space.tria->refine_global(1);
			}
			else {
				const unsigned int dofs_per_cell_pu = slab->space.pu.fe_info->fe->dofs_per_cell;
				std::vector< unsigned int > local_dof_indices(dofs_per_cell_pu);
				unsigned int max_n = n_active_cells_on_slab *
										parameter_set->dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells;

				typename dealii::DoFHandler<dim>::active_cell_iterator
					cell{slab->space.pu.fe_info->dof->begin_active()},
					endc{slab->space.pu.fe_info->dof->end()};

				dealii::Vector<double> indicators(n_active_cells_on_slab);
				indicators= 0.;

				for ( unsigned int cell_no{0} ; cell!= endc; ++cell, ++cell_no){
					cell->get_dof_indices(local_dof_indices);

					for ( unsigned int i = 0 ; i < dofs_per_cell_pu ; i++) {
						indicators[cell_no] += (*eta_it->x[0])(local_dof_indices[i])/dofs_per_cell_pu;
					}
				}

				if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("RichterWick") == 0){
					double threshold = eta_it->x[0]->mean_value()*
							parameter_set->dwr.refine_and_coarsen.space.riwi_alpha;

					dealii::GridRefinement::refine(
						*slab->space.tria,
						indicators,
						threshold,
						max_n
					);
				} else {
					// mark for refinement strategy with fixed fraction
					// (similar but not identical to Hartmann Diploma thesis Ex. Sec. 1.4.2)
					const double top_fraction{ slab->refine_in_time ?
						parameter_set->dwr.refine_and_coarsen.space.top_fraction1 :
						parameter_set->dwr.refine_and_coarsen.space.top_fraction2
					};

					if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("fixed_fraction") == 0){
						dealii::GridRefinement::refine_and_coarsen_fixed_fraction(
							*slab->space.tria,
							indicators,
							top_fraction,
							parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
							max_n
						);
					} else if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("fixed_number") == 0){
						dealii::GridRefinement::refine_and_coarsen_fixed_number(
							*slab->space.tria,
							indicators,
							top_fraction,
							parameter_set->dwr.refine_and_coarsen.space.bottom_fraction,
							max_n
						);
					} else {
						AssertThrow(false,dealii::ExcMessage("unknown spatial refinement"));
					}

				}

				// count which percentage of spatial cells have been marked for refinement
				unsigned int marked_cells = 0;

				for (const auto &cell : slab->space.tria->active_cell_iterators())
					if (cell->refine_flag_set())
						marked_cells++;

				DTM::pout << "\tSpace top fraction = " << std::setprecision(5) << ((double)marked_cells) / slab->space.tria->n_active_cells() << std::endl;

				// execute refinement in space under the conditions of mesh smoothing
				slab->space.tria->execute_coarsening_and_refinement();
			}
		}
		DTM::pout << "\t#Kmax (before refinement) = " << K_max << std::endl;
	}

	// do actual refine in time loop
	{
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (; slab != ends; ++slab) {
			if (slab->refine_in_time) {
				//slab->time.tria->refine_global(1);
				grid->refine_slab_in_time(slab); // splitting slab into two slabs
				slab->refine_in_time = false;
			}
		}
	}
	DTM::pout << "refined in time" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// eta data output
//

template<int dim>
void
Fluid<dim>::
eta_init_data_output() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	// set up which dwr loop(s) are allowed to make data output:
	if (!parameter_set->data_output.error_estimator.dwr_loop.compare("none") ) {
		return;
	}

	// may output data: initialise (mode: all, last or specific dwr loop)
	DTM::pout
		<< "error indicators data output: patches = "
		<< parameter_set->data_output.error_estimator.patches
		<< std::endl;

	////////////////////////////////////////////////////////////////////////////
	// INIT DATA OUTPUT
	//

	error_estimator.data_output_space = std::make_shared< DTM::DataOutput<dim> >();
	error_estimator.data_output_time = std::make_shared< DTM::DataOutput<dim> >();

	std::vector<std::string> data_field_names;
	data_field_names.push_back("error_indicator");

	error_estimator.data_output_space->set_data_field_names(data_field_names);
	error_estimator.data_output_time->set_data_field_names(data_field_names);

	std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci_field;
	dci_field.push_back(dealii::DataComponentInterpretation::component_is_scalar);

	error_estimator.data_output_space->set_data_component_interpretation_field(dci_field);
	error_estimator.data_output_time->set_data_component_interpretation_field(dci_field);


	error_estimator.data_output_space->set_data_output_patches(
		parameter_set->data_output.error_estimator.patches
	);
	error_estimator.data_output_time->set_data_output_patches(
		parameter_set->data_output.error_estimator.patches
	);

	// check if we use a fixed trigger interval, or, do output once on a I_n
	if ( !parameter_set->data_output.error_estimator.trigger_type.compare("fixed") ) {
		error_estimator.data_output_trigger_type_fixed = true;
	}
	else {
		error_estimator.data_output_trigger_type_fixed = false;
	}

	// only for fixed
	error_estimator.data_output_trigger = parameter_set->data_output.error_estimator.trigger;

	if (error_estimator.data_output_trigger_type_fixed) {
		DTM::pout
			<< "error indicators data output: using fixed mode with trigger = "
			<< error_estimator.data_output_trigger
			<< std::endl;
	}
	else {
		DTM::pout
			<< "error indicators data output: using I_n mode (trigger adapts to I_n automatically)"
			<< std::endl;
	}

	error_estimator.data_output_time_value = parameter_set->time.fluid.T;

//	DTM::pout
//		<< "error indicators data output: dwr loop = "
//		<< error_estimator.data_output_dwr_loop
//		<< std::endl;
}


template<int dim>
void
Fluid<dim>::
eta_space_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta,
	const unsigned int dwr_loop) {
	// TODO: might need to be debugged; adapted from primal_do_data_output_on_slab() & copied form dual_do_data_output_on_slab()

	// triggered output mode
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.pu.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	error_estimator.data_output_space->set_DoF_data(
		slab->space.pu.fe_info->dof,
		slab->space.pu.fe_info->partitioning_locally_owned_dofs
	);

 	auto eta_trigger = std::make_shared< dealii::Vector<double> > ();
 	eta_trigger->reinit(
 		slab->space.pu.fe_info->dof->n_dofs()
 	);

	std::ostringstream filename;
	filename
		<< "error-indicators-space-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.pu.fe_info->mapping,
			*slab->time.pu.fe_info->fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);

		Assert(
			slab->time.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		auto cell_time = slab->time.pu.fe_info->dof->begin_active();
		auto endc_time = slab->time.pu.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_face_values_time.reinit(cell_time);

			////////////////////////////////////////////////////////////////////
			// construct quadrature for data output,
			// if triggered output time values are inside this time element
			//

			auto t_m = fe_face_values_time.quadrature_point(0)[0];
			auto t_n = fe_face_values_time.quadrature_point(1)[0];
			auto tau = t_n-t_m;

			std::list<double> output_times;

			if (error_estimator.data_output_time_value < t_m) {
				error_estimator.data_output_time_value = t_m;
			}

			for ( ; (error_estimator.data_output_time_value <= t_n) ||
				(
					(error_estimator.data_output_time_value > t_n) &&
					(std::abs(error_estimator.data_output_time_value - t_n) < tau*1e-12)
				); ) {
				output_times.push_back(error_estimator.data_output_time_value);
				error_estimator.data_output_time_value += error_estimator.data_output_trigger;
			}

			if (output_times.size() && output_times.back() > t_n) {
				output_times.back() = t_n;
			}

			if ((output_times.size() > 1) &&
				(output_times.back() == *std::next(output_times.rbegin()))) {
				// remove the last entry, iff doubled
				output_times.pop_back();
			}

			// convert container
			if (!output_times.size()) {
				continue;
			}

			std::vector< dealii::Point<1> > output_time_points(output_times.size());
			{
				auto time{output_times.begin()};
				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
					double t_trigger{*time};
					output_time_points[q][0] = (t_trigger-t_m)/tau;
				}

				if (output_time_points[0][0] < 0) {
					output_time_points[0][0] = 0;
				}

				if (output_time_points[output_time_points.size()-1][0] > 1) {
					output_time_points[output_time_points.size()-1][0] = 1;
				}
			}

			dealii::Quadrature<1> quad_time(output_time_points);

			// create fe values
			dealii::FEValues<1> fe_values_time(
				*slab->time.pu.fe_info->mapping,
				*slab->time.pu.fe_info->fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);

			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.pu.fe_info->fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);

			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*eta_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.pu.fe_info->dof->n_dofs(); ++i) {
 					(*eta_trigger)[i] += (*eta->x[0])[
 						i
 						// time offset
 						+ slab->space.pu.fe_info->dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

//				std::cout
//					<< "error indicator output generated for t = "
//					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
//					<< std::endl;

				error_estimator.data_output_space->write_data(
					filename.str(),
					eta_trigger,
					// error_estimator.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
eta_time_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta,
	const unsigned int dwr_loop) {
	// TODO: might need to be debugged; adapted from primal_do_data_output_on_slab() & copied form dual_do_data_output_on_slab()

	// triggered output mode
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.pu.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	error_estimator.data_output_time->set_DoF_data(
		slab->space.pu.fe_info->dof,
		slab->space.pu.fe_info->partitioning_locally_owned_dofs
	);

 	auto eta_trigger = std::make_shared< dealii::Vector<double> > ();
 	eta_trigger->reinit(
 		slab->space.pu.fe_info->dof->n_dofs()
 	);

	std::ostringstream filename;
	filename
		<< "error-indicators-time-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.pu.fe_info->mapping,
			*slab->time.pu.fe_info->fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);

		Assert(
			slab->time.pu.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		auto cell_time = slab->time.pu.fe_info->dof->begin_active();
		auto endc_time = slab->time.pu.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_face_values_time.reinit(cell_time);

			////////////////////////////////////////////////////////////////////
			// construct quadrature for data output,
			// if triggered output time values are inside this time element
			//

			auto t_m = fe_face_values_time.quadrature_point(0)[0];
			auto t_n = fe_face_values_time.quadrature_point(1)[0];
			auto tau = t_n-t_m;

			std::list<double> output_times;

			if (error_estimator.data_output_time_value < t_m) {
				error_estimator.data_output_time_value = t_m;
			}

			for ( ; (error_estimator.data_output_time_value <= t_n) ||
				(
					(error_estimator.data_output_time_value > t_n) &&
					(std::abs(error_estimator.data_output_time_value - t_n) < tau*1e-12)
				); ) {
				output_times.push_back(error_estimator.data_output_time_value);
				error_estimator.data_output_time_value += error_estimator.data_output_trigger;
			}

			if (output_times.size() && output_times.back() > t_n) {
				output_times.back() = t_n;
			}

			if ((output_times.size() > 1) &&
				(output_times.back() == *std::next(output_times.rbegin()))) {
				// remove the last entry, iff doubled
				output_times.pop_back();
			}

			// convert container
			if (!output_times.size()) {
				continue;
			}

			std::vector< dealii::Point<1> > output_time_points(output_times.size());
			{
				auto time{output_times.begin()};
				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
					double t_trigger{*time};
					output_time_points[q][0] = (t_trigger-t_m)/tau;
				}

				if (output_time_points[0][0] < 0) {
					output_time_points[0][0] = 0;
				}

				if (output_time_points[output_time_points.size()-1][0] > 1) {
					output_time_points[output_time_points.size()-1][0] = 1;
				}
			}

			dealii::Quadrature<1> quad_time(output_time_points);

			// create fe values
			dealii::FEValues<1> fe_values_time(
				*slab->time.pu.fe_info->mapping,
				*slab->time.pu.fe_info->fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);

			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.pu.fe_info->fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);

			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*eta_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.pu.fe_info->dof->n_dofs(); ++i) {
 					(*eta_trigger)[i] += (*eta->x[0])[
 						i
 						// time offset
 						+ slab->space.pu.fe_info->dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

//				std::cout
//					<< "error indicator output generated for t = "
//					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
//					<< std::endl;

				error_estimator.data_output_time->write_data(
					filename.str(),
					eta_trigger,
					// error_estimator.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
eta_space_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.pu.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	error_estimator.data_output_space->set_DoF_data(
		slab->space.pu.fe_info->dof,
		slab->space.pu.fe_info->partitioning_locally_owned_dofs
	);

	auto eta_trigger = std::make_shared< dealii::Vector<double> > ();
	eta_trigger->reinit(
		slab->space.pu.fe_info->dof->n_dofs()
	);

	std::ostringstream filename;
	filename
		<< "error-indicators-space-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{

		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.pu.fe_info->mapping,
			*slab->time.pu.fe_info->fe,
			dealii::QGauss<1>(slab->time.pu.fe_info->fe->tensor_degree()+1), // PU = dG(0) in time -> QGauss(1)
			dealii::update_values |
			dealii::update_quadrature_points
		);

		auto cell_time = slab->time.pu.fe_info->dof->begin_active();
		auto endc_time = slab->time.pu.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.pu.fe_info->fe->dofs_per_cell
			);

			cell_time->get_dof_indices(local_dof_indices_time);

			for (
				unsigned int qt{0};
				qt < fe_values_time.n_quadrature_points;
				++qt) {
 				*eta_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.pu.fe_info->dof->n_dofs(); ++i) {
 					(*eta_trigger)[i] += (*eta->x[0])[
 						i
 						// time offset
 						+ slab->space.pu.fe_info->dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0]
// 					<< std::endl;

				error_estimator.data_output_space->write_data(
					filename.str(),
					eta_trigger,
					// error_estimator.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
eta_time_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_vectors<1>::iterator &eta,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.pu.fe_info->partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	error_estimator.data_output_time->set_DoF_data(
		slab->space.pu.fe_info->dof,
		slab->space.pu.fe_info->partitioning_locally_owned_dofs
	);

	auto eta_trigger = std::make_shared< dealii::Vector<double> > ();
	eta_trigger->reinit(
		slab->space.pu.fe_info->dof->n_dofs()
	);

	std::ostringstream filename;
	filename
		<< "error-indicators-time-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// TODO: check natural time quadrature (Gauss, Gauss-Radau, etc.)
		//       from input or generated fe<1>
		//
		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.pu.fe_info->mapping,
			*slab->time.pu.fe_info->fe,
			dealii::QGauss<1>(slab->time.pu.fe_info->fe->tensor_degree()+1), // here
			dealii::update_values |
			dealii::update_quadrature_points
		);

		auto cell_time = slab->time.pu.fe_info->dof->begin_active();
		auto endc_time = slab->time.pu.fe_info->dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.pu.fe_info->fe->dofs_per_cell
			);

			cell_time->get_dof_indices(local_dof_indices_time);

			for (
				unsigned int qt{0};
				qt < fe_values_time.n_quadrature_points;
				++qt) {
 				*eta_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.pu.fe_info->dof->n_dofs(); ++i) {
 					(*eta_trigger)[i] += (*eta->x[0])[
 						i
 						// time offset
 						+ slab->space.pu.fe_info->dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0]
// 					<< std::endl;

				error_estimator.data_output_time->write_data(
					filename.str(),
					eta_trigger,
					// error_estimator.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] + (qt == 0) * 1e-5 // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
eta_do_data_output(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_space,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_time,
		const unsigned int dwr_loop,
		bool last
) {
	if (error_estimator.data_output_trigger <= 0) return;

	// set up which dwr loop(s) are allowed to make data output:
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	if ( !parameter_set->data_output.error_estimator.dwr_loop.compare("none") ) {
		return;
	}

	if (!parameter_set->data_output.error_estimator.dwr_loop.compare("last")) {
		// output only the last (final) dwr loop
		if (last) {
			error_estimator.data_output_dwr_loop = dwr_loop;
		}
		else {
			return;
		}
	}
	else {
		if (!parameter_set->data_output.error_estimator.dwr_loop.compare("all")) {
			// output all dwr loops
			if (!last) {
				error_estimator.data_output_dwr_loop = dwr_loop;
			}
			else {
				return;
			}
		}
		else {
			// output on a specific dwr loop
			if (!last) {
				error_estimator.data_output_dwr_loop =
					std::stoi(parameter_set->data_output.error_estimator.dwr_loop)-1;
			}
			else {
				return;
			}
		}
	}

	if (error_estimator.data_output_dwr_loop < 0)
		return;

	if ( static_cast<unsigned int>(error_estimator.data_output_dwr_loop) != dwr_loop )
		return;

	if (!error_estimator.data_output_trigger_type_fixed) {
		// I_n output mode (output on natural Q_n support points in time)
		eta_space_do_data_output_on_slab_Qn_mode(slab, eta_space, dwr_loop);
		eta_time_do_data_output_on_slab_Qn_mode(slab, eta_time, dwr_loop);
	}
	else {
		// fixed trigger output mode
		eta_space_do_data_output_on_slab(slab, eta_space, dwr_loop);
		eta_time_do_data_output_on_slab(slab, eta_time, dwr_loop);
	}
}


template<int dim>
void
Fluid<dim>::
eta_sort_xdmf_by_time(
	const unsigned int dwr_loop
) {
	// name of xdmf file to be sorted by time of the snapshots
	std::ostringstream filename_space;
	filename_space
		<< "error-indicators-space-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop
		<< ".xdmf";

	std::ostringstream filename_time;
	filename_time
		<< "error-indicators-time-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop
		<< ".xdmf";

	for (auto filename_str : {filename_space.str(), filename_time.str()})
	{
		// read the current xdmf file and save it line by line in a std::vector
		std::ifstream old_xdmf_file(filename_str);
		if (!old_xdmf_file.good()) // exit this function, if there is no xdmf file
			return;

		std::vector< std::string > old_lines;

		std::string current_line;
		while (getline (old_xdmf_file, current_line))
			old_lines.push_back(current_line);

		old_xdmf_file.close();

		// create a std::vector for the new lines where the grids are sorted in ascending order by the time
		std::vector< std::string > new_lines;

		// the first 5 lines remain unchanged
		for (unsigned int i = 0; i <= 4; ++i)
			new_lines.push_back(old_lines[i]);

		//////////////////////////////////////////
		// reorder the grids by time
		//
		std::vector< std::vector< std::string > > grids;
		std::vector< std::string > current_grid;

		// each grid consists of 18 lines
		for (unsigned int i = 5; i < old_lines.size()-3; ++i)
		{
			current_grid.push_back(old_lines[i]);
			if (current_grid.size() == 18)
			{
				grids.push_back(current_grid);
				current_grid.clear();
			}
		}

		// get time of grid
		auto get_time = [](std::vector< std::string > grid)
		{
			Assert(grid.size() == 18, dealii::ExcInvalidState());
			// the second line contains the time
			std::string line = grid[1];
			// get the number between the double quotes
			std::string delimiter = "\"";
			std::string token = line.substr(line.find(delimiter)+1, line.size());
			token = token.substr(0, token.find(delimiter));
			return std::stod(token);
		};

		// sort the grids by time
		std::sort(std::begin(grids),
				  std::end(grids),
				  [&](std::vector< std::string > & a, std::vector< std::string > & b)
		{
			return get_time(a) < get_time(b);
		});

		// append sorted grid to new_lines
		for (auto grid : grids)
		{
			for (auto line : grid)
			{
				new_lines.push_back(line);
			}
		}

		// the last 3 lines remain unchanged
		for (unsigned int i = old_lines.size()-3; i < old_lines.size(); ++i)
		{
			new_lines.push_back(old_lines[i]);
		}

		// write to file
		std::ofstream new_xdmf_file;
		new_xdmf_file.open(filename_str);
		for (auto line : new_lines)
			new_xdmf_file << line << std::endl;
		new_xdmf_file.close();
	}
}

} // namespace

#include "Fluid.inst.in"
