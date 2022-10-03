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
using Je_MeanDrag_Assembler = goal::mean::drag::spacetime::Operator::Assembler<dim>;


#include <fluid/assembler/ST_Dual_MeanVorticityAssembly.tpl.hh>
template <int dim>
using Je_MeanVorticity_Assembler = goal::mean::vorticity::spacetime::Operator::Assembler<dim>;

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

#include <ideal.II/dofs/SlabDoFTools.hh>

#include <cstdio>

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

	DTM::pout << "solving only primal problem: " << (parameter_set->primal_only ? "true" : "false") << std::endl;

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
	DTM::pout << std::endl << std::endl;

	//give info about functional values
	DTM::pout << "The following functional values will be calculated:" << std::endl;
	if ( parameter_set->dwr.functional.mean_drag)
		DTM::pout << "mean drag coefficient" << std::endl;

	if ( parameter_set->dwr.functional.mean_lift)
			DTM::pout << "mean lift coefficient" << std::endl;

	if ( parameter_set->dwr.functional.mean_pdiff)
			DTM::pout << "mean pressure difference" << std::endl;

	if ( parameter_set->dwr.functional.mean_vorticity)
			DTM::pout << "mean vorticity" << std::endl;

	DTM::pout << "Refinement will be based on the ";
	if ( !parameter_set->dwr.goal.type.compare("mean_drag")){
		DTM::pout << "mean drag coefficient" << std::endl;
		AssertThrow(parameter_set->dwr.functional.mean_drag,dealii::ExcMessage("You want to refine based on a functional that is not calculated, aborting!"));
	} else if ( !parameter_set->dwr.goal.type.compare("mean_lift")){
		AssertThrow(false,dealii::ExcNotImplemented());
		DTM::pout << "mean lift coefficient" << std::endl;
		AssertThrow(parameter_set->dwr.functional.mean_lift,dealii::ExcMessage("You want to refine based on a functional that is not calculated, aborting!"));
	} else if ( !parameter_set->dwr.goal.type.compare("mean_pdiff")){
		AssertThrow(false,dealii::ExcNotImplemented());
		DTM::pout << "mean pressure difference" << std::endl;
		AssertThrow(parameter_set->dwr.functional.mean_pdiff,dealii::ExcMessage("You want to refine based on a functional that is not calculated, aborting!"));
	} else if ( !parameter_set->dwr.goal.type.compare("mean_vorticity")){
		DTM::pout << "mean vorticity" << std::endl;
		AssertThrow(parameter_set->dwr.functional.mean_vorticity,dealii::ExcMessage("You want to refine based on a functional that is not calculated, aborting!"));
	}

	// determine setw value for dwr loop number of data output filename
	setw_value_dwr_loops = static_cast<unsigned int>(
		std::floor(std::log10(parameter_set->dwr.loops))+1
	);
	
	init_functions();
	init_reference_values();
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
			{
				// refine grid in space
				grid->refine_global(1, 0);
			}
			else if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("global_time"))
			{
				// refine grid in time
				grid->refine_global(0, 1);
			}
			else if	(!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("global"))
			{
				// refine grid in space AND time
				grid->refine_global(1, 1);
			}
			else if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive"))
			{
				// do adaptive space-time mesh refinements and coarsenings
				AssertThrow(
					!parameter_set->primal_only,
					dealii::ExcMessage(
						"need to solve dual problem and assemble error estimator to do adaptive mesh refinement"
					)
				);
				refine_and_coarsen_space_time_grid(dwr_loop-1);
			}
			else{
				// invalid refinement strategy
				AssertThrow(false, dealii::ExcNotImplemented());
			}
			grid->set_manifolds();
		}
		
		DTM::pout
			<< "***************************************************************"
			<< "*****************" << std::endl
			<< "adaptivity loop = " << dwr_loop << std::endl;

		//overwrite temp drag/lift/pressure logs and write headers
		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			std::ofstream out;
			if( parameter_set->dwr.functional.mean_pdiff){
				out.open("pressure.log");
				out << "time,pressure" << std::endl;
				out.close();
			}
			if( parameter_set->dwr.functional.mean_drag){
				out.open("drag.log");
				out << "time,drag" << std::endl;
				out.close();
			}
			if( parameter_set->dwr.functional.mean_lift){
				out.open("lift.log");
				out << "time,lift" << std::endl;
				out.close();
			}
			if( parameter_set->dwr.functional.mean_vorticity){
				out.open("vorticity.log");
				out << "time,vorticity" << std::endl;
				out.close();
			}
		}
		
		grid->set_boundary_indicators();
		
		// primal problem:
		primal_reinit_storage();
		primal_init_data_output();
		primal_do_forward_TMS(dwr_loop, false);

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){

			if( parameter_set->dwr.functional.mean_pdiff){
				std::ostringstream pfilename;
				pfilename << "pressure" << dwr_loop << ".log";
				std::rename("pressure.log",pfilename.str().c_str());
			}

			if( parameter_set->dwr.functional.mean_drag){
				std::ostringstream dfilename;
				dfilename << "drag" << dwr_loop << ".log";
				std::rename("drag.log",dfilename.str().c_str());
			}

			if( parameter_set->dwr.functional.mean_lift){
				std::ostringstream lfilename;
				lfilename << "lift" << dwr_loop << ".log";
				std::rename("lift.log",lfilename.str().c_str());
			}

			if( parameter_set->dwr.functional.mean_vorticity){
				std::ostringstream lfilename;
				lfilename << "vorticity" << dwr_loop << ".log";
				std::rename("vorticity.log",lfilename.str().c_str());
			}
		}

		if (!parameter_set->primal_only)
		{
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
		}


		// compute the number of primal and dual space-time dofs
		unsigned long int n_primal_st_dofs = 0;
		unsigned long int n_dual_st_dofs   = 0;

		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (; slab != ends; ++slab)
		{
			n_primal_st_dofs += slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs();
			if (!parameter_set->primal_only)
				n_dual_st_dofs += slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs();
		}

		DTM::pout << "\n#DoFs(primal; Space-Time) = " << n_primal_st_dofs;
		if (!parameter_set->primal_only)
			DTM::pout << "\n#DoFs(dual; Space-Time)   = " << n_dual_st_dofs;
		DTM::pout << std::endl;

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


	Assert(function.convection.dirichlet.use_count(), dealii::ExcNotInitialized());
	grid->set_dirichlet_function(function.convection.dirichlet);
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
init_reference_values() {
	if (parameter_set->problem.compare("Navier-Stokes") == 0){
	  error_estimator.goal_functional.reference.mean_drag
	   = parameter_set->reference.navier_stokes.mean_drag;

	  error_estimator.goal_functional.reference.mean_lift
	   = parameter_set->reference.navier_stokes.mean_lift;

	  error_estimator.goal_functional.reference.mean_pdiff
	   =  parameter_set->reference.navier_stokes.mean_pdiff;

	  error_estimator.goal_functional.reference.mean_vorticity
	   =  parameter_set->reference.navier_stokes.mean_vorticity;
	}
	else {
	  error_estimator.goal_functional.reference.mean_drag
	   = parameter_set->reference.stokes.mean_drag;

	  error_estimator.goal_functional.reference.mean_lift
	   = parameter_set->reference.stokes.mean_lift;

	  error_estimator.goal_functional.reference.mean_pdiff
	   = parameter_set->reference.stokes.mean_pdiff;

	  error_estimator.goal_functional.reference.mean_vorticity
	   = parameter_set->reference.stokes.mean_vorticity;
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


	primal.sc = std::make_shared<dealii::SolverControl> (10000,newton.lower_bound*1.0e-08,false,false);
	primal.ad = std::make_shared<dealii::TrilinosWrappers::SolverDirect::AdditionalData>(false,"Amesos_Mumps");

	dual.sc = std::make_shared<dealii::SolverControl> (10000,newton.lower_bound*1.0e-08,false,false);
	dual.ad = std::make_shared<dealii::TrilinosWrappers::SolverDirect::AdditionalData>(false,"Amesos_Mumps");
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
		std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();
	
	primal.storage.u->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);

	primal.storage.um =
		std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();

	primal.storage.um->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);

	primal.storage.vorticity =
			std::make_shared< DTM::types::storage_data_trilinos_vectors<1>> ();

	primal.storage.vorticity->resize(
			static_cast<unsigned int>(grid->slabs.size())
	);
}

template<int dim>
void
Fluid<dim>::
primal_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &x,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &xm
) {
	for (unsigned int j{0}; j < x->x.size(); ++j) {
		// x
		x->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

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
			*slab->spacetime.primal.locally_owned_dofs,
			*slab->spacetime.primal.locally_relevant_dofs,
			mpi_comm
		);

		// xm
		xm->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

		xm->x[j]->reinit(
			*slab->space.primal.fe_info->locally_owned_dofs,
			*slab->space.primal.fe_info->locally_relevant_dofs,
			mpi_comm
		);
	}
}

template<int dim>
void
Fluid<dim>::
primal_reinit_vorticity_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &v
) {
	for (unsigned int j{0}; j < v->x.size(); ++j) {
		// x
		v->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

		Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

		Assert(
			slab->space.vorticity.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		Assert(
			slab->time.vorticity.fe_info->dof.use_count(),
			dealii::ExcNotInitialized()
		);

		v->x[j]->reinit(
			*slab->spacetime.vorticity.locally_owned_dofs,
			*slab->spacetime.vorticity.locally_relevant_dofs,
			mpi_comm
		);
	}
}

template<int dim>
void
Fluid<dim>::
primal_assemble_system(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u
) {
	// ASSEMBLY SPACE-TIME OPERATOR MATRIX /////////////////////////////////////

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
}


template<int dim>
void
Fluid<dim>::
primal_assemble_const_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &um
) {
	// ASSEMBLY SPACE-TIME OPERATOR: InitialValue VECTOR ///////////////////////
	primal.Mum = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	
	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	Assert(um->x[0].use_count(),dealii::ExcNotInitialized());

//	*um->x[0] = *primal.um;
	if (!parameter_set->fe.primal_projection_type.compare("none")){
		*um->x[0] = *primal.um;
	}
	else { //do L2 or H1 projection


		*primal.projection_matrix = 0;


		primal.projection_rhs = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		primal.projection_rhs->reinit(*slab->space.primal.fe_info->locally_owned_dofs,mpi_comm);
		*primal.projection_rhs = 0;

		primal.um_projected = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		primal.um_projected->reinit(*slab->space.primal.fe_info->locally_owned_dofs,mpi_comm);
		*primal.um_projected = 0;

		auto tmp_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		tmp_relevant->reinit(
				*slab->space.primal.fe_info->locally_owned_dofs,
				*slab->space.primal.fe_info->locally_relevant_dofs,
				mpi_comm);
		*tmp_relevant = *primal.um;

		// use divergence-free projection on primal.um
		use_gradient_projection = false;
		if (!parameter_set->fe.primal_projection_type.compare("H1")){
			use_gradient_projection = true;
		}

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
					tmp_relevant,
					primal.projection_rhs,
					slab
			);


			// 3. solve projection linear system for primal.um_projected
			primal.projection_iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (*primal.sc,*primal.ad);
			primal.projection_iA->initialize(*primal.projection_matrix);
			primal.projection_iA->solve(*primal.um_projected,*primal.projection_rhs);

			primal.projection_iA = nullptr;
			// distribute hanging node constraints on solution
			slab->space.primal.fe_info->initial_constraints->distribute(
					*primal.um_projected
			);

			*um->x[0] = *primal.um_projected;

			DTM::pout << " (done)" << std::endl;
		}

	}
	primal.Mum->reinit(
		*slab->spacetime.primal.locally_owned_dofs,
		mpi_comm
	);
	*primal.Mum = 0.;
	
	{
		IVAssembler<dim> assembler;

	    *primal.relevant_tmp = *primal.um;
	    //*primal.relevant_tmp = *primal.um_projected;
		DTM::pout << "dwr-instatfluid: assemble space-time slab initial value vector...";
		Assert(primal.um.use_count(), dealii::ExcNotInitialized());
		Assert(primal.Mum.use_count(), dealii::ExcNotInitialized());
		assembler.assemble(
			um->x[0],
			primal.Mum,
			slab
		);

		DTM::pout << " (done)" << std::endl;
	}
	
//	// ASSEMBLY SPACE-TIME OPERATOR: FORCE VECTOR //////////////////////////////
//	primal.f = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
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
	std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector > u

) {
	// ASSEMBLY SPACE-TIME OPERATOR: Rhs Bilinearform VECTOR ///////////////////////
	primal.Fu = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	primal.Fu->reinit(
		*slab->spacetime.primal.locally_owned_dofs,mpi_comm
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
	slab->spacetime.primal.constraints->distribute(
		*primal.b
	);
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
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > x
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
			if (x->locally_owned_elements().is_element(boundary_value.first)){
			// set constrained solution vector component (for iterative lss)
				(*x)[boundary_value.first] = boundary_value.second;
			}
		}
	}
	x->compress(dealii::VectorOperation::insert);
}

template<int dim>
void
Fluid<dim>::
primal_apply_bc(
	std::map<dealii::types::global_dof_index, double> &boundary_values,
	std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix > A,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > x,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > b
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


		// 			////////////////////////////////////////////////////////////////////
		// 			// eliminate constrained column entries
		// 			//
		// 			// NOTE: this is quite expensive, but helps iterative lss
		// 			//       since the boundary value entries are shifted to the
		// 			//       right hand side.
		// 			//
		// 			// NOTE: there is no symmetry assumption on the sparsity pattern,
		// 			//       which is necessary for space-time operators
		// 			//
		// 			for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
		// 				// if the row i of the operator A is not constrained,
		// 				// check if constrained columns need to be eliminated
		// 				if (boundary_values.find(i) == boundary_values.end()) {
		// 					// row i of A is not constrained
		// 					auto el_in_row_i{A->begin(i)};
		// 					auto end_el_in_row_i{A->end(i)};
		//
		// 					// check if a_ij needs to be eliminated
		// 					for ( ; el_in_row_i != end_el_in_row_i; ++el_in_row_i) {
		// 						// get iterator of a_ij
		// 						auto boundary_value =
		// 							boundary_values.find(el_in_row_i->column());
		//
		// 						// if a_ij is constrained
		// 						if (boundary_value != boundary_values.end()) {
		// 							// shift constraint to rhs
		// 							(*b)[i] -=
		// 								el_in_row_i->value()*boundary_value->second;
		//
		// 							// eliminate a_ij
		// 							el_in_row_i->value() = 0.;
		// 						}
		// 					}
		// 				}
		// 			}
	}
}

template<int dim>
void
Fluid<dim>::
primal_solve_slab_problem(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &um
) {
	
	primal.b  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	primal.du = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	primal.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	primal.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();

	Assert(
		slab->space.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	
	primal.b->reinit(
		*slab->spacetime.primal.locally_owned_dofs,mpi_comm
	);

	primal.du->reinit(
			*slab->spacetime.primal.locally_owned_dofs,mpi_comm
	);

	primal.owned_tmp->reinit(
			*slab->spacetime.primal.locally_owned_dofs,mpi_comm
	);

	primal.relevant_tmp->reinit(*slab->space.primal.fe_info->locally_owned_dofs,*slab->space.primal.fe_info->locally_relevant_dofs,mpi_comm);

	////////////////////////////////////////////////////////////////////////////
	// apply inhomogeneous Dirichlet boundary values
	//

	DTM::pout << "dwr-instatfluid: compute boundary values..." ;

	std::map<dealii::types::global_dof_index, double> initial_bc;
	primal_calculate_boundary_values(slab, initial_bc);
//    std::map<dealii::types::global_dof_index, double> zero_bc;
//    primal_calculate_boundary_values(slab, zero_bc, true);

    DTM::pout << " (done)" << std::endl;

    DTM::pout << "dwr-instatfluid: apply previous solution as initial Newton guess..." ;

//    DTM::pout << "relev tmp" << std::endl;
//    primal.relevant_tmp->print(DTM::pout);
    dealii::IndexSet::ElementIterator lri = slab->space.primal.fe_info->locally_owned_dofs->begin();
    dealii::IndexSet::ElementIterator lre = slab->space.primal.fe_info->locally_owned_dofs->end();
    for (; lri != lre ;lri++) {
    	for (unsigned int ii{0} ; ii < slab->time.primal.fe_info->dof->n_dofs() ; ii++) {
    		(*primal.owned_tmp)[*lri+slab->space.primal.fe_info->dof->n_dofs()*ii] = (*primal.um)[*lri];
    	}
    }

//    primal_owned_tmp->compress(dealii::VectorOperation::insert);


//    DTM::pout << "owned tmp" << std::endl;
//    primal.owned_tmp->print(DTM::pout);


    DTM::pout << " (done)" << std::endl;

    DTM::pout << "dwr-instatfluid: apply bc's to initial Newton guess...";
    primal_apply_bc(initial_bc, primal.owned_tmp);
//    slab->spacetime.primal.constraints->distribute(
//		*primal.owned_tmp
//	);

    DTM::pout << " (done)" << std::endl;
//    primal.owned_tmp ->compress(dealii::VectorOperation::insert);
//    DTM::pout << "u_1^0" << std::endl;
    *u->x[0] = *primal.owned_tmp;

    // assemble slab problem const rhs

    *primal.relevant_tmp = *primal.um;
    DTM::pout << "dwr-instatfluid: assemble const part of rhs...";
	primal_assemble_const_rhs(slab, um);
    DTM::pout << " (done)" << std::endl;
	Assert(
		primal.Mum.use_count(),
		dealii::ExcNotInitialized()
	);

    primal_assemble_and_construct_Newton_rhs(slab, u->x[0]);

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
    	primal_assemble_and_construct_Newton_rhs(slab, u->x[0]);
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

				primal.iA = nullptr;
//
				primal.iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (*primal.sc,*primal.ad);
				primal.iA->initialize(*primal.L);
			}
		}

		////////////////////////////////////////////////////////////////////////////
		// solve linear system with direct solver
		//
		primal.iA->solve(*primal.du, *primal.b);
//		primal.b->print(DTM::pout);
//		DTM::pout << "du" << std::endl;
//		primal.du->print(DTM::pout);


		slab->spacetime.primal.constraints->distribute(
			*primal.du
		);

//		exit(EXIT_SUCCESS);
		*primal.owned_tmp = *u->x[0];
		for (line_search_step = 0; line_search_step < newton.line_search_steps; line_search_step++) {
			*primal.owned_tmp += *primal.du;
			*u->x[0] = *primal.owned_tmp;
//			DTM::pout << "u at ls step " << line_search_step << std::endl;
//		    u->x[0]->print(DTM::pout);
//			u->x[0]->add(1.0,*primal.du);

			primal_assemble_and_construct_Newton_rhs(slab, u->x[0]);


			new_newton_residual = primal.b->linfty_norm();

			if (new_newton_residual < newton_residual)
				break;
			else
				*primal.owned_tmp -= *primal.du;

			*primal.du *= newton.line_search_damping;
		}

		*primal.owned_tmp = *u->x[0];
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
	error_estimator.goal_functional.fem.mean_pdiff = 0.;
	error_estimator.goal_functional.fem.mean_vorticity = 0.;
	
	////////////////////////////////////////////////////////////////////////////
	// grid: init slab iterator to first space-time slab: Omega x I_1
	//
	
	Assert(grid.use_count(), dealii::ExcNotInitialized());
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	
	////////////////////////////////////////////////////////////////////////////
	// storage: init iterators to storage_data_trilinos_vectors
	//          corresponding to first space-time slab: Omega x I_1
	//
	
	Assert(primal.storage.u.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.u->size(), dealii::ExcNotInitialized());
	auto u = primal.storage.u->begin();
	
	Assert(primal.storage.um.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.um->size(), dealii::ExcNotInitialized());
	auto um = primal.storage.um->begin();

	Assert(primal.storage.vorticity.use_count(), dealii::ExcNotInitialized());
	Assert(primal.storage.vorticity->size(), dealii::ExcNotInitialized());
	auto vort = primal.storage.vorticity->begin();
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

		if(parameter_set->dwr.functional.mean_vorticity){
			grid->initialize_vorticity_grid_components_on_slab(slab);
			grid->distribute_vorticity_on_slab(slab);

			slab->spacetime.vorticity.locally_owned_dofs =
				std::make_shared<dealii::IndexSet> (
					idealii::SlabDoFTools::extract_locally_owned_dofs(
						slab->space.vorticity.fe_info->dof,
						slab->time.vorticity.fe_info->dof
					)
				);

			slab->spacetime.vorticity.locally_relevant_dofs =
				std::make_shared<dealii::IndexSet> (
					idealii::SlabDoFTools::extract_locally_relevant_dofs(
							slab->space.vorticity.fe_info->dof,
							slab->time.vorticity.fe_info->dof
					)
				);
		}

		// primal dof partitioning
		{
			slab->spacetime.primal.locally_owned_dofs =
				std::make_shared<dealii::IndexSet> (
					idealii::SlabDoFTools::extract_locally_owned_dofs(
							slab->space.primal.fe_info->dof,
							slab->time.primal.fe_info->dof
					)
				);

			slab->spacetime.primal.locally_relevant_dofs =
				std::make_shared<dealii::IndexSet> (
					idealii::SlabDoFTools::extract_locally_relevant_dofs(
							slab->space.primal.fe_info->dof,
							slab->time.primal.fe_info->dof
					)
				);
		}

		{
			slab->spacetime.primal.constraints =
				std::make_shared< dealii::AffineConstraints<double> > ();

			idealii::SlabDoFTools::make_spacetime_constraints(
				slab->space.primal.fe_info->locally_relevant_dofs,
				slab->space.primal.fe_info->constraints, // space constraints
				slab->space.primal.fe_info->dof->n_dofs(),
				slab->time.primal.fe_info->dof->n_dofs(),
				slab->spacetime.primal.locally_relevant_dofs,
				slab->spacetime.primal.constraints
			);

			slab->spacetime.primal.constraints->close();
		}

		{
			slab->spacetime.primal.hanging_node_constraints =
				std::make_shared< dealii::AffineConstraints<double> > ();

			idealii::SlabDoFTools::make_spacetime_constraints(
				slab->space.primal.fe_info->locally_relevant_dofs,
				slab->space.primal.fe_info->hanging_node_constraints, // space constraints
				slab->space.primal.fe_info->dof->n_dofs(),
				slab->time.primal.fe_info->dof->n_dofs(),
				slab->spacetime.primal.locally_relevant_dofs,
				slab->spacetime.primal.hanging_node_constraints
			);

			slab->spacetime.primal.hanging_node_constraints->close();
		}

		if (!parameter_set->problem.compare("Navier-Stokes") || !parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") || slab->t_m == parameter_set->time.fluid.t0)
		{
//			primal.L = nullptr;
			primal.iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (*primal.sc,*primal.ad);
			primal.L = std::make_shared<dealii::TrilinosWrappers::SparseMatrix >();
			primal.projection_matrix = std::make_shared< dealii::TrilinosWrappers::SparseMatrix > ();

			grid->create_sparsity_pattern_primal_on_slab(slab,primal.L,primal.projection_matrix);
		}
		primal_reinit_storage_on_slab(slab, u, um);
		if( parameter_set->dwr.functional.mean_vorticity){
			primal_reinit_vorticity_storage_on_slab(slab, vort);
		}

		primal.um = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		primal.um->reinit(*slab->space.primal.fe_info->locally_owned_dofs,mpi_comm);
		*primal.um = 0.;
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


		}
		else {
			// not the first slab: transfer un solution to um solution
			Assert(primal.un.use_count(), dealii::ExcNotInitialized());


//			primal.um_projected = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
//			primal.um_projected->reinit(slab->space.primal.fe_info->dof->n_dofs());
//			*primal.um_projected = 0.;
			auto un_rel = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
			un_rel->reinit(*std::prev(slab)->space.primal.fe_info->locally_owned_dofs,
					*std::prev(slab)->space.primal.fe_info->locally_relevant_dofs,
					mpi_comm);
			*un_rel = *primal.un;



			// for n > 1 interpolate between two (different) spatial meshes
			// the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->space.primal.fe_info->dof,
				*un_rel,
				// solution on I_n:
				*slab->space.primal.fe_info->dof,
				*slab->space.primal.fe_info->hanging_node_constraints,
				*primal.um
			);

		}
		// NOTE: after the first dwr-loop the initial triangulation could have
		//       hanging nodes. Therefore,
		// distribute hanging node constraints to make the result continuous again:
		slab->space.primal.fe_info->hanging_node_constraints->distribute(
			*primal.um
		);

		// solve slab problem (i.e. apply boundary values and solve for u0)
		primal_solve_slab_problem(slab, u, um);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessings on the solution
		//
		// evaluate solution u(t_n)
		primal.un = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		primal.un->reinit(*slab->space.primal.fe_info->locally_owned_dofs);
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
				{
				    dealii::IndexSet::ElementIterator lri = slab->space.primal.fe_info->locally_owned_dofs->begin();
				    dealii::IndexSet::ElementIterator lre = slab->space.primal.fe_info->locally_owned_dofs->end();
					for (; lri!= lre ; lri++) {
						(*primal.un)[*lri] += (*u->x[0])[
							*lri
							// time offset
							+ slab->space.primal.fe_info->dof->n_dofs() *
								(cell_time->index() * slab->time.primal.fe_info->fe->dofs_per_cell)
							// local in time dof
							+ slab->space.primal.fe_info->dof->n_dofs() * jj
						] * fe_face_values_time.shape_value(jj,1);
					}
				}
			}
		}


		////////////////////////////////////////////////////////////////////////
		// compute functional values:
		//
		compute_functional_values(u, vort, slab);

		// output data
		primal_do_data_output(slab, u, vort, dwr_loop, last);
		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//
		
//		primal.L = nullptr;
		primal.b = nullptr;
// 		primal.f = nullptr;
		primal.Mum = nullptr;
		
		grid->clear_primal_on_slab(slab);

		if( parameter_set->dwr.functional.mean_vorticity){
			// clear vorticity solution
			vort->x[0]->clear();
			slab->space.vorticity.fe_info->dof->clear();
		}

		if (parameter_set->primal_only)
		{
			// also clear primal solution, since it is not being needed for error estimation anymore
			u->x[0]->clear();
			um->x[0]->clear();
		}

//		if ( n > 1){
//			std::prev(u)->x[0]->clear();
//			std::prev(um)->x[0]->clear();
//			std::prev(vort)->x[0]->clear();
//		}

		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//

		++n;
		++slab;
		++u;
		++um;
		++vort;

		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "primal: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
	
	////////////////////////////////////////////////////////////////////////////
	// output exact error

	double reference_goal_functional;
	double fem_goal_functional;

	if ( parameter_set->dwr.functional.mean_drag){
		reference_goal_functional = error_estimator.goal_functional.reference.mean_drag;
		fem_goal_functional = error_estimator.goal_functional.fem.mean_drag;

		DTM::pout << "---------------------------" << std::endl;
		DTM::pout << "Mean drag:" << std::endl;
		DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
		DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
		DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
		DTM::pout << "---------------------------" << std::endl << std::endl;
	}

	if ( parameter_set->dwr.functional.mean_lift){
		reference_goal_functional = error_estimator.goal_functional.reference.mean_lift;
		fem_goal_functional = error_estimator.goal_functional.fem.mean_lift;

		DTM::pout << "---------------------------" << std::endl;
		DTM::pout << "Mean lift:" << std::endl;
		DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
		DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
		DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
		DTM::pout << "---------------------------" << std::endl << std::endl;
	}

	if ( parameter_set->dwr.functional.mean_pdiff){
		reference_goal_functional = error_estimator.goal_functional.reference.mean_pdiff;
		fem_goal_functional = error_estimator.goal_functional.fem.mean_pdiff;

		DTM::pout << "---------------------------" << std::endl;
		DTM::pout << "Mean pressure difference:" << std::endl;
		DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
		DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
		DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
		DTM::pout << "---------------------------" << std::endl << std::endl;
	}

	if ( parameter_set->dwr.functional.mean_vorticity){
		reference_goal_functional = error_estimator.goal_functional.reference.mean_vorticity;
		fem_goal_functional = error_estimator.goal_functional.fem.mean_vorticity;

		DTM::pout << "---------------------------" << std::endl;
		DTM::pout << "Mean vorticity:" << std::endl;
		DTM::pout << "	J(u)               = " << std::setprecision(16) << reference_goal_functional << std::endl;
		DTM::pout << "	J(u_{kh})          = " << std::setprecision(16) << fem_goal_functional << std::endl;
		DTM::pout << "	|J(u) - J(u_{kh})| = " << std::setprecision(16) << std::abs(reference_goal_functional - fem_goal_functional) << std::endl;
		DTM::pout << "---------------------------" << std::endl << std::endl;
	}
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
	// INIT PRIMAL DATA OUTPUT
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

	if(parameter_set->dwr.functional.mean_vorticity){
		primal.vorticity_data_output = std::make_shared<DTM::DataOutput<dim>>();
		std::vector<std::string> dfn;
		dfn.push_back("vorticity");

		primal.vorticity_data_output->set_data_field_names(dfn);
		std::vector< dealii::DataComponentInterpretation::DataComponentInterpretation > dci;
		dci.push_back(dealii::DataComponentInterpretation::component_is_scalar);
		primal.vorticity_data_output->set_data_component_interpretation_field(dci);

		primal.vorticity_data_output->set_data_output_patches(
			parameter_set->data_output.primal.patches
		);
	}

}


template<int dim>
void
Fluid<dim>::
primal_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
	const unsigned int dwr_loop) {
	// triggered output mode
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(u->x[0].use_count(),dealii::ExcNotInitialized());

//	primal.data_output->set_DoF_data(
//		slab->space.primal.fe_info->dof,
//		slab->space.primal.fe_info->partitioning_locally_owned_dofs
//	);
//
// 	auto u_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
// 	u_trigger->reinit(*slab->space.primal.fe_info->locally_owned_dofs,
// 					  *slab->space.primal.fe_info->locally_relevant_dofs,
//					  mpi_comm
// 	);

	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
//
//	{
//		// fe face values time: time face (I_n) information
//		dealii::FEValues<1> fe_face_values_time(
//			*slab->time.primal.fe_info->mapping,
//			*slab->time.primal.fe_info->fe,
//			dealii::QGaussLobatto<1>(2),
//			dealii::update_quadrature_points
//		);
//
//		Assert(
//			slab->time.primal.fe_info->dof.use_count(),
//			dealii::ExcNotInitialized()
//		);
//
//		auto cell_time = slab->time.primal.fe_info->dof->begin_active();
//		auto endc_time = slab->time.primal.fe_info->dof->end();
//
//		for ( ; cell_time != endc_time; ++cell_time) {
//			fe_face_values_time.reinit(cell_time);
//
//			////////////////////////////////////////////////////////////////////
//			// construct quadrature for data output,
//			// if triggered output time values are inside this time element
//			//
//
//			auto t_m = fe_face_values_time.quadrature_point(0)[0];
//			auto t_n = fe_face_values_time.quadrature_point(1)[0];
//			auto tau = t_n-t_m;
//
//			std::list<double> output_times;
//
//			if (primal.data_output_time_value < t_m) {
//				primal.data_output_time_value = t_m;
//			}
//
//			for ( ; (primal.data_output_time_value <= t_n) ||
//				(
//					(primal.data_output_time_value > t_n) &&
//					(std::abs(primal.data_output_time_value - t_n) < tau*1e-12)
//				); ) {
//				output_times.push_back(primal.data_output_time_value);
//				primal.data_output_time_value += primal.data_output_trigger;
//			}
//
//			if (output_times.size() && output_times.back() > t_n) {
//				output_times.back() = t_n;
//			}
//
//			if ((output_times.size() > 1) &&
//				(output_times.back() == *std::next(output_times.rbegin()))) {
//				// remove the last entry, iff doubled
//				output_times.pop_back();
//			}
//
//			// convert container
//			if (!output_times.size()) {
//				continue;
//			}
//
//			std::vector< dealii::Point<1> > output_time_points(output_times.size());
//			{
//				auto time{output_times.begin()};
//				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
//					double t_trigger{*time};
//					output_time_points[q][0] = (t_trigger-t_m)/tau;
//				}
//
//				if (output_time_points[0][0] < 0) {
//					output_time_points[0][0] = 0;
//				}
//
//				if (output_time_points[output_time_points.size()-1][0] > 1) {
//					output_time_points[output_time_points.size()-1][0] = 1;
//				}
//			}
//
//			dealii::Quadrature<1> quad_time(output_time_points);
//
//			// create fe values
//			dealii::FEValues<1> fe_values_time(
//				*slab->time.primal.fe_info->mapping,
//				*slab->time.primal.fe_info->fe,
//				quad_time,
//				dealii::update_values |
//				dealii::update_quadrature_points
//			);
//
//			fe_values_time.reinit(cell_time);
//
//			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.primal.fe_info->fe->dofs_per_cell);
//			cell_time->get_dof_indices(local_dof_indices);
//
//			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
// 				*u_trigger = 0.;
//
// 				// evaluate solution for t_q
// 				for (
// 					unsigned int jj{0};
// 					jj < slab->time.primal.fe_info->fe->dofs_per_cell; ++jj) {
// 				for (
// 					dealii::types::global_dof_index i{0};
// 					i < slab->space.primal.fe_info->dof->n_dofs(); ++i) {
// 					(*u_trigger)[i] += (*u->x[0])[
// 						i
// 						// time offset
// 						+ slab->space.primal.fe_info->dof->n_dofs() *
// 							local_dof_indices[jj]
// 					] * fe_values_time.shape_value(jj,qt);
// 				}}
//
////				std::cout
////					<< "output generated for t = "
////					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
////					<< std::endl;
//
//				primal.data_output->write_data(
//					filename.str(),
//					u_trigger,
//					primal.data_postprocessor,
//					fe_values_time.quadrature_point(qt)[0] // t_trigger
//				);
//			}
//		}
//	}
	
	// check if data for t=T was written
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &vort,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());


	primal.data_output->set_DoF_data(
		slab->space.primal.fe_info->dof
	);

	if(parameter_set->dwr.functional.mean_vorticity){
		primal.vorticity_data_output->set_DoF_data(
			slab->space.vorticity.fe_info->dof
		);
	}

	auto u_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
 	u_trigger->reinit(
 			*slab->space.primal.fe_info->locally_owned_dofs,
 			*slab->space.primal.fe_info->locally_relevant_dofs,
 			mpi_comm);

	auto owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	owned_tmp->reinit(*slab->space.primal.fe_info->locally_owned_dofs,
			*slab->space.primal.fe_info->locally_owned_dofs,
			mpi_comm);

	auto vort_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector> ();
	auto vort_tmp = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
	if (parameter_set->dwr.functional.mean_vorticity){
		vort_trigger->reinit(
			*slab->space.vorticity.fe_info->locally_owned_dofs,
			*slab->space.vorticity.fe_info->locally_relevant_dofs,
			mpi_comm
		);

		vort_tmp->reinit(
				*slab->space.vorticity.fe_info->locally_owned_dofs,
				mpi_comm
		);

	}

	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	std::ostringstream vortname;
	vortname
		<< "vorticity-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// create fe values
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time;
		if ( !(parameter_set->
				fe.primal.convection.time_type_support_points
				.compare("Gauss")) ) {

			fe_quad_time =
					std::make_shared< dealii::QGauss<1> > (
							(parameter_set->fe.primal.convection.r + 1)
					);

		} else if ( !(parameter_set->
				fe.primal.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ){

			if (parameter_set->fe.primal.convection.r < 1){
				fe_quad_time =
						std::make_shared< QRightBox<1> > ();
			} else {
				fe_quad_time =
						std::make_shared< dealii::QGaussLobatto<1> > (
								(parameter_set->fe.primal.convection.r + 1)
						);
			}
		}

		AssertThrow(
			fe_quad_time.use_count(),
			dealii::ExcMessage(
				"FE time: (primal) convection b support points invalid"
			)
		);

		dealii::FEValues<1> fe_values_time(
			*slab->time.primal.fe_info->mapping,
			*slab->time.primal.fe_info->fe,
			*fe_quad_time,
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
 				*owned_tmp = 0.;

 				// evaluate solution for t_q

 				dealii::IndexSet::ElementIterator lri = slab->space.primal.fe_info->locally_owned_dofs->begin();
 				dealii::IndexSet::ElementIterator lre = slab->space.primal.fe_info->locally_owned_dofs->end();

 				for (; lri!= lre; lri++) {
 					for ( unsigned int jj{0}; jj < slab->time.primal.fe_info->fe->dofs_per_cell; jj++){
 						(*owned_tmp)[*lri] +=
 							(*u->x[0])[*lri
									   // time offset
									   + slab->space.primal.fe_info->dof->n_dofs() *
									   local_dof_indices_time[jj]
									  ] * fe_values_time.shape_value(jj,qt);
					}
 				}
 				owned_tmp->compress(dealii::VectorOperation::add);
 				slab->space.primal.fe_info->hanging_node_constraints->distribute(*owned_tmp);

 				*u_trigger = *owned_tmp;

 				primal.data_output->write_data(
					filename.str(),
					u_trigger,
//					primal.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);

 				if(parameter_set->dwr.functional.mean_vorticity){
 					*vort_trigger = 0.;
 					*vort_tmp = 0.;

 	 				dealii::IndexSet::ElementIterator lri = slab->space.vorticity.fe_info->locally_owned_dofs->begin();
 	 				dealii::IndexSet::ElementIterator lre = slab->space.vorticity.fe_info->locally_owned_dofs->end();

// 	 				vort->x[0]->print(std::cout);
 	 				for (; lri!= lre; lri++) {
 	 					for ( unsigned int jj{0}; jj < slab->time.primal.fe_info->fe->dofs_per_cell; jj++){
 	 						(*vort_tmp)[*lri] +=
 	 							(*vort->x[0])[*lri
 										   // time offset
 										   + slab->space.vorticity.fe_info->dof->n_dofs() *
 										   local_dof_indices_time[jj]
 										  ] * fe_values_time.shape_value(jj,qt);
 						}
 	 				}
 	 				vort_tmp->compress(dealii::VectorOperation::add);

 	 				*vort_trigger = *vort_tmp;

 	 				primal.vorticity_data_output->write_data(
 	 					vortname.str(),
						vort_trigger,
						fe_values_time.quadrature_point(qt)[0]
 	 				);
 				}
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &x,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &vort,
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
		primal_do_data_output_on_slab_Qn_mode(slab, x, vort,dwr_loop);
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
		std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();

	dual.storage.z->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);
}


template<int dim>
void
Fluid<dim>::
dual_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z
) {
	for (unsigned int j{0}; j < z->x.size(); ++j) {
		z->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

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
			*slab->spacetime.dual.locally_owned_dofs,
			*slab->spacetime.dual.locally_relevant_dofs,
			mpi_comm
		);
	}
}


template<int dim>
void
Fluid<dim>::
dual_assemble_system(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u
) {
	// ASSEMBLY SPACE-TIME OPERATOR MATRIX /////////////////////////////////////

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
		Assert(u.use_count(), dealii::ExcNotInitialized());
		Assert(u->size(), dealii::ExcNotInitialized());
		dual_assembler.assemble(
			dual.L,
			slab,
			u,
			(parameter_set->problem.compare("Navier-Stokes") == 0)
		);

		DTM::pout << " (done)" << std::endl;
	}
}


template<int dim>
void
Fluid<dim>::
dual_assemble_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u
) {
	// NOTE: for nonlinear goal functionals, we need also the primal solution u for this function

	// ASSEMBLY SPACE-TIME OPERATOR: FinalValue VECTOR ///////////////////////
	dual.Mzn = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Mzn->reinit(
		*slab->spacetime.dual.locally_owned_dofs,
		mpi_comm
	);
	*dual.Mzn = 0.;

	Assert(dual.relevant_tmp.use_count(),dealii::ExcNotInitialized());
	*dual.relevant_tmp = *dual.zn;
	{
		FVDualAssembler<dim> dual_assembler;

		DTM::pout << "dwr-instatfluid: assemble space-time slab final value vector...";
		Assert(dual.zn.use_count(), dealii::ExcNotInitialized());
		Assert(dual.Mzn.use_count(), dealii::ExcNotInitialized());
		dual_assembler.assemble(
			dual.relevant_tmp,
			dual.Mzn,
			slab
		);

		DTM::pout << " (done)" << std::endl;
	}

	// ASSEMBLY SPACE-TIME OPERATOR: GOAL FUNCTIONAL VECTOR //////////////////////////////
	dual.Je = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Je->reinit(
		*slab->spacetime.dual.locally_owned_dofs,
		mpi_comm
	);
	*dual.Je = 0.;

	if(!parameter_set->dwr.goal.type.compare("mean_drag")){
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
	else if (!parameter_set->dwr.goal.type.compare("mean_vorticity")){
		Je_MeanVorticity_Assembler<dim> Je_assembler;
		DTM::pout << "dwr-instatfluid: assemble space-time slab Je_meanviscosity vector...";
		Je_assembler.assemble(
			dual.Je,
			slab,
			u,
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
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z
) {
	Assert(dual.L.use_count(), dealii::ExcNotInitialized());
	Assert(dual.Mzn.use_count(), dealii::ExcNotInitialized());
	Assert(dual.Je.use_count(), dealii::ExcNotInitialized());

	dual.b = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	Assert(
		slab->space.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.b->reinit(
		*slab->spacetime.dual.locally_owned_dofs,
		mpi_comm
	);

	*dual.b = 0.;

	dual.b->add(1., *dual.Mzn);
	dual.b->add(1., *dual.Je);

	////////////////////////////////////////////////////////////////////////////
	// solve linear system with direct solver
	//

	DTM::pout << "dwr-instatfluid: setup direct lss and solve...";

	if (!parameter_set->problem.compare("Navier-Stokes") ||
		!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") ||
		std::abs(slab->t_n-parameter_set->time.fluid.T) < 0.5*slab->tau_n())
	{
		dual.iA = nullptr;
		dual.iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (*dual.sc,*dual.ad);

		dual.iA->initialize(*dual.L);
	}

	dual.iA->solve(*dual.owned_tmp, *dual.b);

	DTM::pout << " (done)" << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// distribute hanging nodes constraints on solution
	//

	slab->spacetime.dual.constraints->distribute(
		*dual.owned_tmp
	);

	*z->x[0] = *dual.owned_tmp;
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
	// storage: init iterators to storage_data_trilinos_vectors
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

		//dual dof partitioning
		{
			slab->spacetime.dual.locally_owned_dofs =
					std::make_shared<dealii::IndexSet> (
							idealii::SlabDoFTools::extract_locally_owned_dofs(
									slab->space.dual.fe_info->dof,
									slab->time.dual.fe_info->dof
							)
					);

			slab->spacetime.dual.locally_relevant_dofs =
					std::make_shared<dealii::IndexSet> (
							idealii::SlabDoFTools::extract_locally_relevant_dofs(
									slab->space.dual.fe_info->dof,
									slab->time.dual.fe_info->dof
							)
					);
		}


		{
			slab->spacetime.dual.constraints =
				std::make_shared< dealii::AffineConstraints<double> > ();

			idealii::SlabDoFTools::make_spacetime_constraints(
				slab->space.dual.fe_info->locally_relevant_dofs,
				slab->space.dual.fe_info->constraints, // space constraints
				slab->space.dual.fe_info->dof->n_dofs(),
				slab->time.dual.fe_info->dof->n_dofs(),
				slab->spacetime.dual.locally_relevant_dofs,
				slab->spacetime.dual.constraints
			);

			slab->spacetime.dual.constraints->close();
		}

		if (!parameter_set->problem.compare("Navier-Stokes") ||
			!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") ||
			std::abs(slab->t_n-parameter_set->time.fluid.T) < 0.5*slab->tau_n())
		{
			dual.iA = std::make_shared<dealii::TrilinosWrappers::SolverDirect> (*dual.sc,*dual.ad);
			dual.L = std::make_shared<dealii::TrilinosWrappers::SparseMatrix> ();
			grid->create_sparsity_pattern_dual_on_slab(slab,dual.L);
		}

		dual_reinit_storage_on_slab(slab, z);


		{
			// error indicators
			grid->initialize_pu_grid_components_on_slab(slab);
			grid->distribute_pu_on_slab(slab);
			// pu dof partitioning
			{
				slab->spacetime.pu.locally_owned_dofs =
						std::make_shared<dealii::IndexSet> (
								idealii::SlabDoFTools::extract_locally_owned_dofs(
										slab->space.pu.fe_info->dof,
										slab->time.pu.fe_info->dof
								)
						);

				slab->spacetime.pu.locally_relevant_dofs =
						std::make_shared<dealii::IndexSet> (
								idealii::SlabDoFTools::extract_locally_relevant_dofs(
										slab->space.pu.fe_info->dof,
										slab->time.pu.fe_info->dof
								)
						);
			}

			{
				slab->spacetime.pu.constraints =
					std::make_shared< dealii::AffineConstraints<double> > ();

				idealii::SlabDoFTools::make_spacetime_constraints(
					slab->space.pu.fe_info->locally_relevant_dofs,
					slab->space.pu.fe_info->constraints, // space constraints
					slab->space.pu.fe_info->dof->n_dofs(),
					slab->time.pu.fe_info->dof->n_dofs(),
					slab->spacetime.pu.locally_relevant_dofs,
					slab->spacetime.pu.constraints
				);

				slab->spacetime.dual.constraints->close();
			}


			eta_reinit_storage_on_slab(
				slab,
				eta_space,
				eta_time
			);

			error_estimator.pu_dwr = std::make_shared< fluid::cGp_dGr::cGq_dGs::ErrorEstimator<dim> > ();
			// set the important variables for the error estimator
			error_estimator.pu_dwr->init(
				mpi_comm,
				function.viscosity,
				grid,
				parameter_set->fe.symmetric_stress,
				parameter_set->dwr.replace_linearization_points,
				parameter_set->dwr.replace_weights,
				parameter_set->fe.primal_order,
				parameter_set->fe.dual_order,
				(parameter_set->problem.compare("Navier-Stokes") == 0)
			);
		}


		dual.zn = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		dual.zn->reinit(*slab->space.dual.fe_info->locally_owned_dofs,mpi_comm);
		*dual.zn = 0.;


		dual.relevant_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		dual.relevant_tmp->reinit(*slab->space.dual.fe_info->locally_owned_dofs,
							   *slab->space.dual.fe_info->locally_relevant_dofs,
							   mpi_comm);

		dual.owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
		dual.owned_tmp->reinit(*slab->spacetime.dual.locally_owned_dofs,mpi_comm);
		if (n == N) {
			////////////////////////////////////////////////////////////////////////////
			// interpolate (or project) initial value(s)
			//

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

			auto zm_rel = std::make_shared< dealii::TrilinosWrappers::MPI::Vector> ();
			zm_rel->reinit(*std::next(slab)->space.dual.fe_info->locally_owned_dofs,
					       *std::next(slab)->space.dual.fe_info->locally_relevant_dofs,
						   mpi_comm);

			*zm_rel = *dual.zm;
			// for n < N interpolate between two (different) spatial meshes
			// the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n+1}:
				*std::next(slab)->space.dual.fe_info->dof,
				*zm_rel,
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
		if (!parameter_set->problem.compare("Navier-Stokes") ||
			!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive") ||
			std::abs(slab->t_n-parameter_set->time.fluid.T) < 0.5*slab->tau_n())
		{
			dual_assemble_system(slab, u->x[0]);
		}
		dual_assemble_rhs(slab, u->x[0]);

		// solve slab problem (i.e. apply boundary values and solve for z0)
		dual_solve_slab_problem(slab, z);

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
			error_estimator.pu_dwr->estimate_on_slab(std::next(slab), std::next(u), std::next(um), std::next(z), std::next(eta_space), std::next(eta_time));
			std::next(u)->x[0]->clear();
			std::next(um)->x[0]->clear();
			std::next(slab)->space.primal.fe_info->dof->clear();

			//TODO: some Segfault here, no idea why
//			// apply B. Endtmayer's post processing of the error indicators
//			// see https://arxiv.org/pdf/1811.07586.pdf (Figure 1)
//			auto space_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
//			auto time_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
//
//			space_relevant ->reinit(
//					*std::next(slab)->spacetime.pu.locally_owned_dofs,
//					*std::next(slab)->spacetime.pu.locally_relevant_dofs,
//					mpi_comm
//			);
//			time_relevant ->reinit(
//					*std::next(slab)->spacetime.pu.locally_owned_dofs,
//					*std::next(slab)->spacetime.pu.locally_relevant_dofs,
//					mpi_comm
//			);
//
//			*space_relevant = *std::next(eta_space)->x[0];
//			*time_relevant = *std::next(eta_time)->x[0];
//
//			for (auto line : std::next(slab)->spacetime.pu.constraints->get_lines())
//			{
//				//go over all line entries
//				for (unsigned int i=0; i<std::pow(2, dim-1); ++i){
//					if (std::next(slab)->spacetime.pu.locally_owned_dofs->is_element(line.entries[i].first)){
//						(*std::next(eta_space)->x[0])[line.entries[i].first]
//						  += (1. / std::pow(2, dim-1))*(*space_relevant)[line.index];
//
//						(*std::next(eta_time)->x[0])[line.entries[i].first]
//						  += (1. / std::pow(2, dim-1))*(*time_relevant)[line.index];
//
//					}
//				}
//				if (std::next(slab)->spacetime.pu.locally_owned_dofs->is_element(line.index)){
//					(*std::next(eta_space)->x[0])[line.index] = 0.;
//					(*std::next(eta_time)->x[0])[line.index] = 0.;
//				}
//			}
		}
		// evaluate error on first slab
		if (n == 1)
		{

			error_estimator.pu_dwr->estimate_on_slab(slab, u, um, z, eta_space, eta_time);
			u->x[0]->clear();
			um->x[0]->clear();
			slab->space.primal.fe_info->dof->clear();

//			// apply B. Endtmayer's post processing of the error indicators
//			// see https://arxiv.org/pdf/1811.07586.pdf (Figure 1)
//			auto space_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
//			auto time_relevant = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
//
//			std::cout << "reiniting relevant vectors" << std::endl;
//			space_relevant ->reinit(
//					*slab->spacetime.pu.locally_owned_dofs,
//					*slab->spacetime.pu.locally_relevant_dofs,
//					mpi_comm
//			);
//			time_relevant ->reinit(
//					*slab->spacetime.pu.locally_owned_dofs,
//					*slab->spacetime.pu.locally_relevant_dofs,
//					mpi_comm
//			);
//
//			std::cout << "communicating estimators" << std::endl;
//			*space_relevant = *eta_space->x[0];
//			*time_relevant = *eta_time->x[0];
//
//			std::cout << "Starting B.E. postprocessing" << std::endl;
//			for (auto line : slab->spacetime.pu.constraints->get_lines())
//			{
//				std::cout << "line " << line.index << " on rank "
//						  << dealii::Utilities::MPI::this_mpi_process(mpi_comm)
//						  << std::endl;
//				//go over all line entries
//				for (unsigned int i=0; i<std::pow(2, dim-1); ++i){
//					if (slab->spacetime.pu.locally_owned_dofs->is_element(
//							line.entries[i].first)){
//						(*eta_space->x[0])[line.entries[i].first]
//						  += (1. / std::pow(2, dim-1))*(*space_relevant)[line.index];
//
//						(*eta_time->x[0])[line.entries[i].first]
//						  += (1. / std::pow(2, dim-1))*(*time_relevant)[line.index];
//
//					}
//				}
//				std::cout << "entries done, zeroing index on rank "
//						  << dealii::Utilities::MPI::this_mpi_process(mpi_comm)
//						  << std::endl;
//				if (slab->spacetime.pu.locally_owned_dofs->is_element(line.index)){
//					(*eta_space->x[0])[line.index] = 0.;
//					(*eta_time->x[0])[line.index] = 0.;
//				}
//			}
		}

		////////////////////////////////////////////////////////////////////////
		// do postprocessing on the solution
		//

		// evaluate solution z(t_m)
		dual.zm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

		dual.zm->reinit(*slab->space.dual.fe_info->locally_owned_dofs,mpi_comm);
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
				{
					dealii::IndexSet::ElementIterator lri = slab->space.dual.fe_info->locally_owned_dofs->begin();
					dealii::IndexSet::ElementIterator lre = slab->space.dual.fe_info->locally_owned_dofs->end();

					for (; lri != lre ; lri++ ){
						(*dual.zm)[*lri] += (*z->x[0])[
							*lri
							// time offset
							+ slab->space.dual.fe_info->dof->n_dofs() *
							(cell_time->index() * slab->time.dual.fe_info->fe->dofs_per_cell)
							// local in time dof
							+ slab->space.dual.fe_info->dof->n_dofs() * jj
							] * fe_face_values_time.shape_value(jj,0);
					}

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

//		dual.L = nullptr;
		dual.b = nullptr;

		dual.Mzn = nullptr;
		dual.Je = nullptr;

		grid->clear_dual_on_slab(slab);

		if (n < N)
		{
			std::next(z)->x[0]->clear();
			std::next(slab)->space.dual.fe_info->dof->clear();
		}
		if (n == 1)
		{
			z->x[0]->clear();
			slab->space.dual.fe_info->dof->clear();
		}

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
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
	const unsigned int dwr_loop) {
	// TODO: might need to be debugged; adapted from primal_do_data_output_on_slab()
	std::cout << "output used in non Qn mode!" << std::endl;
//
//	// triggered output mode
	Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(z->x[0].use_count(),dealii::ExcNotInitialized());
//
//	dual.data_output->set_DoF_data(
//		slab->space.dual.fe_info->dof,
//		slab->space.dual.fe_info->partitioning_locally_owned_dofs
//	);
//
// 	auto z_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
// 	z_trigger->reinit(
// 		slab->space.dual.fe_info->dof->n_dofs()
// 	);
//
	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
//
//	{
//		// fe face values time: time face (I_n) information
//		dealii::FEValues<1> fe_face_values_time(
//			*slab->time.dual.fe_info->mapping,
//			*slab->time.dual.fe_info->fe,
//			dealii::QGaussLobatto<1>(2),
//			dealii::update_quadrature_points
//		);
//
//		Assert(
//			slab->time.dual.fe_info->dof.use_count(),
//			dealii::ExcNotInitialized()
//		);
//
//		auto cell_time = slab->time.dual.fe_info->dof->begin_active();
//		auto endc_time = slab->time.dual.fe_info->dof->end();
//
//		for ( ; cell_time != endc_time; ++cell_time) {
//			fe_face_values_time.reinit(cell_time);
//
//			////////////////////////////////////////////////////////////////////
//			// construct quadrature for data output,
//			// if triggered output time values are inside this time element
//			//
//
//			auto t_m = fe_face_values_time.quadrature_point(0)[0];
//			auto t_n = fe_face_values_time.quadrature_point(1)[0];
//			auto tau = t_n-t_m;
//
//			std::list<double> output_times;
//
//			if (dual.data_output_time_value < t_m) {
//				dual.data_output_time_value = t_m;
//			}
//
//			for ( ; (dual.data_output_time_value <= t_n) ||
//				(
//					(dual.data_output_time_value > t_n) &&
//					(std::abs(dual.data_output_time_value - t_n) < tau*1e-12)
//				); ) {
//				output_times.push_back(dual.data_output_time_value);
//				dual.data_output_time_value += dual.data_output_trigger;
//			}
//
//			if (output_times.size() && output_times.back() > t_n) {
//				output_times.back() = t_n;
//			}
//
//			if ((output_times.size() > 1) &&
//				(output_times.back() == *std::next(output_times.rbegin()))) {
//				// remove the last entry, iff doubled
//				output_times.pop_back();
//			}
//
//			// convert container
//			if (!output_times.size()) {
//				continue;
//			}
//
//			std::vector< dealii::Point<1> > output_time_points(output_times.size());
//			{
//				auto time{output_times.begin()};
//				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
//					double t_trigger{*time};
//					output_time_points[q][0] = (t_trigger-t_m)/tau;
//				}
//
//				if (output_time_points[0][0] < 0) {
//					output_time_points[0][0] = 0;
//				}
//
//				if (output_time_points[output_time_points.size()-1][0] > 1) {
//					output_time_points[output_time_points.size()-1][0] = 1;
//				}
//			}
//
//			dealii::Quadrature<1> quad_time(output_time_points);
//
//			// create fe values
//			dealii::FEValues<1> fe_values_time(
//				*slab->time.dual.fe_info->mapping,
//				*slab->time.dual.fe_info->fe,
//				quad_time,
//				dealii::update_values |
//				dealii::update_quadrature_points
//			);
//
//			fe_values_time.reinit(cell_time);
//
//			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.dual.fe_info->fe->dofs_per_cell);
//			cell_time->get_dof_indices(local_dof_indices);
//
//			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
// 				*z_trigger = 0.;
//
// 				// evaluate solution for t_q
// 				for (
// 					unsigned int jj{0};
// 					jj < slab->time.dual.fe_info->fe->dofs_per_cell; ++jj) {
// 				for (
// 					dealii::types::global_dof_index i{0};
// 					i < slab->space.dual.fe_info->dof->n_dofs(); ++i) {
// 					(*z_trigger)[i] += (*z->x[0])[
// 						i
// 						// time offset
// 						+ slab->space.dual.fe_info->dof->n_dofs() *
// 							local_dof_indices[jj]
// 					] * fe_values_time.shape_value(jj,qt);
// 				}}
//
////				std::cout
////					<< "dual output generated for t = "
////					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
////					<< std::endl;
//
//				dual.data_output->write_data(
//					filename.str(),
//					z_trigger,
//					dual.data_postprocessor,
//					fe_values_time.quadrature_point(qt)[0] // t_trigger
//				);
//			}
//		}
//	}
}


template<int dim>
void
Fluid<dim>::
dual_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
	const unsigned int dwr_loop) {
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	dual.data_output->set_DoF_data(
		slab->space.dual.fe_info->dof
	);

	auto z_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
 	z_trigger->reinit(
 			*slab->space.dual.fe_info->locally_owned_dofs,
			*slab->space.dual.fe_info->locally_relevant_dofs,
			mpi_comm);

	auto owned_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
 	owned_tmp->reinit(
 			*slab->space.dual.fe_info->locally_owned_dofs,
			mpi_comm);

	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// create fe values
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time;
		if ( !(parameter_set->
				fe.dual.convection.time_type_support_points
				.compare("Gauss")) ) {

			fe_quad_time =
					std::make_shared< dealii::QGauss<1> > (
							(parameter_set->fe.primal.convection.r + 1)
					);

		} else if ( !(parameter_set->
				fe.dual.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ){

			if (parameter_set->fe.dual.convection.r < 1){
				fe_quad_time =
						std::make_shared< QRightBox<1> > ();
			} else {
				fe_quad_time =
						std::make_shared< dealii::QGaussLobatto<1> > (
								(parameter_set->fe.dual.convection.r + 1)
						);
			}
		}

		AssertThrow(
			fe_quad_time.use_count(),
			dealii::ExcMessage(
				"FE time: (dual) convection b support points invalid"
			)
		);


		dealii::FEValues<1> fe_values_time(
			*slab->time.dual.fe_info->mapping,
			*slab->time.dual.fe_info->fe,
			dealii::QGauss<1>(slab->time.dual.fe_info->fe->tensor_degree()+1), // here
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
 				*owned_tmp = 0.;

 				// evaluate solution for t_q


 				dealii::IndexSet::ElementIterator lri = slab->space.dual.fe_info->locally_owned_dofs->begin();
 				dealii::IndexSet::ElementIterator lre = slab->space.dual.fe_info->locally_owned_dofs->end();


 				for (; lri!= lre; lri++) {
 					for (
 							unsigned int jj{0};
 							jj < slab->time.dual.fe_info->fe->dofs_per_cell; ++jj) {
 					(*owned_tmp)[*lri] += (*z->x[0])[
 						*lri
 						// time offset
 						+ slab->space.dual.fe_info->dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 					}
 				}

 				*z_trigger = *owned_tmp;

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
dual_do_data_output(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,
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

	if (dealii::Utilities::MPI::this_mpi_process(mpi_comm)==0) {
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
}


////////////////////////////////////////////////////////////////////////////////
// functional values
//
template<int dim>
void
Fluid<dim>::
compute_functional_values(
		const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
		const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &vort,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	Assert(dim==2 || dim==3, dealii::ExcNotImplemented());
	
	// un := u at quadrature point, i.e. un = u(.,t_n)
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un =
			std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();

	un->reinit(*slab->space.primal.fe_info->locally_owned_dofs,mpi_comm);


	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un_rel =
			std::make_shared< dealii::TrilinosWrappers::MPI::Vector>();

	un_rel->reinit(*slab->space.primal.fe_info->locally_owned_dofs,
				   *slab->space.primal.fe_info->locally_relevant_dofs,
				   mpi_comm);

	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > vortn =
				std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();

	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > vortn_rel =
					std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();


	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > vort_owned =
				std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	if ( parameter_set->dwr.functional.mean_vorticity){
		vortn->reinit(*slab->space.vorticity.fe_info->locally_owned_dofs,mpi_comm);

		vortn_rel->reinit(*slab->space.vorticity.fe_info->locally_owned_dofs,
					   *slab->space.vorticity.fe_info->locally_relevant_dofs,
					   mpi_comm);

		vort_owned->reinit(*slab->spacetime.vorticity.locally_owned_dofs,mpi_comm);
	}
	// create vectors with functional values at quadrature points
	// format: (time, value)
	std::vector< std::tuple< double, double > > pressure_values;
	std::vector< std::tuple< double, double > > drag_values;
	std::vector< std::tuple< double, double > > lift_values;
	std::vector< std::tuple< double, double > > vorticity_values;

	auto cell_time = slab->time.primal.fe_info->dof->begin_active();
	auto endc_time = slab->time.primal.fe_info->dof->end();

	std::shared_ptr< dealii::Quadrature<1> > quad_time;
	{
		if ( !(parameter_set->
			fe.primal.convection.time_type_support_points
			.compare("Gauss")) ) {
			quad_time =
			std::make_shared< dealii::QGauss<1> > (
				(parameter_set->fe.primal.convection.r + 1)
			);
		} else if ( !(parameter_set->
				fe.primal.convection.time_type_support_points
				.compare("Gauss-Lobatto")) ){
			if (parameter_set->fe.primal.convection.r < 1){
				quad_time = std::make_shared< QRightBox<1> > ();
			} else {
				quad_time =
						std::make_shared< dealii::QGaussLobatto<1> > (
								(parameter_set->fe.primal.convection.r + 1)
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
			un->reinit(*slab->space.primal.fe_info->locally_owned_dofs,mpi_comm);

			double tn = fe_values_time.quadrature_point(ii)[0];

			// compute solution un at time tn
		    dealii::IndexSet::ElementIterator lri = slab->space.primal.fe_info->locally_owned_dofs->begin();
		    dealii::IndexSet::ElementIterator lre = slab->space.primal.fe_info->locally_owned_dofs->end();
			for (; lri != lre ;lri++)
			{
				(*un)[*lri] += (*u->x[0])[
					*lri
					// time offset
					+ slab->space.primal.fe_info->dof->n_dofs() *
						(cell_time->index() * slab->time.primal.fe_info->fe->dofs_per_cell)
					// local in time dof
					+ slab->space.primal.fe_info->dof->n_dofs() * ii
				];
			}

			*un_rel = *un;
			////////////////////////////////////
			// compute functional values of un
			////////////////////////////////////

			double scaling = 1. / (parameter_set->time.fluid.T - parameter_set->time.fluid.t0);

			if ( parameter_set->dwr.functional.mean_pdiff){
				// pressure
				dealii::Point<dim> M;

				if (dim==2) {
					M[0] = 0.15;
					M[1] = 0.20;
				} else if (dim ==3) {
					M[0] = 0.45;
					M[1] = 0.20;
					M[2] = 0.205;
				}


				double pressure_front = compute_pressure(
					M,
					un_rel,
					slab
				); // pressure - left  point on circle


				if (dim==2) {
					M[0] = 0.25;
					M[1] = 0.20;
				} else if (dim == 3){
					M[0] = 0.55;
					M[1] = 0.20;
					M[2] = 0.205;
				}

				double pressure_back = compute_pressure(
					M,
					un_rel,
					slab
				); // pressure - right point on circle

				// save pressure difference to the vector pressure_values
				double pressure_diff = pressure_front - pressure_back;
				pressure_values.push_back(std::make_tuple(tn, pressure_diff));
				error_estimator.goal_functional.fem.mean_pdiff += scaling * pressure_diff * fe_values_time.JxW(ii);
			}
			////////////////////////////////////
			// drag and lift
			if (parameter_set->dwr.functional.mean_drag || parameter_set->dwr.functional.mean_lift){
				// Compute drag and lift via line integral
				dealii::Tensor<1, dim> drag_lift_value;
				compute_drag_lift_tensor(
					un_rel,
					slab,
					drag_lift_value
				);
				if ( parameter_set->dwr.functional.mean_drag){
					drag_values.push_back(std::make_tuple(tn, drag_lift_value[0]));
					error_estimator.goal_functional.fem.mean_drag += scaling * drag_lift_value[0] * fe_values_time.JxW(ii);
				}
				if ( parameter_set->dwr.functional.mean_lift){
					lift_values.push_back(std::make_tuple(tn, drag_lift_value[1]));
					error_estimator.goal_functional.fem.mean_lift += scaling * drag_lift_value[1] * fe_values_time.JxW(ii);
				}
			}

			if (parameter_set->dwr.functional.mean_vorticity) {
				//Compute vorticity
				double mv1 = compute_vorticity(un_rel, slab, vortn);
//				std::cout << ii << " vorticity vector" << std::endl;
				vortn->compress(dealii::VectorOperation::add);
				double mv2 = dealii::Utilities::MPI::sum(mv1, mpi_comm);
//				vortn->print(std::cout);
				//Compute L2 Norm
				*vortn_rel = *vortn;
//				double mean_vort = vortn->l2_norm();
//				double mean_vort2=std::pow(mean_vort,2);
//				std::cout << "in process calc: " << mv1
//						  << "\n L2-Norm: " <<mean_vort
//						  << "\n L2_Norm^2: " << mean_vort2
//						  << std::endl;

 				vorticity_values.push_back(std::make_tuple(tn, mv2));
				error_estimator.goal_functional.fem.mean_vorticity += scaling * mv2 * fe_values_time.JxW(ii);
				//Save result to storage for output
				dealii::IndexSet::ElementIterator lri = slab->space.vorticity.fe_info->locally_owned_dofs->begin();
				dealii::IndexSet::ElementIterator lre = slab->space.vorticity.fe_info->locally_owned_dofs->end();
				for (; lri != lre ;lri++)
				{

					(*vort_owned)[
					  *lri
					  //time offset
					  + slab->space.vorticity.fe_info->dof->n_dofs() *
					  	  (cell_time->index() * slab->time.vorticity.fe_info->fe->dofs_per_cell)
					  // local in time dof
					  + slab->space.vorticity.fe_info->dof->n_dofs() * ii
					  ] = (*vortn_rel)[*lri];
				}
			}
		} //time dofs
	}

//	vort_owned->print(std::cout);
	if ( parameter_set->dwr.functional.mean_vorticity){
		*vort->x[0] = *vort_owned;
	}

//	std::cout << std::endl << std::endl;
//	vort->x[0]->print(std::cout);
	//////////////////////////////////////////////////////
	// output goal functionals

	DTM::pout
			<< "------------------"
			<< std::endl;

	// pressure
	if(parameter_set->dwr.functional.mean_pdiff){
		DTM::pout << "Pressure difference:" << std::endl;
		std::ofstream p_out;
		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			// append instead pressure difference to text file (ios_base::app):
			p_out.open("pressure.log", std::ios_base::app);
		}
		for (auto &item : pressure_values)
		{
			DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
			DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

			// save to txt file
			if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
				p_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
			}
		}

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			p_out.close();
		}
	}
	// drag

	if(parameter_set->dwr.functional.mean_drag){
		DTM::pout << "Face drag:" << std::endl;


		std::ofstream d_out;

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			d_out.open("drag.log", std::ios_base::app);
		}
		for (auto &item : drag_values)
		{
			DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
			DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

			if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			// save to txt file
				d_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
			}
		}

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			d_out.close();
		}
	}
	// lift

	if(parameter_set->dwr.functional.mean_lift){
		DTM::pout << "Face lift:" << std::endl;

		std::ofstream l_out;
		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			l_out.open("lift.log", std::ios_base::app);
		}

		for (auto &item : lift_values)
		{
			DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
			DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

			if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
				// save to txt file
				l_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
			}
		}

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			l_out.close();
		}
	}
	// vorticity

	if(parameter_set->dwr.functional.mean_vorticity){
		DTM::pout << "Vorticity:" << std::endl;

		std::ofstream l_out;
		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			l_out.open("vorticity.log", std::ios_base::app);
		}

		for (auto &item : vorticity_values)
		{
			DTM::pout << "	" << std::setw(14) << std::setprecision(8) << std::get<0>(item);
			DTM::pout << ":    " << std::setprecision(16) << std::get<1>(item) << std::endl;

			if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
				// save to txt file
				l_out << std::get<0>(item) << "," << std::get<1>(item) << std::endl;
			}
		}

		if ( dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0){
			l_out.close();
		}
	}
}

template<int dim>
double
Fluid<dim>::
compute_pressure(
	dealii::Point<dim> x,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un,
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// evaluate the fe system solution as x_h for a given point x
	dealii::Vector<double> x_h(dim + 1);
	try
	{
		dealii::VectorTools::point_value(
				*slab->space.primal.fe_info->dof,
				*un, // input dof vector at t_n
				x, // evaluation point
				x_h
		);
	}
	catch (typename dealii::VectorTools::ExcPointNotAvailableHere e)
	{}


	auto minmax = dealii::Utilities::MPI::min_max_avg(x_h[dim], mpi_comm);
	if ( std::abs(minmax.min) > minmax.max){
		return minmax.min;
	}
	else {
		return minmax.max;
	}
}

template<int dim>
void
Fluid<dim>::
compute_drag_lift_tensor(
		std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un,
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
		if (cell->is_locally_owned())
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

	double tmp = dealii::Utilities::MPI::sum(drag_lift_value[0],mpi_comm);
	drag_lift_value[0] = tmp;
	tmp = dealii::Utilities::MPI::sum(drag_lift_value[1],mpi_comm);
	drag_lift_value[1] = tmp;
	// 2D-1: 500; 2D-2 and 2D-3: 20 (see Schaefer/Turek 1996)
	double max_velocity = 0.;
	if (dim == 2){
		max_velocity =
		function.convection.dirichlet->value(
					dealii::Point<dim>(0.0, 0.205) // maximal velocity in the middle of the boundary
		)[0];
	}

	if (parameter_set->convection.dirichlet_boundary_function.compare("Convection_Parabolic_Inflow_3") == 0)
	{
		if (max_velocity < 0.3 + 1e-9)
				drag_lift_value *= 500.0; // 2D-1
		else
			drag_lift_value *= 20.0; // 2D-2
	}
	else if (parameter_set->convection.dirichlet_boundary_function.compare("Convection_Parabolic_Inflow_3_sin") == 0)
	{
		if ( dim == 2)
			drag_lift_value *= 20.0; // 2D-3
		else if (dim == 3)
			drag_lift_value *= 20.0/0.41; // 3D-3
	}
}
template<int dim>
double
Fluid<dim>::
compute_vorticity(
		std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > un,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > vortn
) {
	const dealii::QGaussLobatto<dim> quadrature_formula(
		slab->space.vorticity.fe_info->fe->tensor_degree()+1
	);

	dealii::FEValues<dim> fe_values_primal(*slab->space.primal.fe_info->fe,quadrature_formula,
									dealii::update_values | dealii::update_gradients | dealii::update_JxW_values
									| dealii::update_quadrature_points
	);

	dealii::FEValues<dim> fe_values_vort(*slab->space.vorticity.fe_info->fe,quadrature_formula, dealii::update_values |
											dealii::update_quadrature_points | dealii::update_JxW_values);
	dealii::IndexSet loe = vortn->locally_owned_elements();

	const unsigned int dofs_per_cell = slab->space.vorticity.fe_info->fe->dofs_per_cell;
	const unsigned int n_q_points = quadrature_formula.size();
	std::vector<unsigned int> local_dof_indices(dofs_per_cell);

	std::vector<std::vector<dealii::Tensor<1,dim>>> solution_grads(
		n_q_points, std::vector<dealii::Tensor<1,dim>>(dim+1)
	);

	double sqrd_mag = 0.;
	double curl_u;
	typename dealii::DoFHandler<dim>::active_cell_iterator
	  cell_primal = slab->space.primal.fe_info->dof->begin_active(),
	  endc = slab->space.primal.fe_info->dof->end(),
	  cell_vort   = slab->space.vorticity.fe_info->dof->begin_active();

	for ( ; cell_primal != endc; ++cell_primal, ++cell_vort)
		if (cell_primal->is_locally_owned())
		{
			fe_values_primal.reinit(cell_primal);
			fe_values_vort.reinit(cell_vort);
			fe_values_primal.get_function_gradients(*un,solution_grads);
			cell_vort->get_dof_indices(local_dof_indices);
			for (unsigned int k = 0 ; k < dofs_per_cell ; k++){
				unsigned int dof_ind = local_dof_indices[k];
					for ( unsigned int q = 0 ; q < n_q_points; q++){
						curl_u = (solution_grads[q][1][0]-solution_grads[q][0][1])
								*fe_values_vort.shape_value(k,q);
						(*vortn)[dof_ind] += curl_u;
						sqrd_mag+= curl_u*curl_u*fe_values_vort.JxW(q);
					}
			}
		}

	return sqrd_mag;
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
		std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();

	error_estimator.storage.eta_space->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);

	error_estimator.storage.eta_time =
		std::make_shared< DTM::types::storage_data_trilinos_vectors<1> > ();

	error_estimator.storage.eta_time->resize(
		static_cast<unsigned int>(grid->slabs.size())
	);
}

template<int dim>
void
Fluid<dim>::
eta_reinit_storage_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_s,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_t
) {
	// spatial error indicators
	for (unsigned int j{0}; j < eta_s->x.size(); ++j) {
		eta_s->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

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
				*slab->spacetime.pu.locally_owned_dofs,
				mpi_comm
		);
	}

	// temporal error indicators
	for (unsigned int j{0}; j < eta_t->x.size(); ++j) {
		eta_t->x[j] = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

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
				*slab->spacetime.pu.locally_owned_dofs,
				mpi_comm
		);
	}
}

template<int dim>
void
Fluid<dim>::
compute_effectivity_index() {
	// sum up error estimator
	double value_eta_k = 0.;
	for (auto &element : *error_estimator.storage.eta_time)
		value_eta_k += element.x[0]->mean_value()*element.x[0]->size();

	double value_eta_h = 0.;
	for (auto &element : *error_estimator.storage.eta_space)
		value_eta_h += element.x[0]->mean_value()*element.x[0]->size();

	const double value_eta = std::abs(value_eta_k + value_eta_h);

	// true error of FEM simulation in goal functional
	double reference_goal_functional=0.;
	double fem_goal_functional=0.;

	if ( !parameter_set->dwr.goal.type.compare("mean_drag")){
		reference_goal_functional= error_estimator.goal_functional.reference.mean_drag;
		fem_goal_functional= error_estimator.goal_functional.fem.mean_drag;
	} else if ( !parameter_set->dwr.goal.type.compare("mean_lift")){
		reference_goal_functional= error_estimator.goal_functional.reference.mean_lift;
		fem_goal_functional= error_estimator.goal_functional.fem.mean_lift;
	} else if ( !parameter_set->dwr.goal.type.compare("mean_pdiff")){
		reference_goal_functional= error_estimator.goal_functional.reference.mean_pdiff;
		fem_goal_functional= error_estimator.goal_functional.fem.mean_pdiff;
	} else if ( !parameter_set->dwr.goal.type.compare("mean_vorticity")){
		reference_goal_functional= error_estimator.goal_functional.reference.mean_vorticity;
		fem_goal_functional= error_estimator.goal_functional.fem.mean_vorticity;
	}
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

	const unsigned int N_slabs{static_cast<unsigned int>(grid->slabs.size())};
	std::vector<unsigned int> ints_per_slab(N_slabs);
	//determine number of temporal intervals
	unsigned int N_int =0 ;
	{
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (unsigned int n = 0; slab != ends; ++slab, ++n) {
            ints_per_slab[n] =slab->time.tria->n_active_cells();
			N_int+= ints_per_slab[n];
		}
	}

	std::vector<double> eta_k(N_int);

	double eta_k_global {0.};
	double eta_h_global {0.};

	// compute eta_k^n on I_n for n=1..N as well as global estimators
	{
		auto eta_it_k{error_estimator.storage.eta_time->begin()};
		auto eta_it_h{error_estimator.storage.eta_space->begin()};

		auto slab{grid->slabs.begin()};

		auto eta_k_local = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
		auto eta_h_local = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
		unsigned int n_in{0};
		for (unsigned n_sl{0}; n_sl < N_slabs;
				++n_sl, ++eta_it_k, ++eta_it_h, ++slab) {
			Assert(
				(eta_it_k != error_estimator.storage.eta_time->end()),
				dealii::ExcInternalError()
			);
			Assert(
				(eta_it_h != error_estimator.storage.eta_space->end()),
				dealii::ExcInternalError()
			);
			eta_k_local->reinit(*slab->space.pu.fe_info->locally_owned_dofs,
					          mpi_comm);

			eta_h_local->reinit(*slab->space.pu.fe_info->locally_owned_dofs,
					          mpi_comm);
			//get local pu's
			//TODO: if using something else than dG(0) in time for pu this should be changed!

			for (auto &cell : slab->time.tria->active_cell_iterators() ){
				*eta_k_local = 0.;
				*eta_h_local = 0.;
				dealii::IndexSet::ElementIterator loi = slab->space.pu.fe_info->locally_owned_dofs->begin();
				dealii::IndexSet::ElementIterator loe = slab->space.pu.fe_info->locally_owned_dofs->end();
				for (; loi != loe ; loi++){
					(*eta_k_local)[*loi] = (*eta_it_k->x[0])[
						*loi
						//time offset
						+slab->space.pu.fe_info->dof->n_dofs() *
						cell->index()
					 ];

					(*eta_h_local)[*loi] = (*eta_it_h->x[0])[
						*loi
						//time offset
						+slab->space.pu.fe_info->dof->n_dofs() *
						cell->index()
					 ];
				}
				double eta_k_K = eta_k_local->mean_value()*eta_k_local->size();
				double eta_h_K = eta_h_local->mean_value()*eta_h_local->size();
				eta_k[n_in] = std::abs(eta_k_K);
				eta_k_global += eta_k_K;

				double k_n = cell->bounding_box().side_length(0);
				eta_h_global +=
						eta_h_K*
						parameter_set->time.fluid.T/
						(N_int*k_n);

				n_in++;
			}
		}
	}

	// Per definition eta_k[0] is 0 for the primal problem, so just set it to the next time step
	eta_k[0] = eta_k[1];
	if (eta_k[0] == 0.)
		DTM::pout << "eta_k[0] = " << eta_k[0] << std::endl;

	///////////////////////////////
	// output eta_k for each slab

	if (dealii::Utilities::MPI::this_mpi_process(mpi_comm)==0){
		std::ostringstream filename;
		filename
			<< "slabwise_eta_k-dwr_loop-"
			<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop << ".log";

		std::ofstream eta_k_out;
		eta_k_out.open(filename.str());

		unsigned int _i = 0;
		unsigned int n_sl = 0;
		unsigned int n_in = 0;
		double eta_k_slab = 0.;
		for (auto &slab : grid->slabs)
		{
			eta_k_slab = 0.;
			for (unsigned int ii = 0 ; ii < ints_per_slab[n_sl] ; ii++){
				eta_k_slab += eta_k[n_in];
				n_in++;
			}

			// save to txt file
			eta_k_out << slab.t_m << "," << eta_k_slab << std::endl;
			eta_k_out << slab.t_n << "," << eta_k_slab<< std::endl;
			_i++;
			n_sl++;
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
	}

	///////////////////////////////
	// output eta_k for each temporal interval

	if (dealii::Utilities::MPI::this_mpi_process(mpi_comm)==0){
		std::ostringstream filename;
		filename
			<< "intervalwise_eta_k-dwr_loop-"
			<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop << ".log";

		std::ofstream eta_k_out;
		eta_k_out.open(filename.str());

		// reset filename
		filename.str("");
		filename.clear();

		filename
				<< "intervalwise_k_n-dwr_loop-"
				<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop << ".log";

		std::ofstream k_n_out;
		k_n_out.open(filename.str());

		unsigned int n_in = 0;
		for (auto &slab : grid->slabs)
		{
			for (auto &cell : slab.time.tria->active_cell_iterators()){
				const dealii::BoundingBox<1> bbox = cell->bounding_box ();
				eta_k_out << bbox.get_boundary_points().first << ","
						  << eta_k[n_in]
					      << std::endl;

				eta_k_out << bbox.get_boundary_points().second << ","
						  << eta_k[n_in]
					      << std::endl;


				k_n_out << bbox.get_boundary_points().first << ","
						<< bbox.side_length(0)
						<< std::endl;

				k_n_out << bbox.get_boundary_points().second << ","
						<< bbox.side_length(0)
						<< std::endl;
				n_in++;
			}

		}
		eta_k_out.close();
		k_n_out.close();
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
				for ( unsigned int n{0} ; n < N_int ; n++ )
				{
					D_sum += eta_sorted[n];
					if ( D_sum >= D_goal ){
						threshold = eta_sorted[n];
						n = N_int;
					}
				}

			} else if (parameter_set->dwr.refine_and_coarsen.time.strategy.compare("fixed_number") == 0) {
				// check if index for eta_criterium_for_mark_time_refinement is valid
				Assert(static_cast<int>(std::ceil(static_cast<double>(N_int)
						* parameter_set->dwr.refine_and_coarsen.time.top_fraction)) >= 0,
					dealii::ExcInternalError()
				);

				unsigned int index_for_mark_time_refinement {
					static_cast<unsigned int> (
						static_cast<int>(std::ceil(
							static_cast<double>(N_int)
							* parameter_set->dwr.refine_and_coarsen.time.top_fraction
						))
					)
				};

				threshold = eta_sorted[ index_for_mark_time_refinement < N_int ?
											index_for_mark_time_refinement : N_int-1 ];

			}

			auto slab{grid->slabs.begin()};
			auto ends{grid->slabs.end()};
			unsigned int n_in {0};
			for (unsigned int n_sl{0} ; slab != ends; ++slab, ++n_sl) {
				Assert((n_sl < N_slabs), dealii::ExcInternalError());

				for ( auto &cell : slab->time.tria->active_cell_iterators()){
					if (eta_k[n_in] >= threshold){
						cell->set_refine_flag();
						DTM::pout << "Marked interval " << n_in
								  << " for temporal refinement"
								  << std::endl;

						slab->set_refine_in_time_flag();
					}
					n_in++;
				}

			}
		}
	}

	///////////////////////////////
	// spatial refinement
	if (std::abs(eta_k_global) <= equilibration_factor*std::abs(eta_h_global))
	{
//		for (auto &eta_In : *error_estimator.storage.eta_space) {
//			for (auto &eta_K : *eta_In.x[0] ) {
//				eta_K = std::abs(eta_K);
//				Assert(eta_K >= 0., dealii::ExcInternalError());
//			}
//		}
		unsigned int K_max{0};
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		auto eta_it{error_estimator.storage.eta_space->begin()};

		auto eta_local = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
		auto eta_relevant = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();

		for (unsigned int n{0} ; slab != ends; ++slab, ++eta_it, ++n) {

			Assert(
				(eta_it != error_estimator.storage.eta_space->end()),
				dealii::ExcInternalError()
			);

			DTM::pout << "\tslab = " << n << std::endl;

			const auto n_active_cells_on_slab{slab->space.tria->n_active_cells()};
			DTM::pout << "\t#K = " << n_active_cells_on_slab << std::endl;
			K_max = (K_max > n_active_cells_on_slab) ? K_max : n_active_cells_on_slab;

			if ( parameter_set->dwr.refine_and_coarsen.space.top_fraction1 == 1.0 )
			{
				slab->space.tria->refine_global(1);
			}
			else {
				eta_local->reinit(*slab->space.pu.fe_info->locally_owned_dofs,
								  mpi_comm);

				eta_relevant->reinit(*slab->space.pu.fe_info->locally_owned_dofs,
								  *slab->space.pu.fe_info->locally_relevant_dofs,
								  mpi_comm);

				*eta_local = 0.;

				//Go over each locally owned spatial DoF
				dealii::IndexSet::ElementIterator loi = slab->space.pu.fe_info->locally_owned_dofs->begin();
				dealii::IndexSet::ElementIterator loe = slab->space.pu.fe_info->locally_owned_dofs->end();
				for (; loi != loe ; loi++){
					//Sum up over all temporal intervals
					for (auto &cell : slab->time.tria->active_cell_iterators()){
						(*eta_local)[*loi] += (*eta_it->x[0])[
												*loi
												//time offset
												+slab->space.pu.fe_info->dof->n_dofs() *
												cell->index()
											 ];
					}
				}

				//For cell indicators we need communication over ghost cells
				*eta_relevant = *eta_local;



				const unsigned int dofs_per_cell_pu = slab->space.pu.fe_info->fe->dofs_per_cell;
				std::vector< unsigned int > local_dof_indices(dofs_per_cell_pu);
				unsigned int max_n = n_active_cells_on_slab *
						parameter_set->dwr.refine_and_coarsen.space.max_growth_factor_n_active_cells;


				typename dealii::DoFHandler<dim>::active_cell_iterator
				cell{slab->space.pu.fe_info->dof->begin_active()},
				endc{slab->space.pu.fe_info->dof->end()};

				dealii::Vector<double> indicators(n_active_cells_on_slab);
				indicators = 0.;
				//Go over each locally owned spatial cell
				for ( unsigned int cell_no{0}; cell!= endc ; ++cell, ++cell_no){
					if ( cell->is_locally_owned()){
						cell->get_dof_indices(local_dof_indices);

						for (unsigned int i = 0 ; i < dofs_per_cell_pu ; i++){
							indicators[cell_no] += (*eta_relevant)(local_dof_indices[i])/dofs_per_cell_pu;
						}
					}
					indicators[cell_no ] = std::abs(indicators[cell_no]);
				}


				if (parameter_set->dwr.refine_and_coarsen.space.strategy.compare("RichterWick") == 0){
					double threshold = eta_local->mean_value()*
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

//				// count which percentage of spatial cells have been marked for refinement
//				unsigned int marked_cells = 0;
//
//				for (const auto &cell : slab->space.tria->active_cell_iterators())
//					if (cell->refine_flag_set())
//						marked_cells++;
//
//				DTM::pout << "\tSpace top fraction = " << std::setprecision(5) << ((double)marked_cells) / slab->space.tria->n_active_cells() << std::endl;
//
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
			slab->time.tria->execute_coarsening_and_refinement();

		}
	}
	//TODO: go over slabs and check if they should be split into two
	{
		auto slab{grid->slabs.begin()};
		auto ends{grid->slabs.end()};
		for (; slab != ends; ++slab) {
			if (slab->refine_in_time){
				slab->clear_refine_in_time_flag();
				grid->split_slab_in_time(slab);
			}
		}
	}
	DTM::pout << "refined in time" << std::endl;
}
//
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
eta_do_data_output_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta,
	const unsigned int dwr_loop,
	std::string eta_type) {
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
    Assert(eta->x[0].use_count(), dealii::ExcNotInitialized());
	// TODO: might need to be debugged; adapted from primal_do_data_output_on_slab() & copied form dual_do_data_output_on_slab()
//
//	// triggered output mode
//	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
//	Assert(
//		slab->space.pu.fe_info->partitioning_locally_owned_dofs.use_count(),
//		dealii::ExcNotInitialized()
//	);
//	error_estimator.data_output_space->set_DoF_data(
//		slab->space.pu.fe_info->dof,
//		slab->space.pu.fe_info->partitioning_locally_owned_dofs
//	);
//
// 	auto eta_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
// 	eta_trigger->reinit(
// 		slab->space.pu.fe_info->dof->n_dofs()
// 	);
//
	std::ostringstream filename;
	filename
		<< eta_type << "-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
//
//	{
//		// fe face values time: time face (I_n) information
//		dealii::FEValues<1> fe_face_values_time(
//			*slab->time.pu.fe_info->mapping,
//			*slab->time.pu.fe_info->fe,
//			dealii::QGaussLobatto<1>(2),
//			dealii::update_quadrature_points
//		);
//
//		Assert(
//			slab->time.pu.fe_info->dof.use_count(),
//			dealii::ExcNotInitialized()
//		);
//
//		auto cell_time = slab->time.pu.fe_info->dof->begin_active();
//		auto endc_time = slab->time.pu.fe_info->dof->end();
//
//		for ( ; cell_time != endc_time; ++cell_time) {
//			fe_face_values_time.reinit(cell_time);
//
//			////////////////////////////////////////////////////////////////////
//			// construct quadrature for data output,
//			// if triggered output time values are inside this time element
//			//
//
//			auto t_m = fe_face_values_time.quadrature_point(0)[0];
//			auto t_n = fe_face_values_time.quadrature_point(1)[0];
//			auto tau = t_n-t_m;
//
//			std::list<double> output_times;
//
//			if (error_estimator.data_output_time_value < t_m) {
//				error_estimator.data_output_time_value = t_m;
//			}
//
//			for ( ; (error_estimator.data_output_time_value <= t_n) ||
//				(
//					(error_estimator.data_output_time_value > t_n) &&
//					(std::abs(error_estimator.data_output_time_value - t_n) < tau*1e-12)
//				); ) {
//				output_times.push_back(error_estimator.data_output_time_value);
//				error_estimator.data_output_time_value += error_estimator.data_output_trigger;
//			}
//
//			if (output_times.size() && output_times.back() > t_n) {
//				output_times.back() = t_n;
//			}
//
//			if ((output_times.size() > 1) &&
//				(output_times.back() == *std::next(output_times.rbegin()))) {
//				// remove the last entry, iff doubled
//				output_times.pop_back();
//			}
//
//			// convert container
//			if (!output_times.size()) {
//				continue;
//			}
//
//			std::vector< dealii::Point<1> > output_time_points(output_times.size());
//			{
//				auto time{output_times.begin()};
//				for (unsigned int q{0}; q < output_time_points.size(); ++q,++time) {
//					double t_trigger{*time};
//					output_time_points[q][0] = (t_trigger-t_m)/tau;
//				}
//
//				if (output_time_points[0][0] < 0) {
//					output_time_points[0][0] = 0;
//				}
//
//				if (output_time_points[output_time_points.size()-1][0] > 1) {
//					output_time_points[output_time_points.size()-1][0] = 1;
//				}
//			}
//
//			dealii::Quadrature<1> quad_time(output_time_points);
//
//			// create fe values
//			dealii::FEValues<1> fe_values_time(
//				*slab->time.pu.fe_info->mapping,
//				*slab->time.pu.fe_info->fe,
//				quad_time,
//				dealii::update_values |
//				dealii::update_quadrature_points
//			);
//
//			fe_values_time.reinit(cell_time);
//
//			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.pu.fe_info->fe->dofs_per_cell);
//			cell_time->get_dof_indices(local_dof_indices);
//
//			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
// 				*eta_trigger = 0.;
//
// 				// evaluate solution for t_q
// 				for (
// 					unsigned int jj{0};
// 					jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {
// 				for (
// 					dealii::types::global_dof_index i{0};
// 					i < slab->space.pu.fe_info->dof->n_dofs(); ++i) {
// 					(*eta_trigger)[i] += (*eta->x[0])[
// 						i
// 						// time offset
// 						+ slab->space.pu.fe_info->dof->n_dofs() *
// 							local_dof_indices[jj]
// 					] * fe_values_time.shape_value(jj,qt);
// 				}}
//
////				std::cout
////					<< "error indicator output generated for t = "
////					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
////					<< std::endl;
//
//				error_estimator.data_output_space->write_data(
//					filename.str(),
//					eta_trigger,
//					// error_estimator.data_postprocessor,
//					fe_values_time.quadrature_point(qt)[0] // t_trigger
//				);
//			}
//		}
//	}
}

template<int dim>
void
Fluid<dim>::
eta_do_data_output_on_slab_Qn_mode(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_space,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_time,
	const unsigned int dwr_loop ){
	// natural output of solutions on Q_n in their support points in time
	Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());

	error_estimator.data_output_space->set_DoF_data(
		slab->space.pu.fe_info->dof
	);

	error_estimator.data_output_time->set_DoF_data(
		slab->space.pu.fe_info->dof
	);

	auto eta_trigger = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	eta_trigger->reinit(
		*slab->space.pu.fe_info->locally_owned_dofs,
		*slab->space.pu.fe_info->locally_relevant_dofs,
		mpi_comm
	);

	auto
	owned_tmp_space = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	owned_tmp_space->reinit(
		*slab->space.pu.fe_info->locally_owned_dofs,
		mpi_comm
	);

	auto
	owned_tmp_time = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	owned_tmp_time->reinit(
		*slab->space.pu.fe_info->locally_owned_dofs,
		mpi_comm
	);

	std::ostringstream filename_space;
	filename_space
		<< "error-indicators-space-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	std::ostringstream filename_time;
	filename_time
		<< "error-indicators-time-dwr_loop-"
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
 				*owned_tmp_space = 0.;
 				*owned_tmp_time = 0.;

 				dealii::IndexSet::ElementIterator lri = slab->space.pu.fe_info->locally_owned_dofs->begin();
 				dealii::IndexSet::ElementIterator lre = slab->space.pu.fe_info->locally_owned_dofs->end();


 				// evaluate solution for t_q
 				for (; lri!= lre; lri++) {
 					for (
 							unsigned int jj{0};
 							jj < slab->time.pu.fe_info->fe->dofs_per_cell; ++jj) {

						(*owned_tmp_space)[*lri] += (*eta_space->x[0])[
							*lri
							// time offset
							+ slab->space.pu.fe_info->dof->n_dofs() *
								local_dof_indices_time[jj]
						] * fe_values_time.shape_value(jj,qt);

						(*owned_tmp_time)[*lri] += (*eta_time->x[0])[
							*lri
							// time offset
							+ slab->space.pu.fe_info->dof->n_dofs() *
								local_dof_indices_time[jj]
						] * fe_values_time.shape_value(jj,qt);


					}
 				}

 				*eta_trigger = *owned_tmp_space;
// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0]
// 					<< std::endl;

				error_estimator.data_output_space->write_data(
					filename_space.str(),
					eta_trigger,
					// error_estimator.data_postprocessor,
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);

				*eta_trigger = *owned_tmp_time;

				error_estimator.data_output_time->write_data(
					filename_time.str(),
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
eta_do_data_output(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_space,
		const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_time,
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
		eta_do_data_output_on_slab_Qn_mode(slab, eta_space,eta_time, dwr_loop);
	}
	else {
		// fixed trigger output mode
		eta_do_data_output_on_slab(slab, eta_space, dwr_loop, "error-indicators-space");
		eta_do_data_output_on_slab(slab, eta_time,  dwr_loop, "error-indicators-time");
	}
}


template<int dim>
void
Fluid<dim>::
eta_sort_xdmf_by_time(
	const unsigned int dwr_loop
) {
	if (dealii::Utilities::MPI::this_mpi_process(mpi_comm)==0) {
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
}

} // namespace

#include "Fluid.inst.in"
