/**
 * @file Fluid.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
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

// DEAL.II includes
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

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
	
	////////////////////////////////////////////////////////////////////////////
	// adaptivity loop
	//
	
	DTM::pout
		<< std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl;
	
	unsigned int dwr_loop{1};
	unsigned int max_dwr_loop{1};
	do {
//		if (dwr_loop > 1) {
//			// error estimation
//			eta_reinit_storage();
//			compute_error_indicators();
//			
//			// do space-time mesh refinements and coarsenings
//			refine_and_coarsen_space_time_grid();
//			
//			grid->set_manifolds();
//		}
		
		DTM::pout
			<< "***************************************************************"
			<< "*****************" << std::endl
			<< "adaptivity loop = " << dwr_loop << std::endl;
		
		grid->set_boundary_indicators();
		grid->distribute();
		
		// primal problem:
		primal_reinit_storage();
		primal_init_data_output();
		primal_do_forward_TMS();
		primal_do_data_output(dwr_loop,false);
		
		// check if dwr has converged
		// TODO: converge criterion

		// dual problem
		dual_reinit_storage();
		dual_init_data_output();
		dual_do_backward_TMS();
		dual_do_data_output(dwr_loop,false);

		// error estimation
		// TODO: compute error indicators
		// TODO: compute effectivity indices

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
	
	{
		auto slab = grid->slabs.begin();
		for (auto &element : *primal.storage.u) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				element.x[j] = std::make_shared< dealii::Vector<double> > ();
				
				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());
				
				Assert(
					slab->space.primal.dof.use_count(),
					dealii::ExcNotInitialized()
				);
				
				Assert(
					slab->time.primal.dof.use_count(),
					dealii::ExcNotInitialized()
				);
				
				element.x[j]->reinit(
					slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
				);
			}
			++slab;
		}
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
		
		Assert(primal.L.use_count(), dealii::ExcNotInitialized());
		assembler.assemble(
			primal.L,
			slab,
			u,
			( parameter_set->problem.compare("Navier-Stokes")==0 )
		);
		
	}
}


template<int dim>
void
Fluid<dim>::
primal_assemble_const_rhs(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	// ASSEMBLY SPACE-TIME OPERATOR: InitialValue VECTOR ///////////////////////
	primal.Mum = std::make_shared< dealii::Vector<double> > ();
	
	Assert(
		slab->space.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	
	primal.Mum->reinit(
		slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
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
	
//	// ASSEMBLY SPACE-TIME OPERATOR: FORCE VECTOR //////////////////////////////
//	primal.f = std::make_shared< dealii::Vector<double> > ();
//
//	Assert(
//		slab->space.primal.dof.use_count(),
//		dealii::ExcNotInitialized()
//	);
//	Assert(
//		slab->time.primal.dof.use_count(),
//		dealii::ExcNotInitialized()
//	);
//
//	primal.f->reinit(
//		slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
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
		slab->space.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);

	primal.Fu->reinit(
		slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
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

		Assert(primal.Fu.use_count(), dealii::ExcNotInitialized());
		assembler.assemble(
			primal.Fu,
			slab,
			u,
			( parameter_set->problem.compare("Navier-Stokes")==0 )
		);
	}

	*primal.b = *primal.Mum;
	primal.b->add(-1., *primal.Fu);
	primal_apply_bc(boundary_values,primal.b);
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
		=std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
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
			const dealii::QGauss<1> support_points(
				slab->time.primal.fe->tensor_degree()+1
			);

			auto cell_time = slab->time.primal.dof->begin_active();
			auto endc_time = slab->time.primal.dof->end();

			dealii::FEValues<1> time_fe_values(
				*slab->time.primal.mapping,
				*slab->time.primal.fe,
				support_points,
				dealii::update_quadrature_points
			);

			for ( ; cell_time != endc_time; ++cell_time) {
				time_fe_values.reinit(cell_time);

				for (unsigned int qt{0}; qt < support_points.size(); ++qt) {
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
							*slab->space.primal.dof,
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
							*slab->space.primal.dof,
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
							+ slab->space.primal.dof->n_dofs() *
								(cell_time->index()
								* slab->time.primal.fe->dofs_per_cell)
							// local in time dof
							+ slab->space.primal.dof->n_dofs() * qt
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
			const dealii::QGauss<1> support_points(
				slab->time.primal.fe->tensor_degree()+1
			);

			auto cell_time = slab->time.primal.dof->begin_active();
			auto endc_time = slab->time.primal.dof->end();

			dealii::FEValues<1> time_fe_values(
				*slab->time.primal.mapping,
				*slab->time.primal.fe,
				support_points,
				dealii::update_quadrature_points
			);

			for ( ; cell_time != endc_time; ++cell_time) {
				time_fe_values.reinit(cell_time);

				for (unsigned int qt{0}; qt < support_points.size(); ++qt) {
					std::map<dealii::types::global_dof_index,double>
						boundary_values_qt;

					dealii::VectorTools::interpolate_boundary_values (
						*slab->space.primal.dof,
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
							+ slab->space.primal.dof->n_dofs() *
								(cell_time->index()
								* slab->time.primal.fe->dofs_per_cell)
							// local in time dof
							+ slab->space.primal.dof->n_dofs() * qt
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
	if ( boundary_values.size()) {
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
	if ( boundary_values.size()) {
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
		double diagonal_scaling_value{0.};

		for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
			if (std::abs(A->el(i,i)) > std::abs(diagonal_scaling_value)) {
				diagonal_scaling_value = A->el(i,i);
			}
		}

		if (diagonal_scaling_value == double(0.)) {
			diagonal_scaling_value = double(1.);
		}

		Assert(
			(diagonal_scaling_value != double(0.)),
			dealii::ExcInternalError()
		);

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
	const typename DTM::types::storage_data_vectors<1>::iterator &u
) {
	
	primal.b = std::make_shared< dealii::Vector<double> > ();
	primal.du = std::make_shared< dealii::Vector<double> >();
	Assert(
		slab->space.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.primal.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	
	primal.b->reinit(
		slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
	);

	primal.du->reinit(
		slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
	);


	
	////////////////////////////////////////////////////////////////////////////
	// apply inhomogeneous Dirichlet boundary values
	//
	//TODO: copy previous solution as initial solution
	
	DTM::pout << "dwr-instatfluid: compute boundary values..." ;


	std::map<dealii::types::global_dof_index, double> initial_bc;
	primal_calculate_boundary_values(slab,initial_bc);
    std::map<dealii::types::global_dof_index, double> zero_bc;
    primal_calculate_boundary_values(slab,zero_bc,true);

    DTM::pout << " (done)" << std::endl;

    DTM::pout << "dwr-instatfluid: apply previous solution as initial Newton guess..." ;

    for ( unsigned int i{0} ; i < slab->space.primal.dof->n_dofs() ; i++) {
    	for ( unsigned int ii{0} ; ii < slab->time.primal.dof->n_dofs() ; ii++) {
    		(*u->x[0])[i+
					   slab->space.primal.dof->n_dofs()*ii
					   ] = (*primal.um)[i];
    	}
    }

    DTM::pout << " (done)" << std::endl;

    primal_apply_bc(initial_bc,u->x[0]);
	slab->spacetime.primal.constraints->distribute(
		*u->x[0]
	);

	// assemble slab problem const rhs
	primal_assemble_const_rhs(slab);


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

    while ( newton_residual > newton.lower_bound && newton_step < newton.max_steps)
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
			primal_assemble_system(slab, u->x[0]);
			primal_apply_bc(zero_bc,primal.L,primal.du,primal.b);
			////////////////////////////////////////////////////////////////////////////
			// condense hanging nodes in system matrix, if any
			//
			slab->spacetime.primal.constraints->condense(*primal.L);
		}

		////////////////////////////////////////////////////////////////////////////
		// solve linear system with direct solver
		//

		dealii::SparseDirectUMFPACK iA;
		iA.initialize(*primal.L);
		iA.vmult(*primal.du, *primal.b);

		slab->spacetime.primal.constraints->distribute(
			*primal.du
		);

		for ( line_search_step = 0 ; line_search_step < newton.line_search_steps ; line_search_step++) {
			u->x[0]->add(1.0,*primal.du);

			primal_assemble_and_construct_Newton_rhs(slab, zero_bc, u->x[0]);
			new_newton_residual = primal.b->linfty_norm();
			if ( new_newton_residual < newton_residual)
				break;
			else
				u->x[0]->add(-1.0,*primal.du);
			*primal.du*= newton.line_search_damping;
		}
		DTM::pout << std::setprecision(5) << newton_step << "\t"
				  << std::scientific << newton_residual << "\t"
				  << std::scientific << newton_residual/old_newton_residual << "\t";

		if ( newton_residual/old_newton_residual > newton.rebuild)
			DTM::pout << "r\t";
		else
			DTM::pout << " \t";

		DTM::pout << line_search_step << "\t" << std::scientific << std::endl;
		newton_step++;
    }

}


template<int dim>
void
Fluid<dim>::
primal_do_forward_TMS() {
	////////////////////////////////////////////////////////////////////////////
	// prepare time marching scheme (TMS) loop
	//
	
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
		
		if (slab == grid->slabs.begin()) {
			////////////////////////////////////////////////////////////////////////////
			// interpolate (or project) initial value(s)
			//
			std::shared_ptr< dealii::VectorFunctionFromTensorFunction<dim> > dirichlet_function
					=std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
						*function.convection.dirichlet,
						0, (dim+1)
					);

			dirichlet_function->set_time(0);
			function.convection.dirichlet->set_time(0);

			primal.um = std::make_shared< dealii::Vector<double> > ();
			primal.um->reinit(slab->space.primal.dof->n_dofs());
			*primal.um = 0.;

//			Assert(function.u_0.use_count(), dealii::ExcNotInitialized());
//			function.u_0->set_time(slab->t_m);

			Assert((slab != grid->slabs.end()), dealii::ExcInternalError());
			Assert(slab->space.primal.mapping.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(primal.um.use_count(), dealii::ExcNotInitialized());
			
			dealii::VectorTools::interpolate(
				*slab->space.primal.mapping,
				*slab->space.primal.dof,
				*dirichlet_function, //*function.u_0,
				*primal.um
			);

			// NOTE: after the first dwr-loop the initial triangulation could have
			//       hanging nodes. Therefore,
			// distribute hanging node constraints to make the result continuous again:
			slab->space.primal.constraints->distribute(
				*primal.um
			);
		}
		else {
			// not the first slab: transfer un solution to um solution
			Assert(primal.un.use_count(), dealii::ExcNotInitialized());

			primal.um = std::make_shared< dealii::Vector<double> > ();
			Assert(
				slab->space.primal.block_sizes.use_count(),
				dealii::ExcNotInitialized()
			);
			primal.um->reinit(slab->space.primal.dof->n_dofs());
			*primal.um = 0.;

			// for n > 1 interpolate between two (different) spatial meshes
			// the solution u(t_n)|_{I_{n-1}}  to  u(t_m)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n-1}:
				*std::prev(slab)->space.primal.dof,
				*primal.un,
				// solution on I_n:
				*slab->space.primal.dof,
				*slab->space.primal.constraints,
				*primal.um
			);

			slab->space.primal.constraints->distribute(
				*primal.um
			);
		}
		
		// solve slab problem (i.e. apply boundary values and solve for u0)
		primal_solve_slab_problem(slab,u);
		
		////////////////////////////////////////////////////////////////////////
		// do postprocessings on the solution
		//
		
		// evaluate solution u(t_n)
		primal.un = std::make_shared< dealii::Vector<double> > ();
		Assert(
			slab->space.primal.block_sizes.use_count(),
			dealii::ExcNotInitialized()
		);
			primal.un->reinit(slab->space.primal.dof->n_dofs());
		*primal.un = 0.;

		{
			dealii::FEValues<1> fe_face_values_time(
				*slab->time.primal.mapping,
				*slab->time.primal.fe,
				dealii::QGaussLobatto<1>(2),
				dealii::update_values
			);

			auto cell_time = slab->time.primal.dof->begin_active();
			auto last_cell_time = cell_time;
			auto endc_time = slab->time.primal.dof->end();

			for ( ; cell_time != endc_time; ++cell_time) {
				last_cell_time=cell_time;
			}

			cell_time=last_cell_time;
			{
				Assert((cell_time!=endc_time), dealii::ExcInternalError());
				fe_face_values_time.reinit(cell_time);

				// evaluate solution for t_n of Q_n
				for (unsigned int jj{0};
					jj < slab->time.primal.fe->dofs_per_cell; ++jj)
				for (dealii::types::global_dof_index i{0};
					i < slab->space.primal.dof->n_dofs(); ++i) {
					(*primal.un)[i] += (*u->x[0])[
						i
						// time offset
						+ slab->space.primal.dof->n_dofs() *
							(cell_time->index() * slab->time.primal.fe->dofs_per_cell)
						// local in time dof
						+ slab->space.primal.dof->n_dofs() * jj
					] * fe_face_values_time.shape_value(jj,1);
				}
			}
		}

		////////////////////////////////////////////////////////////////////////
		// compute functional values:
		//
		compute_functional_values(primal.un, slab);

		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//
		
		++n;
		++slab;
		++u;
		
		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//
		
		primal.L = nullptr;
		primal.b = nullptr;
// 		primal.f = nullptr;
		primal.Mum = nullptr;
		
		DTM::pout << std::endl;
	}
	
	DTM::pout
		<< "primal: forward TMS problem done" << std::endl
		<< "*******************************************************************"
		<< "*************" << std::endl
		<< std::endl;
	
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
	Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.primal.partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	primal.data_output->set_DoF_data(
		slab->space.primal.dof,
		slab->space.primal.partitioning_locally_owned_dofs
	);
	
 	auto u_trigger = std::make_shared< dealii::BlockVector<double> > ();
 	Assert(
 		slab->space.primal.block_sizes.use_count(),
 		dealii::ExcNotInitialized()
 	);
 	u_trigger->reinit(
 		*slab->space.primal.block_sizes
 	);
	
	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
	
	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.primal.mapping,
			*slab->time.primal.fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);
		
		Assert(
			slab->time.primal.dof.use_count(),
			dealii::ExcNotInitialized()
		);
		
		auto cell_time = slab->time.primal.dof->begin_active();
		auto endc_time = slab->time.primal.dof->end();
		
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
				*slab->time.primal.mapping,
				*slab->time.primal.fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);
			
			fe_values_time.reinit(cell_time);
			
			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.primal.fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);
			
			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*u_trigger = 0.;
 				
 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.primal.fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.primal.dof->n_dofs(); ++i) {
 					(*u_trigger)[i] += (*u->x[0])[
 						i
 						// time offset
 						+ slab->space.primal.dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}
				
				std::cout
					<< "output generated for t = "
					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
					<< std::endl;
				
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
	Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.primal.partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	primal.data_output->set_DoF_data(
		slab->space.primal.dof,
		slab->space.primal.partitioning_locally_owned_dofs
	);
	
	auto u_trigger = std::make_shared< dealii::BlockVector<double> > ();
 	Assert(
 		slab->space.primal.block_sizes.use_count(),
 		dealii::ExcNotInitialized()
 	);
 	u_trigger->reinit(
 		*slab->space.primal.block_sizes
 	);
 	
	std::ostringstream filename;
	filename
		<< "solution-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;
	
	{
		// TODO: check natural time quadrature (Gauss, Gauss-Radau, etc.)
		//       from input or generated fe<1>
		//
		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.primal.mapping,
			*slab->time.primal.fe,
			dealii::QGauss<1>(slab->time.primal.fe->tensor_degree()+1), // here
			dealii::update_values |
			dealii::update_quadrature_points
		);
		
		auto cell_time = slab->time.primal.dof->begin_active();
		auto endc_time = slab->time.primal.dof->end();
		
		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);
			
			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.primal.fe->dofs_per_cell
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
 					jj < slab->time.primal.fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.primal.dof->n_dofs(); ++i) {
 					(*u_trigger)[i] += (*u->x[0])[
 						i
 						// time offset
 						+ slab->space.primal.dof->n_dofs() *
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
					fe_values_time.quadrature_point(qt)[0] // t_trigger
				);
			}
		}
	}
}


template<int dim>
void
Fluid<dim>::
primal_do_data_output(
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
	
	DTM::pout
		<< "primal solution data output: dwr loop = "
		<< primal.data_output_dwr_loop
		<< std::endl;
	
	primal.data_output_time_value = parameter_set->time.fluid.t0;
	
	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	auto u = primal.storage.u->begin();
	
	if (!primal.data_output_trigger_type_fixed) {
		// I_n output mode (output on natural Q_n support points in time)
		while (slab != grid->slabs.end()) {
			primal_do_data_output_on_slab_Qn_mode(slab,u,dwr_loop);
			
			++slab;
			++u;
		}
	}
	else {
		// fixed trigger output mode
		
		while (slab != grid->slabs.end()) {
			primal_do_data_output_on_slab(slab,u,dwr_loop);
			
			++slab;
			++u;
		}
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

	{
		auto slab = grid->slabs.begin();
		for (auto &element : *dual.storage.z) {
			for (unsigned int j{0}; j < element.x.size(); ++j) {
				element.x[j] = std::make_shared< dealii::Vector<double> > ();

				Assert(slab != grid->slabs.end(), dealii::ExcInternalError());

				Assert(
					slab->space.dual.dof.use_count(),
					dealii::ExcNotInitialized()
				);

				Assert(
					slab->time.dual.dof.use_count(),
					dealii::ExcNotInitialized()
				);

				element.x[j]->reinit(
					slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()
				);
			}
			++slab;
		}
	}
}


template<int dim>
void
Fluid<dim>::
dual_assemble_system(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
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

		DTM::pout << "dynamic fluid: assemble space-time slab dual operator matrix...";
		Assert(dual.L.use_count(), dealii::ExcNotInitialized());
		dual_assembler.assemble(
			dual.L,
			slab
		);

#ifdef DEBUG
		{
			if (slab->t_n == 8.0)
			{
				std::ostringstream filename;
				filename << "L_dual"
						<< ".gpl";
				std::ofstream out(filename.str().c_str(), std::ios_base::out);

				dual.L->print(out);
				out.close();
				DTM::pout << "printed dual.L" << std::endl;
			}
		}
#endif

		DTM::pout << " (done)" << std::endl;
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
		slab->space.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Mzn->reinit(
		slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()
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
		slab->space.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.Je->reinit(
		slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()
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
		slab->space.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.dual.dof.use_count(),
		dealii::ExcNotInitialized()
	);

	dual.b->reinit(
		slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()
	);
	*dual.b = 0.;

	dual.b->add(1., *dual.Mzn);
	dual.b->add(1., *dual.Je);

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
				const dealii::QGauss<1> support_points(
					slab->time.dual.fe->tensor_degree()+1
				);

				auto cell_time = slab->time.dual.dof->begin_active();
				auto endc_time = slab->time.dual.dof->end();

				dealii::FEValues<1> time_fe_values(
					*slab->time.dual.mapping,
					*slab->time.dual.fe,
					support_points,
					dealii::update_quadrature_points
				);

				for ( ; cell_time != endc_time; ++cell_time) {
					time_fe_values.reinit(cell_time);

					for (unsigned int qt{0}; qt < support_points.size(); ++qt) {
						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.dual.dof,
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
								+ slab->space.dual.dof->n_dofs() *
									(cell_time->index()
									* slab->time.dual.fe->dofs_per_cell)
								// local in time dof
								+ slab->space.dual.dof->n_dofs() * qt
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
				const dealii::QGauss<1> support_points(
					slab->time.dual.fe->tensor_degree()+1
				);

				auto cell_time = slab->time.dual.dof->begin_active();
				auto endc_time = slab->time.dual.dof->end();

				dealii::FEValues<1> time_fe_values(
					*slab->time.dual.mapping,
					*slab->time.dual.fe,
					support_points,
					dealii::update_quadrature_points
				);

				for ( ; cell_time != endc_time; ++cell_time) {
					time_fe_values.reinit(cell_time);

					for (unsigned int qt{0}; qt < support_points.size(); ++qt) {
						std::map<dealii::types::global_dof_index,double>
							boundary_values_qt;

						dealii::VectorTools::interpolate_boundary_values (
							*slab->space.dual.dof,
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
								+ slab->space.dual.dof->n_dofs() *
									(cell_time->index()
									* slab->time.dual.fe->dofs_per_cell)
								// local in time dof
								+ slab->space.dual.dof->n_dofs() * qt
							;

							boundary_values[idx] = el.second;
						}
					}
				} // no slip
			}
		} // for each (boundary) colour

//#ifdef DEBUG
//		{
//
//			std::ostringstream filename;
//			filename << "L"
//			<< ".gpl";
//			std::ofstream out(filename.str().c_str(), std::ios_base::out);
//
//			dual.L->print(out);
//			out.close();
//		}
//#endif

// 		dealii::MatrixTools::apply_boundary_values(
// 			boundary_values,
// 			*dual.L,
// 			*u->x[0],
// 			*dual.b
// 		);

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
			double diagonal_scaling_value{0.};

			for (dealii::types::global_dof_index i{0}; i < A->m(); ++i) {
				if (std::abs(A->el(i,i)) > std::abs(diagonal_scaling_value)) {
					diagonal_scaling_value = A->el(i,i);
				}
			}

			if (diagonal_scaling_value == double(0.)) {
				diagonal_scaling_value = double(1.);
			}

			Assert(
				(diagonal_scaling_value != double(0.)),
				dealii::ExcInternalError()
			);

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

// #ifdef DEBUG
// 			{
// 				std::ostringstream filename;
// 				filename << "L"
// 				<< ".gpl";
// 				std::ofstream out(filename.str().c_str(), std::ios_base::out);
//
// 				dual.L->print(out);
// 				out.close();
// 			}
// #endif
		}
	}
	DTM::pout << " (done)" << std::endl;

	////////////////////////////////////////////////////////////////////////////
	// condense hanging nodes in system matrix, if any
	//
	slab->spacetime.dual.constraints->condense(*dual.L);


#ifdef DEBUG
		{
			if (slab->t_n == 8.0)
			{
				std::ostringstream filename;
				filename << "L_dual2"
						<< ".gpl";
				std::ofstream out(filename.str().c_str(), std::ios_base::out);

				dual.L->print(out);
				out.close();
				DTM::pout << "printed dual.L" << std::endl;
			}
		}
#endif

	////////////////////////////////////////////////////////////////////////////
	// solve linear system with direct solver
	//

	DTM::pout << "dwr-instatfluid: setup direct lss and solve...";

	dealii::SparseDirectUMFPACK iA;
	iA.initialize(*dual.L);
	iA.vmult(*z->x[0], *dual.b);

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
dual_do_backward_TMS() {
	////////////////////////////////////////////////////////////////////////////
	// prepare time marching scheme (TMS) loop
	//

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

		if (n == N) {
			////////////////////////////////////////////////////////////////////////////
			// interpolate (or project) initial value(s)
			//

			dual.zn = std::make_shared< dealii::BlockVector<double> > ();
			Assert(
				slab->space.dual.block_sizes.use_count(),
				dealii::ExcNotInitialized()
			);
			dual.zn->reinit(*slab->space.dual.block_sizes);
			*dual.zn = 0.;

			Assert(slab->space.dual.mapping.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
			Assert(dual.zn.use_count(), dealii::ExcNotInitialized());

			dealii::VectorTools::interpolate(
				*slab->space.dual.mapping,
				*slab->space.dual.dof,
				dealii::ZeroFunction<dim>(dim+1),
				*dual.zn
			);

			// NOTE: after the first dwr-loop the initial triangulation could have
			//       hanging nodes. Therefore,
			// distribute hanging node constraints to make the result continuous again:
			slab->space.dual.constraints->distribute(
				*dual.zn
			);
		}
		else {
			// not the last slab: transfer zm solution to zn solution
			Assert(dual.zm.use_count(), dealii::ExcNotInitialized());

			dual.zn = std::make_shared< dealii::BlockVector<double> > ();
			Assert(
				slab->space.dual.block_sizes.use_count(),
				dealii::ExcNotInitialized()
			);
			dual.zn->reinit(*slab->space.dual.block_sizes);
			*dual.zn = 0.;

			// for n < N interpolate between two (different) spatial meshes
			// the solution z(t_m)|_{I_{n+1}}  to  z(t_n)|_{I_n}
			dealii::VectorTools::interpolate_to_different_mesh(
				// solution on I_{n+1}:
				*std::next(slab)->space.dual.dof,
				*dual.zm,
				// solution on I_n:
				*slab->space.dual.dof,
				*slab->space.dual.constraints,
				*dual.zn
			);

			slab->space.dual.constraints->distribute(
				*dual.zn
			);
		}

		// assemble slab problem
		dual_assemble_system(slab);
		dual_assemble_rhs(slab);

		// solve slab problem (i.e. apply boundary values and solve for z0)
		dual_solve_slab_problem(slab,z);

		////////////////////////////////////////////////////////////////////////
		// do postprocessings on the solution
		//

		// evaluate solution z(t_m)
		dual.zm = std::make_shared< dealii::BlockVector<double> > ();
		Assert(
			slab->space.dual.block_sizes.use_count(),
			dealii::ExcNotInitialized()
		);
		dual.zm->reinit(*slab->space.dual.block_sizes);
		*dual.zm = 0.;

		{
			dealii::FEValues<1> fe_face_values_time(
				*slab->time.dual.mapping,
				*slab->time.dual.fe,
				dealii::QGaussLobatto<1>(2),
				dealii::update_values
			);

			auto cell_time = slab->time.dual.dof->begin_active();
			auto last_cell_time = cell_time;
			auto endc_time = slab->time.dual.dof->end();

			for ( ; cell_time != endc_time; ++cell_time) {
				last_cell_time=cell_time;
			}

			cell_time=last_cell_time;
			{
				Assert((cell_time!=endc_time), dealii::ExcInternalError());
				fe_face_values_time.reinit(cell_time);

				// evaluate solution for t_m of Q_n
				for (unsigned int jj{0};
					jj < slab->time.dual.fe->dofs_per_cell; ++jj)
				for (dealii::types::global_dof_index i{0};
					i < slab->space.dual.dof->n_dofs(); ++i) {
					(*dual.zm)[i] += (*z->x[0])[
						i
						// time offset
						+ slab->space.dual.dof->n_dofs() *
							(cell_time->index() * slab->time.dual.fe->dofs_per_cell)
						// local in time dof
						+ slab->space.dual.dof->n_dofs() * jj
					] * fe_face_values_time.shape_value(jj,0);
				}
			}
		}

		////////////////////////////////////////////////////////////////////////
		// prepare next I_n slab problem:
		//

		--n;
		--slab;
		--z;

		////////////////////////////////////////////////////////////////////////
		// allow garbage collector to clean up memory
		//

		dual.L = nullptr;
		dual.b = nullptr;

		dual.Mzn = nullptr;
		dual.Je = nullptr;

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
	Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.dual.partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	dual.data_output->set_DoF_data(
		slab->space.dual.dof,
		slab->space.dual.partitioning_locally_owned_dofs
	);

 	auto z_trigger = std::make_shared< dealii::BlockVector<double> > ();
 	Assert(
 		slab->space.dual.block_sizes.use_count(),
 		dealii::ExcNotInitialized()
 	);
 	z_trigger->reinit(
 		*slab->space.dual.block_sizes
 	);

	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// fe face values time: time face (I_n) information
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.dual.mapping,
			*slab->time.dual.fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);

		Assert(
			slab->time.dual.dof.use_count(),
			dealii::ExcNotInitialized()
		);

		auto cell_time = slab->time.dual.dof->begin_active();
		auto endc_time = slab->time.dual.dof->end();

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
				*slab->time.dual.mapping,
				*slab->time.dual.fe,
				quad_time,
				dealii::update_values |
				dealii::update_quadrature_points
			);

			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices(slab->time.dual.fe->dofs_per_cell);
			cell_time->get_dof_indices(local_dof_indices);

			for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
 				*z_trigger = 0.;

 				// evaluate solution for t_q
 				for (
 					unsigned int jj{0};
 					jj < slab->time.dual.fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.dual.dof->n_dofs(); ++i) {
 					(*z_trigger)[i] += (*z->x[0])[
 						i
 						// time offset
 						+ slab->space.dual.dof->n_dofs() *
 							local_dof_indices[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

				std::cout
					<< "dual output generated for t = "
					<< fe_values_time.quadrature_point(qt)[0] // t_trigger
					<< std::endl;

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
	Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
	Assert(
		slab->space.dual.partitioning_locally_owned_dofs.use_count(),
		dealii::ExcNotInitialized()
	);
	dual.data_output->set_DoF_data(
		slab->space.dual.dof,
		slab->space.dual.partitioning_locally_owned_dofs
	);

	auto z_trigger = std::make_shared< dealii::BlockVector<double> > ();
 	Assert(
 		slab->space.dual.block_sizes.use_count(),
 		dealii::ExcNotInitialized()
 	);
 	z_trigger->reinit(
 		*slab->space.dual.block_sizes
 	);

	std::ostringstream filename;
	filename
		<< "dual-dwr_loop-"
		<< std::setw(setw_value_dwr_loops) << std::setfill('0') << dwr_loop;

	{
		// TODO: check natural time quadrature (Gauss, Gauss-Radau, etc.)
		//       from input or generated fe<1>
		//
		// create fe values
		dealii::FEValues<1> fe_values_time(
			*slab->time.dual.mapping,
			*slab->time.dual.fe,
			dealii::QGauss<1>(slab->time.dual.fe->tensor_degree()+1), // here
			dealii::update_values |
			dealii::update_quadrature_points
		);

		auto cell_time = slab->time.dual.dof->begin_active();
		auto endc_time = slab->time.dual.dof->end();

		for ( ; cell_time != endc_time; ++cell_time) {
			fe_values_time.reinit(cell_time);

			std::vector< dealii::types::global_dof_index > local_dof_indices_time(
				slab->time.dual.fe->dofs_per_cell
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
 					jj < slab->time.dual.fe->dofs_per_cell; ++jj) {
 				for (
 					dealii::types::global_dof_index i{0};
 					i < slab->space.dual.dof->n_dofs(); ++i) {
 					(*z_trigger)[i] += (*z->x[0])[
 						i
 						// time offset
 						+ slab->space.dual.dof->n_dofs() *
 							local_dof_indices_time[jj]
 					] * fe_values_time.shape_value(jj,qt);
 				}}

// 				std::cout
// 					<< "output generated for t = "
// 					<< fe_values_time.quadrature_point(qt)[0]
// 					<< std::endl;

				dual.data_output->write_data(
					filename.str(),
					z_trigger,
					primal.data_postprocessor,
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

	DTM::pout
		<< "dual solution data output: dwr loop = "
		<< dual.data_output_dwr_loop
		<< std::endl;

	dual.data_output_time_value = parameter_set->time.fluid.t0;

	Assert(grid->slabs.size(), dealii::ExcNotInitialized());
	auto slab = grid->slabs.begin();
	auto z = dual.storage.z->begin();

	if (!dual.data_output_trigger_type_fixed) {
		// I_n output mode (output on natural Q_n support points in time)
		while (slab != grid->slabs.end()) {
			dual_do_data_output_on_slab_Qn_mode(slab,z,dwr_loop);

			++slab;
			++z;
		}
	}
	else {
		// fixed trigger output mode

		while (slab != grid->slabs.end()) {
			dual_do_data_output_on_slab(slab,z,dwr_loop);

			++slab;
			++z;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// functional values
//
template<int dim>
void
Fluid<dim>::
compute_functional_values(
		std::shared_ptr< dealii::Vector<double> > un,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	Assert(dim==2, dealii::ExcNotImplemented());
	
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
	
	double pressure_diff = pressure_front - pressure_back;
	
	// save pressure difference to text file
	{
		std::ofstream p_out;
		// append instead pressure difference to text file (ios_base::app):
		p_out.open("pressure.log", std::ios_base::app);
		p_out
			<< slab->t_n << "," << pressure_diff
			<< std::endl;
		p_out.close();
	}
	
	DTM::pout
		<< "------------------"
		<< std::endl;
		
	DTM::pout
		<< "Pressure difference:  "
		<< "   " << std::setprecision(16) << pressure_diff << std::endl;
	
	// DTM::pout
	//	<< "P-front: "
	// 	<< "   " << std::setprecision(16) << pressure_front << std::endl;
	
	// DTM::pout
	//	<< "P-back:  "
	//	<< "   " << std::setprecision(16) << pressure_back << std::endl;
	
	DTM::pout << "------------------" << std::endl;
	
	// Compute drag and lift via line integral
	compute_drag_lift_tensor(
		un,
		slab
	);
	
	DTM::pout << "------------------" << std::endl << std::endl;
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
		*slab->space.primal.dof,
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
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
) {
	const dealii::QGauss<dim - 1> face_quadrature_formula(3);
	dealii::FEFaceValues<dim> fe_face_values(*slab->space.primal.fe, face_quadrature_formula,
									 dealii::update_values | dealii::update_gradients | dealii::update_normal_vectors |
										 dealii::update_JxW_values | dealii::update_quadrature_points);

	const unsigned int dofs_per_cell = slab->space.primal.fe->dofs_per_cell;
	const unsigned int n_face_q_points = face_quadrature_formula.size();

	std::vector<unsigned int> local_dof_indices(dofs_per_cell);
	std::vector<dealii::Vector<double>> face_solution_values(n_face_q_points,
													 dealii::Vector<double>(dim + 1));

	std::vector<std::vector<dealii::Tensor<1, dim>>>
		face_solution_grads(n_face_q_points, std::vector<dealii::Tensor<1, dim>>(dim + 1));

	dealii::Tensor<1, dim> drag_lift_value;

	typename dealii::DoFHandler<dim>::active_cell_iterator
		cell = slab->space.primal.dof->begin_active(),
		endc = slab->space.primal.dof->end();

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
		} // end boundary fluid::types::space::boundary_id::prescribed_obstacle for fluid
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


	DTM::pout << "Face drag:   "
			  << "   " << std::setprecision(16) << drag_lift_value[0] << std::endl;
	DTM::pout << "Face lift:   "
			  << "   " << std::setprecision(16) << drag_lift_value[1] << std::endl;

	// save drag and lift values to text files
	{
		std::ofstream drag_out;
		drag_out.open("drag.log", std::ios_base::app); // append instead of overwrite
		drag_out << slab->t_n << "," << drag_lift_value[0] << std::endl;
		drag_out.close();
	}
	
	{
		std::ofstream lift_out;
		lift_out.open("lift.log", std::ios_base::app); // append instead of overwrite
		lift_out << slab->t_n << "," << drag_lift_value[1]  << std::endl;
		lift_out.close();
	}
}

} // namespace

#include "Fluid.inst.in"
