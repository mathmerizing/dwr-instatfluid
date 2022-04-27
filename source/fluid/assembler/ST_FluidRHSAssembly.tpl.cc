/**
 * @file ST_FluidRHSAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 *
 * @Date 2022-01-15, Fluid Newton RHS, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-05, cleanups, UK
 * @date 2021-11-05, initialvalue for ST Stokes, JR
 * 
 * @date 2020-01-08, supg, MPB
 * @date 2019-09-18, space-time initialvalue, UK
 * @date 2019-09-13, space-time force, UK
 * @date 2019-08-30, space-time diffusion, UK
 * @date 2019-01-28, space-time parabolic, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-23, ewave, UK
 * @date 2015-05-18, AWAVE/C++.11, UK
 * @date 2014-04-09, Tensor, UK
 * @date 2012-03-13, UK
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
#include <fluid/assembler/ST_FluidRHSAssembly.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/quadrature_lib.h>

// C++ includes
#include <functional>

namespace fluidrhs {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
FluidRHSAssembly<dim>::FluidRHSAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
		const dealii::Quadrature<dim>    &quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>       &mapping_time,
		const dealii::Quadrature<1>    &quad_time,
		const dealii::Quadrature<1>    &face_nodes
	) :
	// init space
	space_fe_values(
		mapping_space,
		fe_space,
		quad_space,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	),
	space_local_dof_indices(fe_space.dofs_per_cell),
	space_phi(fe_space.dofs_per_cell),
	space_symgrad_phi(fe_space.dofs_per_cell),
	space_grad_phi(fe_space.dofs_per_cell),
	space_div_phi(fe_space.dofs_per_cell),
	space_psi(fe_space.dofs_per_cell),
	// init time
	time_fe_values(
		mapping_time,
		fe_time,
		quad_time,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	) ,
	//
	time_fe_face_values(
		mapping_time,
		fe_time,
		face_nodes,
		dealii::update_values |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	) ,
	//
	time_fe_face_values_neighbor(
		mapping_time,
		fe_time,
		face_nodes,
		dealii::update_values |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	) ,
	time_local_dof_indices(fe_time.dofs_per_cell),
	spacetime_JxW(0),
	v(),
	symgrad_v(),
	grad_v(),
	div_v(0),
	partial_t_v(),
	p(0),
	u_plus(),
	u_minus(),
	viscosity(0)
{}

template<int dim>
FluidRHSAssembly<dim>::FluidRHSAssembly(const FluidRHSAssembly &scratch) :
	space_fe_values(
		scratch.space_fe_values.get_mapping(),
		scratch.space_fe_values.get_fe(),
		scratch.space_fe_values.get_quadrature(),
		scratch.space_fe_values.get_update_flags()
	),
	space_local_dof_indices(scratch.space_local_dof_indices),
	//
	space_phi(scratch.space_phi),
	space_symgrad_phi(scratch.space_symgrad_phi),
	space_grad_phi(scratch.space_grad_phi),
	space_div_phi(scratch.space_div_phi),
	space_psi(scratch.space_psi),
	//

	time_fe_values(
		scratch.time_fe_values.get_mapping(),
		scratch.time_fe_values.get_fe(),
		scratch.time_fe_values.get_quadrature(),
		scratch.time_fe_values.get_update_flags()
	),
	//
	time_fe_face_values(
		scratch.time_fe_face_values.get_mapping(),
		scratch.time_fe_face_values.get_fe(),
		scratch.time_fe_face_values.get_quadrature(),
		scratch.time_fe_face_values.get_update_flags()
	),
	//
	time_fe_face_values_neighbor(
		scratch.time_fe_face_values_neighbor.get_mapping(),
		scratch.time_fe_face_values_neighbor.get_fe(),
		scratch.time_fe_face_values_neighbor.get_quadrature(),
		scratch.time_fe_face_values_neighbor.get_update_flags()
	),
	//
	time_local_dof_indices(scratch.time_local_dof_indices),
	//
	spacetime_JxW(scratch.spacetime_JxW),
	v(scratch.v),
	symgrad_v(scratch.symgrad_v),
	grad_v(scratch.grad_v),
	div_v(scratch.div_v),
	partial_t_v(scratch.partial_t_v),
	p(scratch.p),
    u_plus(scratch.u_plus),
	u_minus(scratch.u_minus),
	viscosity(scratch.viscosity)
{}

}

namespace CopyData {

template<int dim>
FluidRHSAssembly<dim>::FluidRHSAssembly(
	const dealii::FiniteElement<dim> &fe_s,
	const dealii::FiniteElement<1> &fe_t,
	const dealii::types::global_dof_index &n_global_active_cells_t) :
	vi_rhs_vector(
			n_global_active_cells_t,// n_cells_time
			dealii::Vector<double >(
					fe_s.dofs_per_cell * fe_t.dofs_per_cell)
	),
	local_dof_indices(
	  n_global_active_cells_t, //n_cells_time
	  std::vector<dealii::types::global_dof_index>(
			  fe_s.dofs_per_cell * fe_t.dofs_per_cell
	  )
	) {
}

template<int dim>
FluidRHSAssembly<dim>::FluidRHSAssembly(const FluidRHSAssembly &copydata) :
	vi_rhs_vector(copydata.vi_rhs_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
Assembler<dim>::
set_functions(
	std::shared_ptr< dealii::Function<dim> > viscosity) {
	function.viscosity = viscosity;
}

template<int dim>
void
Assembler<dim>::
set_symmetric_stress(
	bool use_symmetric_stress) {
	symmetric_stress = use_symmetric_stress;
}

template<int dim>
void
Assembler<dim>::
assemble(
	std::shared_ptr< dealii::Vector<double> > _Fu,  // output
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< dealii::Vector<double> > _u,
	bool _nonlin
) {
	////////////////////////////////////////////////////////////////////////////
	// check
	Assert(_Fu.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.primal.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->time.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.primal.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());

	Assert(slab->spacetime.primal.constraints.use_count(), dealii::ExcNotInitialized());
	
	Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
	////////////////////////////////////////////////////////////////////////////
	// init
	
	Fu = _Fu;
	u = _u;
	nonlin = _nonlin;
	
	space.dof = slab->space.primal.fe_info->dof;
	space.fe = slab->space.primal.fe_info->fe;
	space.mapping = slab->space.primal.fe_info->mapping;
	space.constraints = slab->space.primal.fe_info->constraints;
	
	time.dof = slab->time.primal.fe_info->dof;
	time.fe = slab->time.primal.fe_info->fe;
	time.mapping = slab->time.primal.fe_info->mapping;

	// FEValuesExtractors
	convection = 0;
 	pressure   = dim;

	////////////////////////////////////////////////////////////////////////////
	// WorkStream assemble
	const dealii::QGaussLobatto<dim> quad_space(
		std::max(
			std::max(
				space.fe->base_element(0).base_element(0).tensor_degree(),
				space.fe->base_element(0).base_element(1).tensor_degree()
			),
			static_cast<unsigned int> (1)
		) + 1
	);

	const dealii::QGauss<1> quad_time(
		time.fe->tensor_degree()+1
	);

	dealii::QGaussLobatto<1> face_nodes(2);

	time.n_global_active_cells = slab->time.tria->n_global_active_cells();

	typedef
	dealii::
	FilteredIterator<const typename dealii::DoFHandler<dim>::active_cell_iterator>
	CellFilter;
	
	// Using WorkStream to assemble.
	dealii::WorkStream::
	run(
		CellFilter(
			dealii::IteratorFilters::LocallyOwnedCell(),
			space.dof->begin_active()
		),
		CellFilter(
			dealii::IteratorFilters::LocallyOwnedCell(),
			space.dof->end()
		),
		std::bind(
			&Assembler<dim>::local_assemble_cell,
			this,
			std::placeholders::_1,
			std::placeholders::_2,
			std::placeholders::_3
		),
		std::bind(
			&Assembler<dim>::copy_local_to_global_cell,
			this,
			std::placeholders::_1
		),
		Assembly::Scratch::FluidRHSAssembly<dim> (
			*space.fe,
			*space.mapping,
			quad_space,
			*time.fe,
			*time.mapping,
			quad_time,
			face_nodes
		),
		Assembly::CopyData::FluidRHSAssembly<dim> (
			*space.fe,
			*time.fe,
			time.n_global_active_cells
		)
	);
}

/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::FluidRHSAssembly<dim> &scratch,
	Assembly::CopyData::FluidRHSAssembly<dim> &copydata) {
	cell->get_dof_indices(scratch.space_local_dof_indices);
	scratch.space_fe_values.reinit(cell);
	
	auto cell_time = time.dof->begin_active();
	auto endc_time = time.dof->end();
	
	unsigned int n;
	for ( ; cell_time != endc_time; ++cell_time) {
		n = cell_time->index();
//		std::cout << "n: " << n << std::endl;
		copydata.vi_rhs_vector[n] = 0;
		
		// dof mapping
		cell_time->get_dof_indices(scratch.time_local_dof_indices);
		for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
		for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii) {
			copydata.local_dof_indices[n][
				i + ii*space.fe->dofs_per_cell
			] =
				scratch.space_local_dof_indices[i]
				+ scratch.time_local_dof_indices[ii]*space.dof->n_dofs();
		}
		
		// assemble:volume
		scratch.time_fe_values.reinit(cell_time);
		for ( unsigned int qt{0}; qt < scratch.time_fe_values.n_quadrature_points; ++qt){
			function.viscosity->set_time(scratch.time_fe_values.quadrature_point(qt)[0]);

			for ( unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {
				scratch.viscosity = function.viscosity->value(
					scratch.space_fe_values.quadrature_point(q),0
				);

				for ( unsigned int k{0} ; k < space.fe->dofs_per_cell ; ++k) {
					scratch.space_phi[k] =
							scratch.space_fe_values[convection].value(k,q);
					scratch.space_symgrad_phi[k] =
							scratch.space_fe_values[convection].symmetric_gradient(k,q);
					scratch.space_grad_phi[k] =
							scratch.space_fe_values[convection].gradient(k,q);
					scratch.space_div_phi[k] =
							scratch.space_fe_values[convection].divergence(k,q);
					scratch.space_psi[k] =
							scratch.space_fe_values[pressure].value(k,q);
 				}

				scratch.partial_t_v = 0;
				scratch.v 		    = 0;
				scratch.symgrad_v   = 0;
				scratch.grad_v      = 0;
				scratch.div_v       = 0;
				scratch.p           = 0;

				for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell; ++ii)
				for ( unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
					//correct ST solution vector entry
					double u_i_ii = (*u)[
						scratch.space_local_dof_indices[i]
						   	// time offset
							+ space.dof->n_dofs() *
							   (n * time.fe->dofs_per_cell)
							// local in time dof
							+ space.dof->n_dofs() * ii
							];
					//partial_t v(qt,qx)
					scratch.partial_t_v += u_i_ii
							* scratch.space_fe_values[convection].value(i,q)
							* scratch.time_fe_values.shape_grad(ii,qt)[0];

					//all other evals use shape values in time, so multiply only once
					u_i_ii *= scratch.time_fe_values.shape_value(ii,qt);

					//v
					scratch.v += u_i_ii * scratch.space_phi[i];

					//symgrad v
					scratch.symgrad_v += u_i_ii * scratch.space_symgrad_phi[i];

					//grad v
					scratch.grad_v += u_i_ii * scratch.space_grad_phi[i];

					//div v
					scratch.div_v += u_i_ii * scratch.space_div_phi[i];

					//p
					scratch.p += u_i_ii * scratch.space_psi[i];

				}
				scratch.spacetime_JxW = scratch.space_fe_values.JxW(q) * scratch.time_fe_values.JxW(qt);
				// for Navier-Stokes assemble Convection term
				if ( nonlin ){
					for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell; ++ii)
					for (unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
						copydata.vi_rhs_vector[n](
								i + ii*space.fe->dofs_per_cell
						) +=
							// convection convection term
							( scratch.space_phi[i]*
									scratch.time_fe_values.shape_value(ii,qt)*
									scratch.grad_v*scratch.v*
									scratch.spacetime_JxW
							);
					}
				}

				for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell; ++ii)
				for (unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
					copydata.vi_rhs_vector[n](
							i + ii*space.fe->dofs_per_cell
					) +=
							// convection time derivative
							(	scratch.space_phi[i] *
									scratch.time_fe_values.shape_value(ii,qt) *
									scratch.partial_t_v * scratch.spacetime_JxW
							)
							// convection Laplacian
							+ (
								(symmetric_stress ?
									(
										scratch.space_symgrad_phi[i]
											  *scratch.time_fe_values.shape_value(ii,qt)*

											scratch.viscosity * 2. *
											  scratch.symgrad_v
									):
									(
										scalar_product(
											scratch.space_grad_phi[i]
											   *scratch.time_fe_values.shape_value(ii,qt),
											 scratch.viscosity * scratch.grad_v
										)
									)
								) * scratch.spacetime_JxW
							)
							// pressure
							- (scratch.space_div_phi[i]
									 *scratch.time_fe_values.shape_value(ii,qt)*
								 scratch.p * scratch.spacetime_JxW
							 )
							// div-free constraint
							+ ( scratch.space_psi[i]
								 *scratch.time_fe_values.shape_value(ii,qt)*
								 scratch.div_v*
								 scratch.spacetime_JxW)
					;
				}
			}//space qp

		}//time qp


 		// prepare [.]_t_m trace operator
 		scratch.time_fe_face_values.reinit(cell_time);
		// assemble: face (w^+ * u^-)
 		for ( unsigned int q{0} ; q < scratch.space_fe_values.n_quadrature_points ; ++q) {
 			scratch.u_plus = 0;
 			for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell ; ++ii)
			for ( unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
				//correct ST solution vector entry
				double u_i_ii = (*u)[
						scratch.space_local_dof_indices[i]
						// time offset
						+ space.dof->n_dofs() *
						   (n * time.fe->dofs_per_cell)
						// local in time dof
						+ space.dof->n_dofs() * ii
						];
				// function eval u(x,t_0)^+
				scratch.u_plus +=
						u_i_ii * scratch.space_fe_values[convection].value(i,q)
							*scratch.time_fe_face_values.shape_value(ii,0);
			}

 			for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell ; ++ii)
			for ( unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
				copydata.vi_rhs_vector[n](
						i + ii*space.fe->dofs_per_cell
				) +=
				  // trace operator: w(x,t_0)^+ * u(x,t_0)^+
						scratch.space_fe_values[convection].value(i,q)
						  * scratch.time_fe_face_values.shape_value(ii,0) *
						scratch.u_plus * scratch.space_fe_values.JxW(q)
				;
			}
 		}

		// assemble: face (w^+ * u^-)
		if ( n ){
			for ( unsigned int q{0} ; q < scratch.space_fe_values.n_quadrature_points; ++q){
				scratch.u_minus = 0;
				for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell ; ++ii)
				for ( unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
					//correct ST solution vector entry of previous time cell
					double u_i_ii = (*u)[
							scratch.space_local_dof_indices[i]
							// time offset
							+ space.dof->n_dofs() *
							   ((n-1) * time.fe->dofs_per_cell)
							// local in time dof
							+ space.dof->n_dofs() * ii
							];
					// function eval u^-
					scratch.u_minus +=
							u_i_ii * scratch.space_fe_values[convection].value(i,q)
								* scratch.time_fe_face_values_neighbor.shape_value(ii,1);
				}

				for ( unsigned int ii{0} ; ii < time.fe->dofs_per_cell ; ++ii)
				for ( unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
					copydata.vi_rhs_vector[n](
						i + ii * space.fe->dofs_per_cell
				    ) -=
				    	//trace operator: -w(x,t_0)^+ * u(x,t_0)^-
				    	scratch.space_fe_values[convection].value(i,q)
							* scratch.time_fe_face_values.shape_value(ii,0)
						* scratch.u_minus * scratch.space_fe_values.JxW(q)
						;
				}
			}

		}



		// update
		if ( (n+1) < time.n_global_active_cells) {
			scratch.time_fe_face_values_neighbor.reinit(cell_time);
		}

	}
}

template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::FluidRHSAssembly<dim> &copydata) {
	Assert(copydata.vi_rhs_vector.size(), dealii::ExcNotInitialized());
	Assert(copydata.vi_rhs_vector.size() == copydata.local_dof_indices.size(),
			dealii::ExcNotInitialized()
	);

	for ( unsigned int n{0} ; n < copydata.vi_rhs_vector.size(); ++ n){
		space.constraints->distribute_local_to_global(
			copydata.vi_rhs_vector[n],
			copydata.local_dof_indices[n],
			*Fu
		);
	}
}

}}}

#include "ST_FluidRHSAssembly.inst.in"
