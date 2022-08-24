/**
 * @file ST_FluidAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-05, dynamics for stokes, JR
 *
 * @date 2019-11-21, space-time stokes, UK
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
#include <fluid/assembler/ST_FluidAssembly.tpl.hh>
#include <fluid/QRightBox.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>

// C++ includes
#include <functional>

namespace fluid {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
FluidAssembly<dim>::FluidAssembly( // @suppress("Class members should be properly initialized")
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
		const dealii::Quadrature<dim>    &quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>       &mapping_time,
		const dealii::Quadrature<1>    &quad_time,
		const dealii::Quadrature<1>    &face_nodes) :
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
	space_phi(fe_space.dofs_per_cell),
	space_symgrad_phi(fe_space.dofs_per_cell),
	space_grad_phi(fe_space.dofs_per_cell),
	space_div_phi(fe_space.dofs_per_cell),
	space_psi(fe_space.dofs_per_cell),
	space_local_dof_indices(fe_space.dofs_per_cell),
	// init time
	time_fe_values(
		mapping_time,
		fe_time,
		quad_time,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	),
	time_fe_quad_values(
		mapping_time,
		fe_time,
		dealii::Quadrature<1>(fe_time.get_unit_support_points()),
		dealii::update_values |
		dealii::update_quadrature_points
	),
	time_fe_face_values(
		mapping_time,
		fe_time,
		face_nodes,
		dealii::update_values |
		dealii::update_quadrature_points
	),
	time_fe_face_values_neighbor(
		mapping_time,
		fe_time,
		face_nodes,
		dealii::update_values
	),
	time_zeta(fe_time.dofs_per_cell),
	time_grad_zeta(fe_time.dofs_per_cell),
	time_local_dof_indices(fe_time.dofs_per_cell),
	time_local_dof_indices_neighbor(fe_time.dofs_per_cell),
	v(),
	grad_v() {
}

template<int dim>
FluidAssembly<dim>::FluidAssembly(const FluidAssembly &scratch) :
	space_fe_values(
		scratch.space_fe_values.get_mapping(),
		scratch.space_fe_values.get_fe(),
		scratch.space_fe_values.get_quadrature(),
		scratch.space_fe_values.get_update_flags()
	),
	space_phi(scratch.space_phi),
	space_symgrad_phi(scratch.space_symgrad_phi),
	space_grad_phi(scratch.space_grad_phi),
	space_div_phi(scratch.space_div_phi),
	space_psi(scratch.space_psi),
	space_dofs_per_cell(scratch.space_dofs_per_cell),
	space_JxW(scratch.space_JxW),
	space_local_dof_indices(scratch.space_local_dof_indices),
	//
	time_fe_values(
		scratch.time_fe_values.get_mapping(),
		scratch.time_fe_values.get_fe(),
		scratch.time_fe_values.get_quadrature(),
		scratch.time_fe_values.get_update_flags()
	),
	time_fe_quad_values(
		scratch.time_fe_quad_values.get_mapping(),
		scratch.time_fe_quad_values.get_fe(),
		scratch.time_fe_quad_values.get_quadrature(),
		scratch.time_fe_quad_values.get_update_flags()
	),
	time_fe_face_values(
		scratch.time_fe_face_values.get_mapping(),
		scratch.time_fe_face_values.get_fe(),
		scratch.time_fe_face_values.get_quadrature(),
		scratch.time_fe_face_values.get_update_flags()
	),
	time_fe_face_values_neighbor(
		scratch.time_fe_face_values_neighbor.get_mapping(),
		scratch.time_fe_face_values_neighbor.get_fe(),
		scratch.time_fe_face_values_neighbor.get_quadrature(),
		scratch.time_fe_face_values_neighbor.get_update_flags()
	),
	time_zeta(scratch.time_zeta),
	time_grad_zeta(scratch.time_grad_zeta),
	time_dofs_per_cell(scratch.time_dofs_per_cell),
	time_JxW(scratch.time_JxW),
	time_local_dof_indices(scratch.time_local_dof_indices),
	time_local_dof_indices_neighbor(scratch.time_local_dof_indices_neighbor),
	//
	v(scratch.v),
	grad_v(scratch.grad_v),
	viscosity(scratch.viscosity) {
}

}

namespace CopyData {

template<int dim>
FluidAssembly<dim>::FluidAssembly(
	const dealii::FiniteElement<dim> &fe_s,
	const dealii::FiniteElement<1> &fe_t,
	const dealii::types::global_dof_index &n_global_active_cells_t) :
	vi_ui_matrix(
		n_global_active_cells_t, // n_cells time
		dealii::FullMatrix<double> (
			fe_s.dofs_per_cell * fe_t.dofs_per_cell,
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	),
	vi_ue_matrix(
		n_global_active_cells_t-1, // n_cells time -1
		dealii::FullMatrix<double> (
			fe_s.dofs_per_cell * fe_t.dofs_per_cell,
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	),
	local_dof_indices(
		n_global_active_cells_t, // n_cells time
		std::vector<dealii::types::global_dof_index>(
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	),
	local_dof_indices_neighbor(
		n_global_active_cells_t-1, // n_cells time - 1
		std::vector<dealii::types::global_dof_index>(
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	) {
}

template<int dim>
FluidAssembly<dim>::FluidAssembly(const FluidAssembly &copydata) :
	vi_ui_matrix(copydata.vi_ui_matrix),
	vi_ue_matrix(copydata.vi_ue_matrix),
	local_dof_indices(copydata.local_dof_indices),
	local_dof_indices_neighbor(copydata.local_dof_indices_neighbor) {
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
set_time_quad_type(
	std::string _quad_type) {
	time.quad_type = _quad_type;
}

template<int dim>
void
Assembler<dim>::
assemble(
	std::shared_ptr< dealii::SparseMatrix<double> > _L,
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< dealii::Vector<double> > _u,
	bool _nonlin
) {
	////////////////////////////////////////////////////////////////////////////
	// check
	Assert(dim==2 || dim==3, dealii::ExcNotImplemented());
	
	Assert(_L.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.primal.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.primal.fe_info->constraints.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->time.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.primal.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.primal.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->spacetime.primal.constraints.use_count(), dealii::ExcNotInitialized());
	
	Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());
	
	////////////////////////////////////////////////////////////////////////////
	// init
	
	L = _L;
	u = _u;
	nonlin = _nonlin;
	
	space.dof = slab->space.primal.fe_info->dof;
	space.fe = slab->space.primal.fe_info->fe;
	space.mapping = slab->space.primal.fe_info->mapping;
	space.constraints = slab->space.primal.fe_info->constraints;
	
	time.dof = slab->time.primal.fe_info->dof;
	time.fe = slab->time.primal.fe_info->fe;
	time.mapping = slab->time.primal.fe_info->mapping;
	
 	spacetime.constraints = slab->spacetime.primal.constraints;
	
	// FEValuesExtractors
	convection = 0;
	pressure   = dim;
	
	// check fe: ((fluid)) = ((FE_Q^d, FE_Q))
	Assert(
		(space.fe->n_base_elements()==1),
		dealii::ExcMessage("fe not correct (fluid system)")
	);
	
	Assert(
		(space.fe->base_element(0).n_base_elements()==2),
		dealii::ExcMessage("fe: (fluid) not correct (FEQ+FEQ system)")
	);
	
	////////////////////////////////////////////////////////////////////////////
	// WorkStream assemble
	
	const dealii::QGauss<dim> quad_space(
		std::max(
			std::max(
				space.fe->base_element(0).base_element(0).tensor_degree(),
				space.fe->base_element(0).base_element(1).tensor_degree()
			),
			static_cast<unsigned int> (1)
		) + 2
	);

	const dealii::QGauss<1> quad_time(
		time.fe->tensor_degree()+2
	);

//	std::shared_ptr< dealii::Quadrature<1> > quad_time;
//	if (!time.quad_type.compare("Gauss-Lobatto")){
//		if (time.fe->tensor_degree()<1){
//			quad_time = std::make_shared<QRightBox<1>>();
//		}
//		else {
//			quad_time = std::make_shared<dealii::QGaussLobatto<1> >(time.fe->tensor_degree()+1);
//		}
//
//	}else {
//		quad_time = std::make_shared<dealii::QGauss<1> >(time.fe->tensor_degree()+1);
//	}

	const dealii::QGaussLobatto<1> face_nodes(2);
	
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
		std::bind (
			&Assembler<dim>::local_assemble_cell,
			this,
			std::placeholders::_1,
			std::placeholders::_2,
			std::placeholders::_3
		),
		std::bind (
			&Assembler<dim>::copy_local_to_global_cell,
			this,
			std::placeholders::_1
		),
		Assembly::Scratch::FluidAssembly<dim> (
			*space.fe,
			*space.mapping,
			quad_space,
			*time.fe,
			*time.mapping,
			quad_time,
			face_nodes
		),
		Assembly::CopyData::FluidAssembly<dim> (
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
	Assembly::Scratch::FluidAssembly<dim> &scratch,
	Assembly::CopyData::FluidAssembly<dim> &copydata) {
	cell->get_dof_indices(scratch.space_local_dof_indices);
	scratch.space_fe_values.reinit(cell);
	
	auto cell_time = time.dof->begin_active();
	auto endc_time = time.dof->end();
	
// 	for (unsigned int n{0}; n < time.n_global_active_cells; ++n)
	unsigned int n;
	for ( ; cell_time != endc_time; ++cell_time) {
		n=cell_time->index();
		copydata.vi_ui_matrix[n] = 0;
		
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
		
		// prefetch data
		
		
		
		
		// assemble: volume
		scratch.time_fe_values.reinit(cell_time);
		for (unsigned int qt{0}; qt < scratch.time_fe_values.n_quadrature_points; ++qt) {
			function.viscosity->set_time(scratch.time_fe_values.quadrature_point(qt)[0]);
			
			for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {
				scratch.viscosity = function.viscosity->value(
					scratch.space_fe_values.quadrature_point(q),0
				);
				
				for (unsigned int k{0}; k < space.fe->dofs_per_cell; ++k) {
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
				
				if (nonlin) {
					scratch.v 		    = 0;
					scratch.grad_v      = 0;

					// Space-Time version of do_function_values (https://www.dealii.org/current/doxygen/deal.II/fe_2fe__values_8cc_source.html#l03073)
					for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
					for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
						// correct ST solution vector entry
						double u_i_ii = (*u)[
											 scratch.space_local_dof_indices[i]
										// time offset
										+ space.dof->n_dofs() *
											scratch.time_local_dof_indices[ii]
															  ];
//							scratch.space_local_dof_indices[i]
//								// time offset
//								+ space.dof->n_dofs() *
//								   (n * time.fe->dofs_per_cell)
//								// local in time dof
//								+ space.dof->n_dofs() * ii
//								];

						// all other evals use shape values in time, so multiply only once
						u_i_ii *= scratch.time_fe_values.shape_value(ii,qt);

						// v
						scratch.v += u_i_ii * scratch.space_phi[i];

						// grad v
						scratch.grad_v += u_i_ii * scratch.space_grad_phi[i];
					}

					// for Navier-Stokes assemble convection term derivatives
					for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
					for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
					for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
					for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
						copydata.vi_ui_matrix[n](
							i + ii*space.fe->dofs_per_cell,
							j + jj*space.fe->dofs_per_cell
						) +=
							// convection C(u)_bb
							(
								scratch.space_phi[i] *
									scratch.time_fe_values.shape_value(ii,qt) *

								(
									scratch.space_grad_phi[j] * scratch.v
									+ scratch.grad_v * scratch.space_phi[j]
								)
									* scratch.time_fe_values.shape_value(jj,qt) *

								scratch.space_fe_values.JxW(q)
									* scratch.time_fe_values.JxW(qt)
							);
					}
				}

				for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
				for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
				for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
				for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
					copydata.vi_ui_matrix[n](
						i + ii*space.fe->dofs_per_cell,
						j + jj*space.fe->dofs_per_cell
					) +=
						// convection M_bb: w^+(x,t) * \partial_t b^+(x,t)
						(  scratch.space_fe_values[convection].value(i,q)
								* scratch.time_fe_values.shape_value(ii,qt) *

							scratch.space_fe_values[convection].value(j,q)
								* scratch.time_fe_values.shape_grad(jj,qt)[0] *

							scratch.space_fe_values.JxW(q)
								* scratch.time_fe_values.JxW(qt)
						)
						// convection A_bb
						+ (
							(symmetric_stress ?
									(
											scratch.space_symgrad_phi[i]
												* scratch.time_fe_values.shape_value(ii,qt) *

											scratch.viscosity * 2. *
											scratch.space_symgrad_phi[j]
												* scratch.time_fe_values.shape_value(jj,qt)
									) :
									(
											scalar_product(
													scratch.space_grad_phi[i]
														* scratch.time_fe_values.shape_value(ii,qt),

													scratch.viscosity *
													scratch.space_grad_phi[j]
														* scratch.time_fe_values.shape_value(jj,qt)
											)
									)
							) *
							
							scratch.space_fe_values.JxW(q)
								* scratch.time_fe_values.JxW(qt)
						)
						// pressure B_bp
						- (
							scratch.space_div_phi[i]
								* scratch.time_fe_values.shape_value(ii,qt) *

							scratch.space_psi[j]
								* scratch.time_fe_values.shape_value(jj,qt) *

							scratch.space_fe_values.JxW(q)
								* scratch.time_fe_values.JxW(qt)
						)
//						// div-free constraint B_pb
//						+ (
//							scratch.space_psi[i]
//								* scratch.time_fe_values.shape_value(ii,qt) *
//
//							scratch.space_div_phi[j]
//								* scratch.time_fe_values.shape_value(jj,qt) *
//
//							scratch.space_fe_values.JxW(q)
//								* scratch.time_fe_values.JxW(qt)
//						)
					;
				}
			} // x_q
		} // t_q


		// pointwise divergence free condition
		scratch.time_fe_quad_values.reinit(cell_time);
		for (unsigned int qt{0}; qt < scratch.time_fe_quad_values.n_quadrature_points; ++qt) {
			// assemble: div(v),phi_p
			for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {
				unsigned int ii = qt;
				unsigned int jj = qt;
				for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
				for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
					copydata.vi_ui_matrix[n](
						i + ii*space.fe->dofs_per_cell,
						j + jj*space.fe->dofs_per_cell
					) +=
						// div-free constraint B_pb
						scratch.space_fe_values[pressure].value(i,q) *

						scratch.space_fe_values[convection].divergence(j,q) *

						scratch.space_fe_values.JxW(q)
					;
				}
			}
		} // t_q


 		// prepare [.]_t_m trace operator
 		scratch.time_fe_face_values.reinit(cell_time);
 		// assemble: face (w^+ * u^+)
 		for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {
 			for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
 			for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
 			for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
 			for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
 				copydata.vi_ui_matrix[n](
 					i + ii*space.fe->dofs_per_cell,
 					j + jj*space.fe->dofs_per_cell
 				) +=
 					scratch.space_fe_values[convection].value(i,q)
 						* scratch.time_fe_face_values.shape_value(ii,0) *

 					scratch.space_fe_values[convection].value(j,q)
 						* scratch.time_fe_face_values.shape_value(jj,0) *

 					scratch.space_fe_values.JxW(q)
 				;
 			}
 		}
		
 		// assemble: face (w^+ * u^-)
 		if (n) {
 			copydata.vi_ue_matrix[n-1] = 0;

 			// dof mapping
 			for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
 			for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii) {
 				copydata.local_dof_indices_neighbor[n-1][
 					i + ii*space.fe->dofs_per_cell
 				] =
 					scratch.space_local_dof_indices[i]
 					+ scratch.time_local_dof_indices_neighbor[ii]
 						* space.dof->n_dofs();
 			}}

 			// assemble: face (w^+ * u^-)
 			for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {
 				for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
 				for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
 				for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
 				for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
 					copydata.vi_ue_matrix[n-1](
 						i + ii*space.fe->dofs_per_cell,
 						j + jj*space.fe->dofs_per_cell
 					) -=
 						// trace operator: - w(x,t_0)^+ * u(x,t_0)^-
 						scratch.space_fe_values[convection].value(i,q)
 							* scratch.time_fe_face_values.shape_value(ii,0) *

 						scratch.space_fe_values[convection].value(j,q)
 							* scratch.time_fe_face_values_neighbor.shape_value(jj,1) *

 						scratch.space_fe_values.JxW(q)
 					;
 				}
 			}
 		}
		
		// update
		if ((n+1) < time.n_global_active_cells) {
			cell_time->get_dof_indices(
				scratch.time_local_dof_indices_neighbor
			);
			
			scratch.time_fe_face_values_neighbor.reinit(cell_time);
		}
	}
}


/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::FluidAssembly<dim> &copydata) {
	Assert(copydata.vi_ui_matrix.size(), dealii::ExcNotInitialized());
	Assert(
		(copydata.vi_ui_matrix.size()-1 == copydata.vi_ue_matrix.size()),
		dealii::ExcInvalidState()
	);
	Assert(
		copydata.vi_ui_matrix.size() == copydata.local_dof_indices.size(),
		dealii::ExcNotInitialized()
	);
	Assert(
		copydata.vi_ue_matrix.size() == copydata.local_dof_indices_neighbor.size(),
		dealii::ExcNotInitialized()
	);
	
	for (unsigned int n{0}; n < copydata.vi_ui_matrix.size(); ++n) {
		// volume ((w,u))_Q + trace (jump) (w^+,u^+)_Omega
		spacetime.constraints->distribute_local_to_global(
			copydata.vi_ui_matrix[n],
			copydata.local_dof_indices[n],
			copydata.local_dof_indices[n],
			*L
		);
	}
	
	for (unsigned int n{1}; n <= copydata.vi_ue_matrix.size(); ++n) {
		// trace (jump) - (w^+,u^-)_Omega
		spacetime.constraints->distribute_local_to_global(
			copydata.vi_ue_matrix[n-1],
			copydata.local_dof_indices[n],
			copydata.local_dof_indices_neighbor[n-1],
			*L
		);
	}
}

}}}

#include "ST_FluidAssembly.inst.in"
