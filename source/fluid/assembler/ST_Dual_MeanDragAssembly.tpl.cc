/**
 * @file ST_Dual_MeanDragAssembly.tpl.cc
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-12-21, mean drag assembly, JR
 * @date 2021-10-28, space-time force stokes, MPB, (UK)
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
#include <fluid/assembler/ST_Dual_MeanDragAssembly.tpl.hh>
#include <fluid/types/boundary_id.hh> // for Drag boundary id

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>

// C++ includes
#include <functional>

namespace goal {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
Je_MeanDragAssembly<dim>::Je_MeanDragAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
//		const dealii::Quadrature<dim>    &quad_space,
		const dealii::Quadrature<dim-1>  &face_quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>       &mapping_time,
		const dealii::Quadrature<1>    &quad_time,
		// other
		const double                   &t0,
		const double                   &T) :
	// init space
//	space_fe_values(
//		mapping_space,
//		fe_space,
//		quad_space,
//		dealii::update_values |
//		dealii::update_JxW_values |
//		dealii::update_quadrature_points
//	),
	space_fe_face_values(
			mapping_space,
			fe_space,
			face_quad_space,
			dealii::update_values |
			dealii::update_gradients |
			dealii::update_normal_vectors |
			dealii::update_JxW_values |
			dealii::update_quadrature_points
		),
	space_local_dof_indices(fe_space.dofs_per_cell),
//	space_phi(fe_space.dofs_per_cell),
	space_grad_v(fe_space.dofs_per_cell),
	space_p(fe_space.dofs_per_cell),
	// init time
	time_fe_values(
		mapping_time,
		fe_time,
		quad_time,
		dealii::update_values |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	),
	time_local_dof_indices(fe_time.dofs_per_cell),
	t0(t0),
	T(T){
}

template<int dim>
Je_MeanDragAssembly<dim>::Je_MeanDragAssembly(const Je_MeanDragAssembly &scratch) :
//	space_fe_values(
//		scratch.space_fe_values.get_mapping(),
//		scratch.space_fe_values.get_fe(),
//		scratch.space_fe_values.get_quadrature(),
//		scratch.space_fe_values.get_update_flags()
//	),
	space_fe_face_values(
		scratch.space_fe_face_values.get_mapping(),
		scratch.space_fe_face_values.get_fe(),
		scratch.space_fe_face_values.get_quadrature(),
		scratch.space_fe_face_values.get_update_flags()
	),
	space_local_dof_indices(scratch.space_local_dof_indices),
//	space_phi(scratch.space_phi),
	space_grad_v(scratch.space_grad_v),
	space_p(scratch.space_p),
	//
	time_fe_values(
		scratch.time_fe_values.get_mapping(),
		scratch.time_fe_values.get_fe(),
		scratch.time_fe_values.get_quadrature(),
		scratch.time_fe_values.get_update_flags()
	),
	time_local_dof_indices(scratch.time_local_dof_indices),
	t0(scratch.t0),
	T(scratch.T)
{
}

}

namespace CopyData {

template<int dim>
Je_MeanDragAssembly<dim>::Je_MeanDragAssembly(
		const dealii::FiniteElement<dim> &fe_s,
		const dealii::FiniteElement<1> &fe_t,
		const dealii::types::global_dof_index &n_global_active_cells_t) :
	vi_Jei_vector(
		n_global_active_cells_t, // n_cells time
		dealii::Vector<double> (
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	),
	local_dof_indices(
		n_global_active_cells_t, // n_cells time
		std::vector<dealii::types::global_dof_index>(
			fe_s.dofs_per_cell * fe_t.dofs_per_cell
		)
	) {
}

template<int dim>
Je_MeanDragAssembly<dim>::Je_MeanDragAssembly(const Je_MeanDragAssembly &copydata) :
	vi_Jei_vector(copydata.vi_Jei_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}

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
set_symmetric_stress(bool use_symmetric_stress)
{
	symmetric_stress = use_symmetric_stress;
}

template<int dim>
void
Assembler<dim>::
assemble(
	std::shared_ptr< dealii::Vector<double> > Je,
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const double &t0,
	const double &T) {
	////////////////////////////////////////////////////////////////////////////
	// check
	Assert(dim==2 || dim==3, dealii::ExcNotImplemented());

	Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.dual.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.dual.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
	Assert(slab->space.dual.fe_info->constraints.use_count(), dealii::ExcNotInitialized());
	
	Assert(slab->time.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.dual.fe_info->fe.use_count(), dealii::ExcNotInitialized());
	Assert(slab->time.dual.fe_info->mapping.use_count(), dealii::ExcNotInitialized());
	
	Assert(Je.use_count(), dealii::ExcNotInitialized());
	Assert(Je->size(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////////
	// init
	
	*Je = 0.;

	_Je = std::make_shared< dealii::Vector<double> > ();
	_Je->reinit(
		slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()
	);

	space.dof = slab->space.dual.fe_info->dof;
	space.fe = slab->space.dual.fe_info->fe;
	space.mapping = slab->space.dual.fe_info->mapping;
	space.constraints = slab->space.dual.fe_info->constraints;
	
	time.dof = slab->time.dual.fe_info->dof;
	time.fe = slab->time.dual.fe_info->fe;
	time.mapping = slab->time.dual.fe_info->mapping;
	
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
	//
	
	const dealii::QGauss<dim - 1> face_quad_space(
		std::max(
			std::max(
				space.fe->base_element(0).base_element(0).tensor_degree(),
				space.fe->base_element(0).base_element(1).tensor_degree()
			),
			static_cast<unsigned int> (1)
		) + 4
	);

	const dealii::QGauss<1> quad_time(
		time.fe->tensor_degree()+1
	);
	
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
		Assembly::Scratch::Je_MeanDragAssembly<dim> (
			*space.fe,
			*space.mapping,
//			quad_space,
			face_quad_space,
			*time.fe,
			*time.mapping,
			quad_time,
			t0,
			T
		),
		Assembly::CopyData::Je_MeanDragAssembly<dim> (
			*space.fe,
			*time.fe,
			time.n_global_active_cells
		)
	);

	Je->add(1., *_Je);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::Je_MeanDragAssembly<dim> &scratch,
	Assembly::CopyData::Je_MeanDragAssembly<dim> &copydata) {
	cell->get_dof_indices(scratch.space_local_dof_indices);
	
	auto cell_time = time.dof->begin_active();
	auto endc_time = time.dof->end();
	
	// average the integral over time
	const double time_scaling = 1. / (scratch.T - scratch.t0);

// 	for (unsigned int n{0}; n < time.n_global_active_cells; ++n)
	unsigned int n;
	for ( ; cell_time != endc_time; ++cell_time) {
		n=cell_time->index();
		copydata.vi_Jei_vector[n] = 0;
		
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
		
		// assemble: volume
		scratch.time_fe_values.reinit(cell_time);
		for (unsigned int qt{0}; qt < scratch.time_fe_values.n_quadrature_points; ++qt) {
			function.viscosity->set_time(scratch.time_fe_values.quadrature_point(qt)[0]);
			
			for (unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
				if (cell->face(face)->at_boundary() &&
					cell->face(face)->boundary_id() == fluid::types::space::boundary_id::prescribed_obstacle)
				{
					scratch.space_fe_face_values.reinit(cell, face);
					for (unsigned int q{0}; q < scratch.space_fe_face_values.n_quadrature_points; ++q) {
						scratch.viscosity = function.viscosity->value(
							scratch.space_fe_face_values.quadrature_point(q),0
						);

						for (unsigned int k{0}; k < space.fe->dofs_per_cell; ++k) {
							scratch.space_grad_v[k] =
								scratch.space_fe_face_values[convection].gradient(k,q);
							scratch.space_p[k] =
								scratch.space_fe_face_values[pressure].value(k,q);
						}

						dealii::Tensor<2, dim> sigma_fluid;
						dealii::Tensor<2, dim> pI;

						for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
						for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
							// reset all values to zero
							sigma_fluid.clear();
							pI.clear();

							for (unsigned int k{0}; k < dim; ++k)
								pI[k][k] = scratch.space_p[i];

							sigma_fluid = -pI + scratch.viscosity * scratch.space_grad_v[i];
							if (symmetric_stress)
							{
								sigma_fluid += scratch.viscosity * transpose(scratch.space_grad_v[i]);
							}
							const dealii::Tensor<1, dim> drag_lift_value = -1.0 * sigma_fluid * scratch.space_fe_face_values.normal_vector(q);

							copydata.vi_Jei_vector[n](
								i + ii*space.fe->dofs_per_cell) +=
								time_scaling *
								20. * // valid for 2D-2 and 2D-3
								(
									drag_lift_value[0]
										* scratch.time_fe_values.shape_value(ii,qt) *

									scratch.space_fe_face_values.JxW(q)
										* scratch.time_fe_values.JxW(qt)
								)
							;
						}
					}
				}
			}
		} // t_q
	}
}

/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::Je_MeanDragAssembly<dim> &copydata) {
	Assert(copydata.vi_Jei_vector.size(), dealii::ExcNotInitialized());
	Assert(
		(copydata.vi_Jei_vector.size() == copydata.local_dof_indices.size()),
		dealii::ExcNotInitialized()
	);
	
	for (unsigned int n{0}; n < copydata.vi_Jei_vector.size(); ++n) {
		space.constraints->distribute_local_to_global(
			copydata.vi_Jei_vector[n],
			copydata.local_dof_indices[n],
			*_Je
		);
	}
}

}}}

#include "ST_Dual_MeanDragAssembly.inst.in"
