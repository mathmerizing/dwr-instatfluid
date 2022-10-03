/**
 * @file ST_Dual_MeanVorticityAssembly.tpl.cc
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 * 
 * @date 2022-09-27, mean vorticity assembly, JPT
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
#include <fluid/assembler/ST_Dual_MeanVorticityAssembly.tpl.hh>
#include <fluid/types/boundary_id.hh> // for Vorticity boundary id

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
namespace mean{
namespace vorticity{
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
Je_MeanVorticityAssembly<dim>::Je_MeanVorticityAssembly(
	const dealii::DoFHandler<dim>    &dof_dual,
	const dealii::DoFHandler<dim>    &dof_primal,
	// space
	const dealii::FiniteElement<dim> &fe_space,
	const dealii::Mapping<dim> &mapping_space,
	const dealii::Quadrature<dim> &quad_space,
	// time
	const dealii::FiniteElement<1> &fe_time,
	const dealii::Mapping<1>	   &mapping_time,
	const dealii::Quadrature<1>    &quad_time,
	// primal space
	const dealii::FiniteElement<dim> &primal_fe_space,
	const dealii::Mapping<dim>       &primal_mapping_space,
	// primal time
	const dealii::FiniteElement<1> &primal_fe_time,
	const dealii::Mapping<1>       &primal_mapping_time,
	// other
	const double                   &t0,
	const double                   &T) :
	// init space
	dof_dual(dof_dual),
	dof_primal(dof_primal),
	space_fe_values(
		mapping_space,
		fe_space,
		quad_space,
		dealii::update_values |
		dealii::update_JxW_values |
		dealii::update_gradients |
		dealii::update_quadrature_points
	),
	primal_space_fe_values(
		primal_mapping_space,
		primal_fe_space,
		quad_space,
		dealii::update_values |
		dealii::update_gradients |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	),
	space_local_dof_indices(fe_space.dofs_per_cell),
	primal_space_local_dof_indices(primal_fe_space.dofs_per_cell),
	primal_space_grad_v(),
	primal_space_grad_phi(primal_fe_space.dofs_per_cell),
	space_grad_phi(fe_space.dofs_per_cell),
	curl_u(0),
	curl_phi(0),
	// init time
	time_fe_values(
		mapping_time,
		fe_time,
		quad_time,
		dealii::update_values |
		dealii::update_JxW_values |
		dealii::update_quadrature_points
	),
	primal_time_fe_values(
		primal_mapping_time,
		primal_fe_time,
		quad_time,
		dealii::update_values |
		dealii::update_quadrature_points
	),
	time_local_dof_indices(fe_time.dofs_per_cell),
	t0(t0),
	T(T){
}

template<int dim>
Je_MeanVorticityAssembly<dim>::Je_MeanVorticityAssembly(const Je_MeanVorticityAssembly &scratch) :

	dof_dual(scratch.dof_dual),
	dof_primal(scratch.dof_primal),
	space_fe_values(
		scratch.space_fe_values.get_mapping(),
		scratch.space_fe_values.get_fe(),
		scratch.space_fe_values.get_quadrature(),
		scratch.space_fe_values.get_update_flags()
	),primal_space_fe_values(
			scratch.primal_space_fe_values.get_mapping(),
			scratch.primal_space_fe_values.get_fe(),
			scratch.primal_space_fe_values.get_quadrature(),
			scratch.primal_space_fe_values.get_update_flags()
		),
	space_local_dof_indices(scratch.space_local_dof_indices),
	primal_space_local_dof_indices(scratch.primal_space_local_dof_indices),
	primal_space_grad_v(scratch.primal_space_grad_v),
	primal_space_grad_phi(scratch.primal_space_grad_phi),
	space_grad_phi(scratch.space_grad_phi),
	curl_u(scratch.curl_u),
	curl_phi(scratch.curl_phi),
	//
	time_fe_values(
		scratch.time_fe_values.get_mapping(),
		scratch.time_fe_values.get_fe(),
		scratch.time_fe_values.get_quadrature(),
		scratch.time_fe_values.get_update_flags()
	),
	primal_time_fe_values(
		scratch.primal_time_fe_values.get_mapping(),
		scratch.primal_time_fe_values.get_fe(),
		scratch.primal_time_fe_values.get_quadrature(),
		scratch.primal_time_fe_values.get_update_flags()
	),
	time_local_dof_indices(scratch.time_local_dof_indices),
	t0(scratch.t0),
	T(scratch.T)
{
}

}

namespace CopyData {

template<int dim>
Je_MeanVorticityAssembly<dim>::Je_MeanVorticityAssembly(
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
Je_MeanVorticityAssembly<dim>::Je_MeanVorticityAssembly(const Je_MeanVorticityAssembly &copydata) :
	vi_Jei_vector(copydata.vi_Jei_vector),
	local_dof_indices(copydata.local_dof_indices) {
}

}}

template<int dim>
void
Assembler<dim>::
assemble(
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > Je,
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > u,
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
	
	Assert(slab->spacetime.dual.constraints.use_count(), dealii::ExcNotInitialized());

	Assert(Je.use_count(), dealii::ExcNotInitialized());
	Assert(u.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////////
	// init
	
	*Je = 0.;

	_Je = Je;
	_u = u;

	space.dof = slab->space.dual.fe_info->dof;
	space.fe = slab->space.dual.fe_info->fe;
	space.mapping = slab->space.dual.fe_info->mapping;
	space.constraints = slab->space.dual.fe_info->constraints;
	
	primal.space.dof = slab->space.primal.fe_info->dof;
	primal.space.fe = slab->space.primal.fe_info->fe;
	primal.space.mapping = slab->space.primal.fe_info->mapping;

	time.dof = slab->time.dual.fe_info->dof;
	time.fe = slab->time.dual.fe_info->fe;
	time.mapping = slab->time.dual.fe_info->mapping;
	
	primal.time.dof = slab->time.primal.fe_info->dof;
	primal.time.fe = slab->time.primal.fe_info->fe;
	primal.time.mapping = slab->time.primal.fe_info->mapping;

	spacetime.constraints = slab->spacetime.dual.constraints;

	// FEValuesExtractors
	convection = 0;
	
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
		Assembly::Scratch::Je_MeanVorticityAssembly<dim> (
			*slab->space.dual.fe_info->dof,
			*slab->space.primal.fe_info->dof,
			*space.fe,
			*space.mapping,
			quad_space,
			*time.fe,
			*time.mapping,
			quad_time,
			*primal.space.fe,
			*primal.space.mapping,
			*primal.time.fe,
			*primal.time.mapping,
			t0,
			T
		),
		Assembly::CopyData::Je_MeanVorticityAssembly<dim> (
			*space.fe,
			*time.fe,
			time.n_global_active_cells
		)
	);

	Je->compress(dealii::VectorOperation::add);
}


/// Local assemble on cell.
template<int dim>
void Assembler<dim>::local_assemble_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::Je_MeanVorticityAssembly<dim> &scratch,
	Assembly::CopyData::Je_MeanVorticityAssembly<dim> &copydata) {


	typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(&cell->get_triangulation(),
																   cell->level(),
																   cell->index(),
																   &scratch.dof_dual);

	typename dealii::DoFHandler<dim>::active_cell_iterator cell_primal(&cell->get_triangulation(),
																	cell->level(),
																	cell->index(),
																   &scratch.dof_primal);



	scratch.space_fe_values.reinit(cell_dual);
	scratch.primal_space_fe_values.reinit(cell_primal);
	
	cell_dual->get_dof_indices(scratch.space_local_dof_indices);
	cell_primal->get_dof_indices(scratch.primal_space_local_dof_indices);

	auto cell_time = time.dof->begin_active();
	auto primal_cell_time = primal.time.dof->begin_active();
	auto endc_time = time.dof->end();
	
	// average the integral over time
	const double time_scaling = 1. / (scratch.T - scratch.t0);

	unsigned int n;
	for ( ; cell_time != endc_time; ++cell_time, ++cell_primal) {
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

		// assemble:volume
		scratch.time_fe_values.reinit(cell_time);
		scratch.primal_time_fe_values.reinit(primal_cell_time);

		for (unsigned int qt{0}; qt < scratch.time_fe_values.n_quadrature_points; ++qt) {

			for ( unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points; ++q) {

				for ( unsigned int k{0} ; k < space.fe->dofs_per_cell ; ++k) {
					scratch.space_grad_phi[k] =
							scratch.space_fe_values[convection].gradient(k,q);

 				}for ( unsigned int k{0} ; k <primal. space.fe->dofs_per_cell ; ++k) {
					scratch.primal_space_grad_phi[k] =
							scratch.primal_space_fe_values[convection].gradient(k,q);
 				}
				scratch.primal_space_grad_v = 0;

				for (unsigned int ii{0} ; ii < primal.time.fe->dofs_per_cell; ++ii)
					for (unsigned int i{0} ; i < primal.space.fe->dofs_per_cell ; ++i){
						// correct ST solution vector entry
						double u_i_ii = (*_u)[
							scratch.primal_space_local_dof_indices[i]
								// time offset
								+ primal.space.dof->n_dofs() *
								   (n * primal.time.fe->dofs_per_cell)
								// local in time dof
								+ primal.space.dof->n_dofs() * ii
								]*scratch.primal_time_fe_values.shape_value(ii,qt);;

						// grad v
						scratch.primal_space_grad_v += u_i_ii * scratch.primal_space_grad_phi[i];
				}

				scratch.curl_u = (scratch.primal_space_grad_v[1][0])-(scratch.primal_space_grad_v[0][1]);
				for (unsigned int ii{0} ; ii < time.fe->dofs_per_cell; ++ii)
				for (unsigned int i{0} ; i < space.fe->dofs_per_cell ; ++i){
					scratch.curl_phi = (scratch.space_grad_phi[i][1][0]-scratch.space_grad_phi[i][0][1]);

					copydata.vi_Jei_vector[n](
							i + ii*space.fe->dofs_per_cell)+=
									time_scaling *2.*scratch.curl_u*scratch.curl_phi *
									scratch.space_fe_values.JxW(q)
										* scratch.time_fe_values.JxW(qt);
				}

			}
		} // t_q
	}
}

/// Copy local assembly to global matrix.
template<int dim>
void Assembler<dim>::copy_local_to_global_cell(
	const Assembly::CopyData::Je_MeanVorticityAssembly<dim> &copydata) {
	Assert(copydata.vi_Jei_vector.size(), dealii::ExcNotInitialized());
	Assert(
		(copydata.vi_Jei_vector.size() == copydata.local_dof_indices.size()),
		dealii::ExcNotInitialized()
	);
	
	for (unsigned int n{0}; n < copydata.vi_Jei_vector.size(); ++n) {
		spacetime.constraints->distribute_local_to_global(
			copydata.vi_Jei_vector[n],
			copydata.local_dof_indices[n],
			*_Je
		);
	}
}

}}}}}

#include "ST_Dual_MeanVorticityAssembly.inst.in"
