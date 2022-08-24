/**
 * @file ST_FluidAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @date 2021-11-09, rm block, UK
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

/*  Copyright (C) 2012-2022 by Uwe Koecher and contriubtors                   */
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

#ifndef __ST_FluidAssembly_tpl_hh
#define __ST_FluidAssembly_tpl_hh

// PROJECT includes
#include <fluid/types/slabs.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

// C++ includes
#include <memory>
#include <vector>

namespace fluid {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
struct FluidAssembly {
	FluidAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
		const dealii::Quadrature<dim>    &quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>       &mapping_time,
		const dealii::Quadrature<1>    &quad_time,
		const dealii::Quadrature<1>    &face_nodes
	);
	
	FluidAssembly(const FluidAssembly &scratch);
	
	// space
	dealii::FEValues<dim> space_fe_values;
	
	// convection
	std::vector< dealii::Tensor<1,dim> >          space_phi;
	std::vector< dealii::SymmetricTensor<2,dim> > space_symgrad_phi;
	std::vector< dealii::Tensor<2,dim> >          space_grad_phi;
	std::vector<double>                           space_div_phi;
	
	// pressure
	std::vector<double>                           space_psi;
	
	unsigned int space_dofs_per_cell;
	double space_JxW;
	std::vector< dealii::types::global_dof_index > space_local_dof_indices;
	
	// time
	dealii::FEValues<1> time_fe_values;
	dealii::FEValues<1> time_fe_quad_values;
	dealii::FEValues<1> time_fe_face_values;
	dealii::FEValues<1> time_fe_face_values_neighbor;
	
	std::vector<double>               time_zeta;
	std::vector<dealii::Tensor<1,1> > time_grad_zeta;
	
	unsigned int time_dofs_per_cell;
	double time_JxW;
	std::vector< dealii::types::global_dof_index > time_local_dof_indices;
	std::vector< dealii::types::global_dof_index > time_local_dof_indices_neighbor;
	
	// other
	// solution evals
    dealii::Tensor<1,dim> v;
	dealii::Tensor<2,dim> grad_v;

	double viscosity;
};

} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct FluidAssembly{
	FluidAssembly(
		const dealii::FiniteElement<dim> &fe_s,
		const dealii::FiniteElement<1> &fe_t,
		const dealii::types::global_dof_index &n_global_active_cells_t
	);
	
	FluidAssembly(const FluidAssembly &copydata);
	
	std::vector< dealii::FullMatrix<double> > vi_ui_matrix;
	std::vector< dealii::FullMatrix<double> > vi_ue_matrix;
	std::vector< std::vector<dealii::types::global_dof_index> > local_dof_indices;
	std::vector< std::vector<dealii::types::global_dof_index> > local_dof_indices_neighbor;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


template<int dim>
class Assembler {
public:
	Assembler() = default;
	~Assembler() = default;
	
	void set_functions(
		std::shared_ptr< dealii::Function<dim> > viscosity
	);
	
	void set_symmetric_stress(bool use_symmetric_stress);

	void set_time_quad_type(std::string quad_type);

	void assemble(
		std::shared_ptr< dealii::SparseMatrix<double> > L,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	    std::shared_ptr< dealii::Vector<double> > _u,
		bool _nonlin = false
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::FluidAssembly<dim> &scratch,
		Assembly::CopyData::FluidAssembly<dim> &copydata
	);

	void copy_local_to_global_cell(
		const Assembly::CopyData::FluidAssembly<dim> &copydata
	);
	
private:
	std::shared_ptr< dealii::SparseMatrix<double> > L;
	
	bool nonlin;

	bool symmetric_stress;

	struct {
		std::shared_ptr< dealii::Function<dim> > viscosity;
	} function;
	
	struct {
		std::shared_ptr< dealii::DoFHandler<dim> > dof;
		std::shared_ptr< dealii::FESystem<dim> > fe;
		std::shared_ptr< dealii::Mapping<dim> > mapping;
		std::shared_ptr< dealii::AffineConstraints<double> > constraints;
	} space;
	
	struct {
		dealii::types::global_dof_index n_global_active_cells;
		
		std::shared_ptr< dealii::DoFHandler<1> > dof;
		std::shared_ptr< dealii::FiniteElement<1> > fe;
		std::shared_ptr< dealii::Mapping<1> > mapping;
		std::string quad_type;
	} time;
	
 	struct {
 		std::shared_ptr< dealii::AffineConstraints<double> > constraints;
 	} spacetime;
	
	dealii::FEValuesExtractors::Vector convection;
	dealii::FEValuesExtractors::Scalar pressure;
    std::shared_ptr< dealii::Vector<double> > u;
};

}}} // namespaces

#endif
