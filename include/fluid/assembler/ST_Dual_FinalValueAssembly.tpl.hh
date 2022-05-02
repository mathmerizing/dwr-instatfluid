/**
 * @file ST_Dual_FinalValueAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-12-20, finalvalue for ST Stokes, JR
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
 *
 * Implements the assembly of the dual Navier-Stokes solution from one slab to the next one.
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

#ifndef __ST_Dual_FinalValueAssembly_tpl_hh
#define __ST_Dual_FinalValueAssembly_tpl_hh

// PROJECT includes
#include <fluid/types/slabs.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

// C++ includes
#include <memory>
#include <vector>

namespace finalvalue {
namespace spacetime {
namespace dual {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
struct FinalValueAssembly {
	FinalValueAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
		const dealii::Quadrature<dim>    &quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>       &mapping_time,
		const dealii::Quadrature<1>    &face_nodes
	);
	
	FinalValueAssembly(const FinalValueAssembly &scratch);
	
	// space
	dealii::FEValues<dim> space_fe_values;
	std::vector< dealii::types::global_dof_index > space_local_dof_indices;
	
	// time
	dealii::FEValues<1> time_fe_face_values;
	std::vector< dealii::types::global_dof_index > time_local_dof_indices;
	
	// other
	dealii::Tensor<1, dim> zn;
};

} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct FinalValueAssembly{
	FinalValueAssembly(
		const dealii::FiniteElement<dim> &fe_s,
		const dealii::FiniteElement<1> &fe_t
	);
	
	FinalValueAssembly(const FinalValueAssembly &copydata);
	
	dealii::Vector<double> vi_zn_vector;
	std::vector<dealii::types::global_dof_index> local_dof_indices;
};

} // namespace CopyData
} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


template<int dim>
class Assembler {
public:
	Assembler() = default;
	~Assembler() = default;
	
	void assemble(
		std::shared_ptr< dealii::Vector<double> > zn,  // input
		std::shared_ptr< dealii::Vector<double> > Mzn, // output
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::FinalValueAssembly<dim> &scratch,
		Assembly::CopyData::FinalValueAssembly<dim> &copydata
	);
	
	void copy_local_to_global_cell(
		const Assembly::CopyData::FinalValueAssembly<dim> &copydata
	);
	
private:
	////////////////////////////////////////////////////////////////////////////
	std::shared_ptr< dealii::Vector<double> > zn;
	std::shared_ptr< dealii::Vector<double> > Mzn;
	
	struct {
		std::shared_ptr< dealii::DoFHandler<dim> > dof;
		std::shared_ptr< dealii::FiniteElement<dim> > fe;
		std::shared_ptr< dealii::Mapping<dim> > mapping;
	} space;
	
	struct {
		std::shared_ptr< dealii::DoFHandler<1> > dof;
		std::shared_ptr< dealii::FiniteElement<1> > fe;
		std::shared_ptr< dealii::Mapping<1> > mapping;
	} time;
	
	struct {
		std::shared_ptr< dealii::AffineConstraints<double> > constraints;
	} spacetime;
	
	// FEValuesExtractors for the Tensor<1,dim> convection field and the pressure
	dealii::FEValuesExtractors::Vector convection;
//	dealii::FEValuesExtractors::Scalar pressure;
};

}}}} // namespaces

#endif
