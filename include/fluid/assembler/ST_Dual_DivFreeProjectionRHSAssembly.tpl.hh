/**
 * @file ST_Dual_DivFreeProjectionRHSAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @authro Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @date 2022-06-02, dual divergence free projection, JR
 * @date 2022-05-18, divergence free projection, JR
 * @date 2021-11-05, cleanups, UK
 * @date 2021-11-05, initialvalue for ST fluid, JR
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

#ifndef __ST_Dual_DivFreeProjectionRHSAssembly_tpl_hh
#define __ST_Dual_DivFreeProjectionRHSAssembly_tpl_hh

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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

// C++ includes
#include <memory>
#include <vector>

namespace projectionrhs {
namespace spacetime {
namespace dual {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
struct DualDivFreeProjectionRHSAssembly {
	DualDivFreeProjectionRHSAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim>       &mapping_space,
		const dealii::Quadrature<dim>    &quad_space
	);
	
	DualDivFreeProjectionRHSAssembly(const DualDivFreeProjectionRHSAssembly &scratch);
	
	// space
	dealii::FEValues<dim> space_fe_values;

	// convection
	std::vector< dealii::Tensor<2,dim> >          space_grad_phi;
	std::vector< dealii::Tensor<1,dim> >          space_phi;

	std::vector< dealii::types::global_dof_index > space_local_dof_indices;
	
	// other
	dealii::Tensor<1, dim> zn;
	dealii::Tensor<2, dim> grad_zn;
};

} // namespace Scratch
namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct DualDivFreeProjectionRHSAssembly{
	DualDivFreeProjectionRHSAssembly(
		const dealii::FiniteElement<dim> &fe_s
	);
	
	DualDivFreeProjectionRHSAssembly(const DualDivFreeProjectionRHSAssembly &copydata);
	
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
	
	void set_gradient_projection(bool use_gradient_projection);

	void assemble(
		std::shared_ptr< dealii::Vector<double> > zn,  // input
		std::shared_ptr< dealii::Vector<double> > Mzn, // output
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);
	
protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::DualDivFreeProjectionRHSAssembly<dim> &scratch,
		Assembly::CopyData::DualDivFreeProjectionRHSAssembly<dim> &copydata
	);
	
	void copy_local_to_global_cell(
		const Assembly::CopyData::DualDivFreeProjectionRHSAssembly<dim> &copydata
	);
	
private:
	////////////////////////////////////////////////////////////////////////////
	std::shared_ptr< dealii::Vector<double> > zn;
	std::shared_ptr< dealii::Vector<double> > Mzn;
	
	bool gradient_projection; // TRUE: use H^1_0 projection; FALSE: use L^2 projection

	struct {
		std::shared_ptr< dealii::DoFHandler<dim> > dof;
		std::shared_ptr< dealii::FiniteElement<dim> > fe;
		std::shared_ptr< dealii::Mapping<dim> > mapping;
		std::shared_ptr< dealii::AffineConstraints<double> > constraints;
	} space;
	
	// FEValuesExtractors for the Tensor<1,dim> convection field and the pressure
	dealii::FEValuesExtractors::Vector convection;
//	dealii::FEValuesExtractors::Scalar pressure;
};

}}}} // namespaces

#endif
