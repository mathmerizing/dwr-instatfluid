/**
 * @file ST_Dual_MeanDragAssembly.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhäuser (MPB)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-12-21, assemble mean drag goal functional, JR
 * @date 2020-01-23, space-time, UK
 * @date 2018-03-09, derived from L2_Je_global_L2L2_Assembly and MPB code, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-26, auto mode, UK
 * @date 2017-09-13, xwave/ewave, UK
 * @date 2015-05-19, AWAVE/C++.11, UK
 * @date 2012-08-31
 *
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

#ifndef __ST_Dual_MeanDragAssembly_tpl_hh
#define __ST_Dual_MeanDragAssembly_tpl_hh

// PROJECT includes
#include <fluid/types/slabs.tpl.hh>

// DEAL.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

// DTM++ includes
#include <DTM++/types/storage_data_vectors.tpl.hh>

// C++ includes
#include <iterator>
#include <functional>
#include <memory>
#include <vector>

namespace goal {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template<int dim>
struct Je_MeanDragAssembly {
	Je_MeanDragAssembly(
		// space
		const dealii::FiniteElement<dim> &fe_space,
		const dealii::Mapping<dim> &mapping_space,
//		const dealii::Quadrature<dim> &quad_space,
		const dealii::Quadrature<dim-1> &face_quad_space,
		// time
		const dealii::FiniteElement<1> &fe_time,
		const dealii::Mapping<1>	   &mapping_time,
		const dealii::Quadrature<1>    &quad_time,
		// other
		const double                   &t0,
		const double                   &T
	);
	
	Je_MeanDragAssembly(const Je_MeanDragAssembly &scratch);
	
	// space
//	dealii::FEValues<dim> space_fe_values;
	dealii::FEFaceValues<dim> space_fe_face_values;
	std::vector< dealii::types::global_dof_index > space_local_dof_indices;

	// convection
//	std::vector< dealii::Tensor<1,dim> > space_phi;
	std::vector< dealii::Tensor<2,dim> > space_grad_v;
	std::vector< double > space_p;

	// time
	dealii::FEValues<1> time_fe_values;
	std::vector< dealii::types::global_dof_index > time_local_dof_indices;

	// other
	double                t0;
	double                T;
	double viscosity;
};

}
namespace CopyData {

template<int dim>
struct Je_MeanDragAssembly {
	Je_MeanDragAssembly(
			const dealii::FiniteElement<dim> &fe_s,
			const dealii::FiniteElement<1> &fe_t,
			const dealii::types::global_dof_index &n_global_active_cells_t
	);

	Je_MeanDragAssembly(const Je_MeanDragAssembly &copydata);
	
	std::vector< dealii::Vector<double> > vi_Jei_vector;
	std::vector< std::vector<dealii::types::global_dof_index> > local_dof_indices;
};

}}
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
	
	void assemble(
		std::shared_ptr< dealii::Vector<double> > Je,
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const double &t0,
		const double &T
	);


protected:
	void local_assemble_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::Je_MeanDragAssembly<dim> &scratch,
		Assembly::CopyData::Je_MeanDragAssembly<dim> &copydata
	);
	
	void copy_local_to_global_cell(
		const Assembly::CopyData::Je_MeanDragAssembly<dim> &copydata
	);

private:
	std::shared_ptr< dealii::Vector<double> > _Je;

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
	} time;

 	struct {
 		std::shared_ptr< dealii::AffineConstraints<double> > constraints;
 	} spacetime;

	dealii::FEValuesExtractors::Vector convection;
	dealii::FEValuesExtractors::Scalar pressure;


	bool symmetric_stress = false;

	struct {
		std::shared_ptr< dealii::Function<dim> > viscosity;
	} function;
};

}}} // namespaces

#endif
