/*
 * SlabDoFTools.hh
 *
 *  Created on: Aug 9, 2022
 *      Author: thiele
 */

#ifndef INCLUDE_IDEAL_II_SLABDOFTOOLS_HH_
#define INCLUDE_IDEAL_II_SLABDOFTOOLS_HH_


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>

namespace idealii
{
	namespace SlabDoFTools{
		enum STCoupling{
			Upwinding = 0,
			Downwinding = 1,
			Continuous
		};

		template<int dim, int spacedim>
		dealii::IndexSet
		extract_locally_owned_dofs(
				std::shared_ptr<dealii::DoFHandler<dim,spacedim>> space_dof,
				std::shared_ptr<dealii::DoFHandler<1,1>>   time_dof
		);

		template<int dim, int spacedim>
		dealii::IndexSet
		extract_locally_relevant_dofs(
				std::shared_ptr<dealii::DoFHandler<dim,spacedim>> space_dof,
				std::shared_ptr<dealii::DoFHandler<1,1>>   time_dof
		);


		template<typename T>
		void
		make_spacetime_constraints(
			std::shared_ptr<dealii::IndexSet> space_relevant_dofs,
			std::shared_ptr<dealii::AffineConstraints<T>> space_constraints,
			dealii::types::global_dof_index n_space_dofs,
			dealii::types::global_dof_index  n_time_dofs,
			std::shared_ptr<dealii::IndexSet> spacetime_relevant_dofs,
			std::shared_ptr<dealii::AffineConstraints<T>> spacetime_constraints
		);


	}
}
#endif /* INCLUDE_IDEAL_II_SLABDOFTOOLS_HH_ */
