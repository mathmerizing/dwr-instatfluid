/*
 * SlabDoFTools.cc
 *
 *  Created on: Aug 9, 2022
 *      Author: thiele
 */

#include <ideal.II/dofs/SlabDoFTools.hh>

#include <deal.II/dofs/dof_tools.h>

namespace idealii
{
	namespace SlabDoFTools{

		//ToDo: move to SlabDoFHandler class or where deal.II does this
		template<int dim, int spacedim>
		dealii::IndexSet
		extract_locally_owned_dofs(
			std::shared_ptr<dealii::DoFHandler<dim,spacedim>> space_dof,
			std::shared_ptr<dealii::DoFHandler<1,1>>   time_dof
		){
			dealii::IndexSet space_owned_dofs
				= space_dof->locally_owned_dofs();


			dealii::IndexSet spacetime_owned_dofs(
					space_owned_dofs.size() *
					time_dof->n_dofs()
			);

			for (dealii::types::global_dof_index time_dof_index{0};
				 time_dof_index < time_dof->n_dofs();
				 time_dof_index++
			){
				spacetime_owned_dofs.add_indices(
					space_owned_dofs,
					time_dof_index * space_dof->n_dofs() //offset
				);
			}

			return spacetime_owned_dofs;
		}

		//ToDo: make SlabDoFHandler class and pass that
		template<int dim, int spacedim>
		dealii::IndexSet
		extract_locally_relevant_dofs(
			std::shared_ptr<dealii::DoFHandler<dim,spacedim>> space_dof,
			std::shared_ptr<dealii::DoFHandler<1,1>>   time_dof
		){
			dealii::IndexSet space_relevant_dofs;
			dealii::DoFTools::extract_locally_relevant_dofs(
				*space_dof,
				space_relevant_dofs
			);


			dealii::IndexSet spacetime_relevant_dofs(
					space_relevant_dofs.size() *
					time_dof->n_dofs()
			);

			for (dealii::types::global_dof_index time_dof_index{0};
				 time_dof_index < time_dof->n_dofs();
				 time_dof_index++
			){
				spacetime_relevant_dofs.add_indices(
					space_relevant_dofs,
					time_dof_index * space_dof->n_dofs() //offset
				);
			}

			return spacetime_relevant_dofs;
		}


		template<typename T>
		void
		make_spacetime_constraints(
			std::shared_ptr<dealii::IndexSet> space_relevant_dofs,
			std::shared_ptr<dealii::AffineConstraints<T>> space_constraints,
			dealii::types::global_dof_index n_space_dofs,
			dealii::types::global_dof_index  n_time_dofs,
			std::shared_ptr<dealii::IndexSet> spacetime_relevant_dofs,
			std::shared_ptr<dealii::AffineConstraints<T>> spacetime_constraints
		){
			Assert(space_constraints.use_count(),dealii::ExcNotInitialized());
			Assert(spacetime_constraints.use_count(),dealii::ExcNotInitialized());

			spacetime_constraints->clear();
			spacetime_constraints->reinit(*spacetime_relevant_dofs);

			//go over locally relevant dofs
			for ( auto id = space_relevant_dofs->begin() ;
				  id != space_relevant_dofs->end();
				  id++ )
			{
				// check if this is a constrained dofs
				if( space_constraints->is_constrained(*id) ){

					//get entries for constrained dof
					const std::vector< std::pair< dealii::types::global_dof_index, double > > *
					entries = space_constraints->get_constraint_entries(*id);

					//go over temporal dofs
					for (dealii::types::global_dof_index d{0};
							d < n_time_dofs; ++d )
					{
						//A line has to be added either way
						spacetime_constraints->add_line(*id + d * n_space_dofs );

						//For Dirichlet constraints entries is empty and we are done
						//Otherwise we need to add the shifted entries
						if ( entries->size() > 0){
							for (auto entry: *entries){
								spacetime_constraints->add_entry(
									*id + d * n_space_dofs,
									entry.first + d * n_space_dofs,
									entry.second
								);
							}
						}
					}
				}
			}
		}
	}
}

#include "SlabDoFTools.inst.in"
