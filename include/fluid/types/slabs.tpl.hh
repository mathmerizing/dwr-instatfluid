/**
 * @file slabs.tpl.hh
 * @author Uwe Koecher (UK)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-07, stokes, UK
 * @date 2019-08-27, merge Q_n into DWR slabs, UK
 * @date 2018-03-06, UK
 */

/*  Copyright (C) 2012-2019 by Uwe Koecher                                    */
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


#ifndef __fluid_slabs_hh
#define __fluid_slabs_hh

// DEAL.II includes
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>

namespace fluid {
namespace types {
namespace spacetime {
namespace dwr {

/// slab: collects data structures and functions of a space-time slab for dwr
template <int dim>
struct s_slab {
	struct {
		/// deal.II Triangulation<dim> for \f$ \Omega_h \f$ on \f$ I_n \f$.
		std::shared_ptr< dealii::Triangulation<dim> > tria;
		
		// additional data for slab
		struct {
			std::shared_ptr< dealii::DoFHandler<dim> > dof;
			std::shared_ptr< dealii::FESystem<dim> > fe;
			std::shared_ptr< dealii::Mapping<dim> >  mapping;
			
			std::shared_ptr< dealii::IndexSet >
				locally_owned_dofs;
			std::shared_ptr< std::vector< dealii::IndexSet > >
				partitioning_locally_owned_dofs;
			std::shared_ptr< std::vector< dealii::types::global_dof_index > >
				block_sizes;
			
			std::shared_ptr< dealii::AffineConstraints<double> > constraints;
			std::shared_ptr< dealii::BlockSparsityPattern > sp_block_L;
			std::shared_ptr< dealii::SparsityPattern > sp_L;
		} primal;
		
 		struct {
 			std::shared_ptr< dealii::DoFHandler<dim> > dof;
 			std::shared_ptr< dealii::FESystem<dim> > fe;
 			std::shared_ptr< dealii::Mapping<dim> >  mapping;

 			std::shared_ptr< dealii::IndexSet > locally_owned_dofs;
 			std::shared_ptr< std::vector< dealii::IndexSet > >
 				partitioning_locally_owned_dofs;
 			std::shared_ptr< std::vector< dealii::types::global_dof_index > >
 				block_sizes;

 			std::shared_ptr< dealii::AffineConstraints<double> > constraints;
 			std::shared_ptr< dealii::BlockSparsityPattern > sp_block_L;
 			std::shared_ptr< dealii::SparsityPattern > sp_L;
 		} dual;
	} space;
	
	struct {
		std::shared_ptr< dealii::Triangulation<1> > tria;
		
		struct {
			std::shared_ptr< dealii::DoFHandler<1> > dof;
			std::shared_ptr< dealii::FiniteElement<1> > fe;
			std::shared_ptr< dealii::Mapping<1> > mapping;
		} primal;
		
 		struct {
 			std::shared_ptr< dealii::DoFHandler<1> > dof;
 			std::shared_ptr< dealii::FiniteElement<1> > fe;
 			std::shared_ptr< dealii::Mapping<1> > mapping;
 		} dual;
	} time;
	
	struct {
		struct {
			std::shared_ptr< dealii::IndexSet > locally_owned_dofs;
			std::shared_ptr< dealii::AffineConstraints<double> > constraints;
			std::shared_ptr< dealii::SparsityPattern > sp;
		} primal;
		
 		struct {
 			std::shared_ptr< dealii::IndexSet > locally_owned_dofs;
 			std::shared_ptr< dealii::AffineConstraints<double> > constraints;
 			std::shared_ptr< dealii::SparsityPattern > sp;
 		} dual;
	} spacetime;
	
	double t_m; ///< left endpoint of \f$ I_n=(t_m, t_n) \f$
	double t_n; ///< right endpoint of \f$ I_n=(t_m, t_n) \f$
	
	bool refine_in_time; // flag for marking: refinement in time of this slab
	bool coarsen_in_time; // flag for marking: coarsen in time of this slab
	
	// additional member functions
	
	/// get \f$ \tau_n = t_n - t_m \f$ of this slab
	double tau_n() const { return (t_n-t_m); };
	
	void set_refine_in_time_flag() { refine_in_time=true; };
	void clear_refine_in_time_flag() { refine_in_time=false; };
	
	void set_coarsen_in_time_flag() { coarsen_in_time=true; };
	void clear_coarsen_in_time_flag() { coarsen_in_time=false; };
};

template <int dim>
using slab = struct s_slab<dim>;

template <int dim>
using slabs = std::list< slab<dim> >;

}}}}

#endif
