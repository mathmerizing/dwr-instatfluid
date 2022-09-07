/**
 * @file Grid.tpl.hh
 * 
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-04-25, merge PU/high/low, JR
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-23, dyn. Stokes, UK
 * @date 2019-11-11, merge Biot-Allard/DWR/new ST, UK
 * 
 * @date 2019-08-27, merge into DWR, UK
 * @date 2019-01-28, space-time, UK
 * 
 * @date 2018-07-20, refine_slab_in_time, UK
 * @date 2018-03-07, included in dwr-heat, UK
 * @date 2018-03-05, work on the data structures, UK
 * @date 2017-08-01, Heat/DWR, UK
 * @date 2016-02-10, condiffrea, UK
 * @date 2016-01-14, condiff, UK
 * @date 2016-01-12, UK
 * @date 2015-11-11, UK
 * @date 2015-05-15, DTM++/AcousticWave Module, UK
 * @date (2012-07-26), 2013-08-15, ElasticWave, UK
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

#ifndef __fluid_Grid_tpl_hh
#define __fluid_Grid_tpl_hh

// PROJECT includes
#include <fluid/parameters/ParameterSet.hh>
#include <fluid/types/slabs.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/base/tensor_function.h>

namespace fluid {

template<int dim>
class Grid {
public:
	Grid(std::shared_ptr< fluid::ParameterSet > _parameter_set,
		MPI_Comm mpi_comm = MPI_COMM_WORLD) : mpi_comm(mpi_comm), coarse_tria() {
		Assert(_parameter_set.use_count(), dealii::ExcNotInitialized());
		parameter_set = _parameter_set;
	};
	
	virtual ~Grid();
	
	virtual void initialize_slabs();
	virtual void initialize_low_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	virtual void initialize_high_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	virtual void initialize_pu_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	

	virtual bool split_slab_in_time(
		typename fluid::types::spacetime::dwr::slabs<dim>::iterator slab
	);

	virtual void refine_slab_in_time(
		typename fluid::types::spacetime::dwr::slabs<dim>::iterator slab
	);
	
	virtual void generate();
	
	virtual void refine_global(
		const unsigned int &space_n = 1,
		const unsigned int &time_n = 1
	);
	
	virtual void set_manifolds();
	virtual void set_boundary_indicators();
	virtual void set_dirichlet_function(std::shared_ptr< dealii::TensorFunction<1,dim> > fun);
	
	virtual void distribute();
	virtual void distribute_low_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	virtual void distribute_high_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	virtual void distribute_pu_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);

	virtual void interpolate_dirichlet_bc(std::shared_ptr<dealii::DoFHandler<dim>> dof,
			                              std::shared_ptr<dealii::AffineConstraints<double>> constraints,
										  double tm,
										  bool inhom);

	virtual void create_sparsity_pattern_primal_on_slab(
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
			std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix>,
			std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix>
	);

	virtual void create_sparsity_pattern_dual_on_slab(
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
			std::shared_ptr< dealii::TrilinosWrappers::SparseMatrix>);
	virtual void clear_primal_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);
	virtual void clear_dual_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab);

	fluid::types::spacetime::dwr::slabs<dim> slabs;
	
	std::shared_ptr< dealii::ComponentMask > component_mask_convection;
	std::shared_ptr< dealii::ComponentMask > component_mask_pressure;
	
	dealii::GridIn<dim>            grid_in;
	dealii::GridOut                grid_out;
	
protected:
	std::shared_ptr< fluid::ParameterSet > parameter_set;
	MPI_Comm mpi_comm;
	std::shared_ptr<dealii::Triangulation<dim> > coarse_tria;
	std::shared_ptr< dealii::TensorFunction<1,dim> > dirichlet_function;
};

} // namespace

#endif
