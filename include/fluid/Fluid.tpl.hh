/**
 * @file Fluid.tpl.hh
 *
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-22, ST hanging nodes, UK
 * @date 2021-11-05, dynamics for stokes, JR, UK
 * @date 2021-10-30, force assembler, MPB, UK
 * @date 2019-11-07, space-time, quasi-stationary Stokes, UK
 */

/*  Copyright (C) 2012-2021 by Uwe Koecher and contributors                   */
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

#ifndef __Fluid_tpl_hh
#define __Fluid_tpl_hh

// PROJECT includes
#include <fluid/parameters/ParameterSet.hh>
#include <fluid/grid/Grid.tpl.hh>

#include <fluid/FluidDataPostprocessor.tpl.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>
#include <DTM++/base/Problem.hh>
#include <DTM++/io/DataOutput.tpl.hh>
#include <DTM++/types/storage_data_block_vectors.tpl.hh>
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/numerics/vector_tools.h>

// C++ includes
#include <memory>
#include <algorithm>
#include <list>
#include <iterator>
#include <string>

namespace fluid {

template<int dim>
class Fluid : public DTM::Problem {
public:
	Fluid() = default;
	virtual ~Fluid() = default;
	
	virtual void set_input_parameters(
		std::shared_ptr< dealii::ParameterHandler > parameter_handler
	);
	
	virtual void run();

protected:
	std::shared_ptr< fluid::ParameterSet > parameter_set;
	
	std::shared_ptr< fluid::Grid<dim> > grid;
	virtual void init_grid();
	
	struct {
		std::shared_ptr< dealii::Function<dim> > viscosity;
		
		struct {
			std::shared_ptr< dealii::TensorFunction<1,dim> > force;
		} fluid;
		
		struct {
			std::shared_ptr< dealii::TensorFunction<1,dim> > dirichlet;
			std::shared_ptr< dealii::TensorFunction<1,dim> > neumann;
// 			std::shared_ptr< dealii::TensorFunction<1,dim> > initial_value;
// 			std::shared_ptr< dealii::TensorFunction<1,dim> > exact_solution;
		} convection;
		
// 		struct {
// 			std::shared_ptr< dealii::Function<dim> > dirichlet;
// 			std::shared_ptr< dealii::Function<dim> > neumann;
// 			std::shared_ptr< dealii::Function<dim> > initial_value;
// 			std::shared_ptr< dealii::Function<dim> > exact_solution;
// 		} pressure;
	} function;
	
	virtual void init_functions();
	
	////////////////////////////////////////////////////////////////////////////
	// primal problem:
	//
	
	/// primal: data structures for forward time marching
	struct {
		// storage container
		struct {
			/// primal solution dof list
			std::shared_ptr< DTM::types::storage_data_vectors<1> > u;
		} storage;
		
		/// temporary storage for primal solution u at \f$ t_m \f$
		std::shared_ptr< dealii::BlockVector<double> > um;

		/// temporary storage for primal solution u at \f$ t_n \f$
		std::shared_ptr< dealii::BlockVector<double> > un;

		/// temporary storage for primal right hand side assembly
		std::shared_ptr< dealii::Vector<double> > Mum;
		
		// Matrix L, rhs vectors b and f
		std::shared_ptr< dealii::SparseMatrix<double> > L;
		std::shared_ptr< dealii::Vector<double> > b;
//		std::shared_ptr< dealii::Vector<double> > f;
		
		// Data Output
		std::shared_ptr< fluid::DataPostprocessor<dim> > data_postprocessor;
		std::shared_ptr< DTM::DataOutput<dim> > data_output;
		int    data_output_dwr_loop;
		double data_output_time_value;
		double data_output_trigger;
		bool   data_output_trigger_type_fixed;
	} primal;
	
	virtual void primal_reinit_storage();
	
	virtual void primal_assemble_system(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);
	
 	virtual void primal_assemble_rhs(
 		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
 	);
	
	virtual void primal_solve_slab_problem(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &x
	);
	
	virtual void primal_do_forward_TMS();
	
	// post-processing functions for data output
	virtual void primal_init_data_output();
	
	virtual void primal_do_data_output_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &x,
		const unsigned int dwr_loop
	);
	
	virtual void primal_do_data_output_on_slab_Qn_mode(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &x,
		const unsigned int dwr_loop
	);
	
	virtual void primal_do_data_output(
		const unsigned int dwr_loop,
		bool last
	);
	
	////////////////////////////////////////////////////////////////////////////
	// dual problem:
	//

	/// dual: data structures for forward time marching
	struct {
		// storage container
		struct {
			/// dual solution dof list
			std::shared_ptr< DTM::types::storage_data_vectors<1> > z;
		} storage;

		/// temporary storage for dual solution u at \f$ t_m \f$
		std::shared_ptr< dealii::BlockVector<double> > zm;

		/// temporary storage for dual solution u at \f$ t_n \f$
		std::shared_ptr< dealii::BlockVector<double> > zn;

		/// temporary storage for dual right hand side assembly
		std::shared_ptr< dealii::Vector<double> > Mzn;

		// Matrix L, rhs vectors b and f
		std::shared_ptr< dealii::SparseMatrix<double> > L;
		std::shared_ptr< dealii::Vector<double> > b;
		std::shared_ptr< dealii::Vector<double> > Je;

		// Data Output
		std::shared_ptr< fluid::DataPostprocessor<dim> > data_postprocessor;
		std::shared_ptr< DTM::DataOutput<dim> > data_output;
		int    data_output_dwr_loop;
		double data_output_time_value;
		double data_output_trigger;
		bool   data_output_trigger_type_fixed;
	} dual;

	virtual void dual_reinit_storage();

	virtual void dual_assemble_system(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual void dual_assemble_rhs(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual void dual_solve_slab_problem(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z
	);

	virtual void dual_do_backward_TMS();

	// post-processing functions for data output
	virtual void dual_init_data_output();

	virtual void dual_do_data_output_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z,
		const unsigned int dwr_loop
	);

	virtual void dual_do_data_output_on_slab_Qn_mode(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z,
		const unsigned int dwr_loop
	);

	virtual void dual_do_data_output(
		const unsigned int dwr_loop,
		bool last
	);

	////////////////////////////////////////////////////////////////////////////
	// functional values:
	//

	virtual void compute_functional_values(
			std::shared_ptr< dealii::BlockVector<double> > un,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual double compute_pressure(
			dealii::Point<dim> x,
			std::shared_ptr< dealii::BlockVector<double> > un,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual void compute_drag_lift_tensor(
			std::shared_ptr< dealii::BlockVector<double> > un,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	////////////////////////////////////////////////////////////////////////////
	// other
	//
	
	unsigned int setw_value_dwr_loops;
};

} // namespace

#endif