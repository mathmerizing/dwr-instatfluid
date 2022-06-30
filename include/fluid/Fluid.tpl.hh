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
#include <fluid/ErrorEstimator/ErrorEstimator.tpl.hh>

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
#include <deal.II/lac/sparse_direct.h>

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

	struct{
		unsigned int max_steps;
		double 		 lower_bound;
		double       rebuild;
		unsigned int line_search_steps;
		double       line_search_damping;
	} newton;

	
	virtual void init_newton_parameters();

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
		std::shared_ptr< dealii::Vector<double> > um;

		/// temporary storage for primal solution u at \f$ t_n \f$
		std::shared_ptr< dealii::Vector<double> > un;

		/// temporary storage for primal right hand side assembly
		std::shared_ptr< dealii::Vector<double> > Mum;
		
		/// temporary storage for newton vectors
		std::shared_ptr< dealii::Vector<double> > du;
		std::shared_ptr< dealii::Vector<double> > Fu;
		// Matrix L, rhs vectors b and f
		std::shared_ptr< dealii::SparseMatrix<double> > L;
		std::shared_ptr< dealii::Vector<double> > b;
//		std::shared_ptr< dealii::Vector<double> > f;
		
		dealii::SparseDirectUMFPACK iA;

		// Data Output
		std::shared_ptr< fluid::DataPostprocessor<dim> > data_postprocessor;
		std::shared_ptr< DTM::DataOutput<dim> > data_output;
		int    data_output_dwr_loop;
		double data_output_time_value;
		double data_output_trigger;
		bool   data_output_trigger_type_fixed;
	} primal;
	
	virtual void primal_reinit_storage();
	virtual void primal_reinit_storage_on_slab(
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
			const typename DTM::types::storage_data_vectors<1>::iterator &x
	);
	
	virtual void primal_assemble_system(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double > > u
	);
	
 	virtual void primal_assemble_const_rhs(
 		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
 	);

 	virtual void primal_assemble_and_construct_Newton_rhs(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::map<dealii::types::global_dof_index, double> &boundary_values,
		std::shared_ptr< dealii::Vector<double > > u
	);

 	virtual void primal_calculate_boundary_values(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::map<dealii::types::global_dof_index, double> &boundary_values,
		bool zero = false
 	);
	virtual void primal_apply_bc(
		std::map<dealii::types::global_dof_index, double> &boundary_values,
		std::shared_ptr< dealii::Vector<double> > x
	);

	virtual void primal_apply_bc(
		std::map<dealii::types::global_dof_index, double> &boundary_values,
		std::shared_ptr< dealii::SparseMatrix<double> > A,
		std::shared_ptr< dealii::Vector<double> > x,
		std::shared_ptr< dealii::Vector<double> > b
	);

	virtual void primal_solve_slab_problem(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &x
	);

	virtual void primal_subtract_pressure_mean(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double> > x
	);
	
	virtual void primal_do_forward_TMS(
			const unsigned int dwr_loop,
			bool last
	);
	
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
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &x,
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
		std::shared_ptr< dealii::Vector<double> > zm;

		/// temporary storage for dual solution u at \f$ t_n \f$
		std::shared_ptr< dealii::Vector<double> > zn;

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
	virtual void dual_reinit_storage_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z
	);

	virtual void dual_assemble_system(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double > > u
	);

	virtual void dual_assemble_rhs(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual void dual_solve_slab_problem(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z
	);

	virtual void dual_do_backward_TMS(
		const unsigned int dwr_loop,
		bool last
	);

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
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &z,
		const unsigned int dwr_loop,
		bool last
	);

	virtual void dual_sort_xdmf_by_time(
		const unsigned int dwr_loop
	);

	////////////////////////////////////////////////////////////////////////////
	// functional values:
	//

	virtual void compute_functional_values(
			const typename DTM::types::storage_data_vectors<1>::iterator &u,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual double compute_pressure(
			dealii::Point<dim> x,
			std::shared_ptr< dealii::Vector<double> > un,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab
	);

	virtual void compute_drag_lift_tensor(
			std::shared_ptr< dealii::Vector<double> > un,
			const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
			dealii::Tensor<1, dim> &drag_lift_value
	);

	////////////////////////////////////////////////////////////////////////////
	// error estimation and space-time grid adaption
	//
	struct {
		struct {
			/// error indicator \f$ \eta_{I_n} \f$  list
			std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_space;
			std::shared_ptr< DTM::types::storage_data_vectors<1> > eta_time;
//			std::shared_ptr< DTM::types::storage_data_vectors<1> > eta;
		} storage;

		// Data Output
		// std::shared_ptr< DTM::ErrorIndicatorDataPostprocessor<dim> > data_postprocessor;
		std::shared_ptr< DTM::DataOutput<dim> > data_output_space;
		std::shared_ptr< DTM::DataOutput<dim> > data_output_time;
		int    data_output_dwr_loop;
		double data_output_time_value;
		double data_output_trigger;
		bool   data_output_trigger_type_fixed;

		/// error estimator
		std::shared_ptr< fluid::cGp_dGr::cGq_dGs::ErrorEstimator<dim> > pu_dwr;

		struct {
			// J(u) = ...
			// reference computations
			struct {
				// from own computations on fine spatio-temporal meshes
				struct {
					double mean_drag = 0.4027165386203608; // mean drag from 0s to 8s for Stokes 2D-3
					double mean_lift = 0.002576246687436928; // mean lift from 0s to 8s for Stokes 2D-3
				} Stokes;

				// computed reference values for Navier-Stokes from featflow results online
				struct {
					double mean_drag = 1.6031368118815639; // mean drag from 0s to 8s for Navier-Stokes 2D-3
					double mean_lift = -0.010209786642844209; // mean lift from 0s to 8s for Navier-Stokes 2D-3
				} NSE;
			} reference;

			// J(u_{kh}) = ...
			struct {
				double mean_drag; // mean drag from 0s to 8s for (Navier-)Stokes 2D-3
				double mean_lift; // mean lift from 0s to 8s for (Navier-)Stokes 2D-3
			} fem;

			// for debugging: J(u_{kh}) = ...
			struct {
				double J;
			} debug;
		} goal_functional;

	} error_estimator;

	virtual void eta_reinit_storage();
	virtual void eta_reinit_storage_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_t
	);

	virtual void compute_effectivity_index();

	virtual void refine_and_coarsen_space_time_grid(const unsigned int dwr_loop);

	// post-processing functions for data output
	virtual void eta_init_data_output();

	virtual void eta_space_do_data_output_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
		const unsigned int dwr_loop
	);

	virtual void eta_time_do_data_output_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_t,
		const unsigned int dwr_loop
	);


	virtual void eta_space_do_data_output_on_slab_Qn_mode(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
		const unsigned int dwr_loop
	);

	virtual void eta_time_do_data_output_on_slab_Qn_mode(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_t,
		const unsigned int dwr_loop
	);

	virtual void eta_do_data_output(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_t,
		const unsigned int dwr_loop,
		bool last
	);

	virtual void eta_sort_xdmf_by_time(
		const unsigned int dwr_loop
	);

	////////////////////////////////////////////////////////////////////////////
	// other
	//
	
	unsigned int setw_value_dwr_loops;
};

} // namespace

#endif
