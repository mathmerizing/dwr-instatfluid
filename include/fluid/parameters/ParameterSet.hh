/**
 * @file   ParameterSet.hh
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-04-25, merge error estimator, JR
 * @Date 2022-01-17, Added Newton parameters, JPT
 * @Date 2022-01-14, Fluid, JPT
 * @date 2019-11-06, stokes, UK
 * @date 2018-09-14, unified to other DTM programs, UK
 * @date 2018-07-25, new parameters dwr, UK
 * @date 2018-03-06, included from ewave, UK
 * @date 2017-10-25, UK
 * @date 2017-02-06, UK
 *
 * @brief Keeps all parsed input parameters in a struct.
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

#ifndef __ParameterSet_hh
#define __ParameterSet_hh

// PROJECT includes

// DEAL.II includes
#include <deal.II/base/parameter_handler.h>

// C++ includes
#include <string>
#include <memory>

namespace fluid {

struct ParameterSet {
	ParameterSet(std::shared_ptr< dealii::ParameterHandler > handler);
	
	unsigned int dim;
	std::string problem;
	
	// problem specification
	// fe
	struct {
		bool symmetric_stress;

		std::string primal_projection_type;
		std::string dual_projection_type;

		struct FEDescription {
			struct {
				std::string space_type;
				std::string space_type_support_points;
				unsigned int p;
				
				std::string time_type;
				std::string time_type_support_points;
				unsigned int r;
			} convection;
			
			struct {
				std::string space_type;
				std::string space_type_support_points;
				unsigned int p;
				
				std::string time_type;
				std::string time_type_support_points;
				unsigned int r;
			} pressure;
		};

		std::string primal_order; // low or high
		std::string dual_order;   // low or high

		struct FEDescription primal;
		struct FEDescription dual;
		
		struct FEDescription low;
		struct FEDescription high;
	} fe;
	
	// mesh specification
	bool use_mesh_input_file;
	std::string mesh_input_filename;
	
	std::string TriaGenerator;
	std::string TriaGenerator_Options;
	
	struct {
		struct {
			struct {
				std::string Grid_Class;
				std::string Grid_Class_Options;
				
			} fluid;
		} boundary;
		
		struct {
			struct {
				unsigned int global_refinement;
			} fluid;
		} mesh;
	} space;
	
	struct {
		struct {
			double t0;
			double T;
			double tau_n;
			unsigned int initial_time_tria_refinement;
			unsigned int max_intervals_per_slab;
		} fluid;
	} time;
	
	// parameter specification
	std::string viscosity_function;
	std::string viscosity_options;
	
	struct {
		std::string force_function;
		std::string force_options;
		unsigned int force_assembler_n_quadrature_points;
	} fluid;
	
	struct {
		std::string initial_value_function;
		std::string initial_value_options;
		
		std::string dirichlet_boundary_function;
		std::string dirichlet_boundary_options;
		unsigned int dirichlet_assembler_n_quadrature_points;
		
		std::string neumann_boundary_function;
		std::string neumann_boundary_options;
		unsigned int neumann_assembler_n_quadrature_points;
		
		std::string exact_solution_function;
		std::string exact_solution_options;
	} convection;
	
	struct {
		std::string initial_value_function;
		std::string initial_value_options;
		
		std::string exact_solution_function;
		std::string exact_solution_options;
	} pressure;
	
	// data output
	struct {
		struct {
			std::string dwr_loop;
			std::string trigger_type;
			
			double trigger;
			unsigned int patches;
		} primal;
		
		struct {
			std::string dwr_loop;
			std::string trigger_type;
			
			double trigger;
			unsigned int patches;
		} dual;

		struct {
			std::string dwr_loop;
			std::string trigger_type;

			double trigger;
			unsigned int patches;
		} error_estimator;
	} data_output;
	
	//Newton
	struct {
		unsigned int max_steps;
		double       lower_bound;
		double       rebuild;
		unsigned int line_search_steps;
		double       line_search_damping;
	} newton;

	// dwr
	struct {
		struct {
			std::string type;
			
			std::string weight_function;
			std::string weight_options;
		} goal;
		
		struct {
			bool in_use;
			bool reduction_mode;
			
			unsigned int max_iterations;
			double tolerance;
			double reduction;
		} solver_control;
		
		unsigned int loops;
		
		struct {
			struct {
				std::string strategy; // global, global_time, global_space, adaptive
			} spacetime;

			struct {
				std::string strategy; // global, fixed-number, fixed-fraction, RichterWick
				
				double top_fraction1;
				double top_fraction2;
				double bottom_fraction;
				unsigned int max_growth_factor_n_active_cells;
				
				double riwi_alpha;

				double theta1; // Schwegler
				double theta2; // Schwegler
			} space;
			
			struct {
				std::string strategy; // global, fixed-number, fixed-fraction
				
				double top_fraction;
			} time;
		} refine_and_coarsen;

		bool replace_linearization_points;
		bool replace_weights;
	} dwr;
};

} // namespace

#endif
