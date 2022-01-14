/**
 * @file Grid.tpl.cc
 * 
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * 
 * @Date 2022-01-14, Fluid, JPT
 * @date 2022-01-03, added dual, JR
 * @date 2021-11-22, ST hanging nodes, UK
 * @date 2021-11-05, spacetime constraints, JR, UK
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

// PROJECT includes
#include <fluid/grid/Grid.tpl.hh>
#include <fluid/grid/TriaGenerator.tpl.hh>

#include <fluid/types/boundary_id.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe.h>
// #include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
// #include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

// C++ includes
#include <cmath>
#include <limits>
#include <memory>

namespace fluid {

template<int dim>
Grid<dim>::
~Grid() {
	// clear all dof handlers
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
	for (; slab != ends; ++slab) {
		Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.primal.dof.use_count(), dealii::ExcNotInitialized());
		
		slab->space.primal.dof->clear();
		slab->time.primal.dof->clear();

		Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.dual.dof.use_count(), dealii::ExcNotInitialized());

		slab->space.dual.dof->clear();
		slab->time.dual.dof->clear();
	}
}


template<int dim>
void
Grid<dim>::
initialize_slabs() {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
	
	Assert(
		slabs.empty(),
		dealii::ExcMessage(
			"Internal Error: slabs must be empty when calling this function"
		)
	);
	
	// determine initial time intervals
	unsigned int numoftimeintervals;
	numoftimeintervals = static_cast<unsigned int>(std::floor(
		(parameter_set->time.fluid.T-parameter_set->time.fluid.t0)
		/ parameter_set->time.fluid.tau_n
	));
	if (std::abs((numoftimeintervals*parameter_set->time.fluid.tau_n)
		-(parameter_set->time.fluid.T-parameter_set->time.fluid.t0))
		>= std::numeric_limits< double >::epsilon()*parameter_set->time.fluid.T) {
		numoftimeintervals += 1;
	}
	// init spatial "grids" of each slab
	for (unsigned int i{1}; i<= numoftimeintervals; ++i) {
		slabs.emplace_back();
		auto &slab = slabs.back();
		
		////////////////////////////////////////////////////////////////////////
		// space
		
		////////////////////
		// common components
		//
		slab.space.tria = std::make_shared< dealii::Triangulation<dim> >(
			typename dealii::Triangulation<dim>::MeshSmoothing(
				dealii::Triangulation<dim>::maximum_smoothing
			)
		);
		
		/////////////////////////
		// primal grid components
		//
		Assert(slab.space.tria.use_count(), dealii::ExcNotInitialized());
		slab.space.primal.dof = std::make_shared< dealii::DoFHandler<dim> > (
			*slab.space.tria
		);
		
		// check FE configuration parameters
		Assert(
			(!parameter_set->fe.primal.convection.space_type.compare("cG")),
			dealii::ExcMessage("primal convection (Fluid velocity) fe must be of the cG family")
		);
		
		Assert(
			(!parameter_set->fe.primal.pressure.space_type.compare("cG")),
			dealii::ExcMessage("primal pressure fe must be of the cG family")
		);
		
		if (parameter_set->fe.primal.convection.p
			<= parameter_set->fe.primal.pressure.p) {
			DTM::pout
				<< "WARNING: you try to use an unstable primal FE"
				<< std::endl;
		}
		
		// create primal fe system
		{
			// create fe quadratures for support points
			// convection (Fluid velocity) (\boldsymbol b)
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_b;
			{
				if ( !(parameter_set->
					fe.primal.convection.space_type_support_points
					.compare("Gauss-Lobatto")) ) {
					
					fe_quad_b = std::make_shared< dealii::QGaussLobatto<1> > (
						(parameter_set->fe.primal.convection.p + 1)
					);
					
					DTM::pout
						<< "FE: (primal) convection b: "
						<< "created QGaussLobatto<1> quadrature"
						<< std::endl;
				}
			}
			
			AssertThrow(
				fe_quad_b.use_count(),
				dealii::ExcMessage("FE: (primal) convection b support points invalid")
			);
			
			// pressure
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_p;
			{
				if ( !(parameter_set->
					fe.primal.pressure.space_type_support_points
					.compare("Gauss-Lobatto")) ) {
					
					fe_quad_p = std::make_shared< dealii::QGaussLobatto<1> > (
						(parameter_set->fe.primal.pressure.p + 1)
					);
					
					DTM::pout
						<< "FE: (primal) pressure p: "
						<< "created QGaussLobatto<1> quadrature"
						<< std::endl;
				}
			}
			
			AssertThrow(
				fe_quad_p.use_count(),
				dealii::ExcMessage("FE: (primal) pressure p support points invalid")
			);
			
			// create convection FE
			std::shared_ptr<dealii::FiniteElement<dim>> fe_b;
			{
				if (
					!parameter_set->fe.primal.convection.space_type.compare("cG")
				) {
					
					fe_b = std::make_shared<dealii::FE_Q<dim>> (
						*fe_quad_b
					);
				}
				
				AssertThrow(
					fe_b.use_count(),
					dealii::ExcMessage("primal convection FE not known")
				);
			}
			
			// create pressure FE
			std::shared_ptr<dealii::FiniteElement<dim>> fe_p;
			{
				if (
					!parameter_set->fe.primal.pressure.space_type.compare("cG")
				) {
					
					fe_p = std::make_shared<dealii::FE_Q<dim>> (
						*fe_quad_p
					);
				}
				
				AssertThrow(
					fe_p.use_count(),
					dealii::ExcMessage("primal pressure FE not known")
				);
			}
			
			// create FE System
			slab.space.primal.fe = std::make_shared< dealii::FESystem<dim> > (
				// Fluid FE (dim+1)
				dealii::FESystem<dim> (
					// convection FE (component 0 ... 1*dim-1)
					*fe_b, dim,
					// pressure FE (component 1*dim)
					*fe_p, 1
				), 1
			);
			
			DTM::pout
				<< "fluid: created "
				<< slab.space.primal.fe->get_name()
				<< std::endl;
		}
		
		slab.space.primal.constraints = std::make_shared< dealii::AffineConstraints<double> > ();
		slab.space.primal.sp_block_L = std::make_shared< dealii::BlockSparsityPattern >();
		slab.space.primal.sp_L = std::make_shared< dealii::SparsityPattern >();
		slab.space.primal.mapping = std::make_shared< dealii::MappingQ<dim> > (
			std::max(
				static_cast<unsigned int> (1),
				std::max(
					parameter_set->fe.primal.convection.p,
					parameter_set->fe.primal.pressure.p
				)
			)
		);
		
		/////////////////////////
		// dual grid components
		//
		Assert(slab.space.tria.use_count(), dealii::ExcNotInitialized());
		slab.space.dual.dof = std::make_shared< dealii::DoFHandler<dim> > (
			*slab.space.tria
		);

		// check FE configuration parameters
		Assert(
			(!parameter_set->fe.dual.convection.space_type.compare("cG")),
			dealii::ExcMessage("dual convection (Fluid velocity) fe must be of the cG family")
		);

		Assert(
			(!parameter_set->fe.dual.pressure.space_type.compare("cG")),
			dealii::ExcMessage("dual pressure fe must be of the cG family")
		);

		if (parameter_set->fe.dual.convection.p
			<= parameter_set->fe.dual.pressure.p) {
			DTM::pout
				<< "WARNING: you try to use an unstable dual FE"
				<< std::endl;
		}

		// create dual fe system
		{
			// create fe quadratures for support points
			// convection (Fluid velocity) (\boldsymbol z_b)
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_b;
			{
				if ( !(parameter_set->
					fe.dual.convection.space_type_support_points
					.compare("Gauss-Lobatto")) ) {

					fe_quad_z_b = std::make_shared< dealii::QGaussLobatto<1> > (
						(parameter_set->fe.dual.convection.p + 1)
					);

					DTM::pout
						<< "FE: (dual) convection z_b: "
						<< "created QGaussLobatto<1> quadrature"
						<< std::endl;
				}
			}

			AssertThrow(
				fe_quad_z_b.use_count(),
				dealii::ExcMessage("FE: (dual) convection b support points invalid")
			);

			// pressure
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_p;
			{
				if ( !(parameter_set->
					fe.dual.pressure.space_type_support_points
					.compare("Gauss-Lobatto")) ) {

					fe_quad_z_p = std::make_shared< dealii::QGaussLobatto<1> > (
						(parameter_set->fe.dual.pressure.p + 1)
					);

					DTM::pout
						<< "FE: (dual) pressure p: "
						<< "created QGaussLobatto<1> quadrature"
						<< std::endl;
				}
			}

			AssertThrow(
				fe_quad_z_p.use_count(),
				dealii::ExcMessage("FE: (dual) pressure p support points invalid")
			);

			// create convection FE
			std::shared_ptr<dealii::FiniteElement<dim>> fe_z_b;
			{
				if (
					!parameter_set->fe.dual.convection.space_type.compare("cG")
				) {

					fe_z_b = std::make_shared<dealii::FE_Q<dim>> (
						*fe_quad_z_b
					);
				}

				AssertThrow(
					fe_z_b.use_count(),
					dealii::ExcMessage("dual convection FE not known")
				);
			}

			// create pressure FE
			std::shared_ptr<dealii::FiniteElement<dim>> fe_z_p;
			{
				if (
					!parameter_set->fe.dual.pressure.space_type.compare("cG")
				) {

					fe_z_p = std::make_shared<dealii::FE_Q<dim>> (
						*fe_quad_z_p
					);
				}

				AssertThrow(
					fe_z_p.use_count(),
					dealii::ExcMessage("dual pressure FE not known")
				);
			}

			// create FE System
			slab.space.dual.fe = std::make_shared< dealii::FESystem<dim> > (
				// Fluid FE (dim+1)
				dealii::FESystem<dim> (
					// convection FE (component 0 ... 1*dim-1)
					*fe_z_b, dim,
					// pressure FE (component 1*dim)
					*fe_z_p, 1
				), 1
			);

			DTM::pout
				<< "fluid: (dual) created "
				<< slab.space.dual.fe->get_name()
				<< std::endl;
		}

		slab.space.dual.constraints = std::make_shared< dealii::AffineConstraints<double> > ();
		slab.space.dual.sp_block_L = std::make_shared< dealii::BlockSparsityPattern >();
		slab.space.dual.sp_L = std::make_shared< dealii::SparsityPattern >();
		slab.space.dual.mapping = std::make_shared< dealii::MappingQ<dim> > (
			std::max(
				static_cast<unsigned int> (1),
				std::max(
					parameter_set->fe.dual.convection.p,
					parameter_set->fe.dual.pressure.p
				)
			)
		);

		////////////////////////////////////////////////////////////////////////
		// time
		
		////////////////////
		// common components
		//
		slab.time.tria = std::make_shared< dealii::Triangulation<1> >();
		
		/////////////////////////
		// primal grid components
		//
		Assert(slab.time.tria.use_count(), dealii::ExcNotInitialized());
		slab.time.primal.dof = std::make_shared< dealii::DoFHandler<1> > (
			*slab.time.tria
		);
		
		// check FE configuration parameters
		Assert(
			(!parameter_set->fe.primal.convection.time_type.compare("dG")),
			dealii::ExcMessage("primal convection time fe must be of the dG family")
		);
		
		// create primal fe system (time)
		{
			// create fe quadratures for support points
			// convection (b)
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_convection;
			{
				if ( !(parameter_set->
					fe.primal.convection.time_type_support_points
					.compare("Gauss")) ) {
					
					fe_quad_time_convection =
					std::make_shared< dealii::QGauss<1> > (
						(parameter_set->fe.primal.convection.r + 1)
					);
					
// 					DTM::pout
// 						<< "FE time: (primal) convection b: "
// 						<< "created QGauss<1> quadrature"
// 						<< std::endl;
				}
			}
			
			AssertThrow(
				fe_quad_time_convection.use_count(),
				dealii::ExcMessage(
					"FE time: (primal) convection b support points invalid"
				)
			);
			
			// create FE time
			{
				if (
					!parameter_set->fe.primal.convection.time_type.compare("dG")
				) {
					slab.time.primal.fe =
					std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
						*fe_quad_time_convection
					);
				}
				
				AssertThrow(
					slab.time.primal.fe.use_count(),
					dealii::ExcMessage("primal convection FE time not known")
				);
			}
			
// 			DTM::pout
// 				<< "slab.time.primal.fe = "
// 				<< slab.time.primal.fe->get_name()
// 				<< std::endl;
		}
		
		slab.time.primal.mapping = std::make_shared< dealii::MappingQ<1> > (
			parameter_set->fe.primal.convection.r
		);
	
		/////////////////////////
		// dual grid components
		//
		Assert(slab.time.tria.use_count(), dealii::ExcNotInitialized());
		slab.time.dual.dof = std::make_shared< dealii::DoFHandler<1> > (
			*slab.time.tria
		);

		// check FE configuration parameters
		Assert(
			(!parameter_set->fe.dual.convection.time_type.compare("dG")),
			dealii::ExcMessage("dual convection time fe must be of the dG family")
		);

		// create dual fe system (time)
		{
			// create fe quadratures for support points
			// convection (b)
			std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_dual_convection;
			{
				if ( !(parameter_set->
					fe.dual.convection.time_type_support_points
					.compare("Gauss")) ) {

					fe_quad_time_dual_convection =
					std::make_shared< dealii::QGauss<1> > (
						(parameter_set->fe.dual.convection.r + 1)
					);

// 					DTM::pout
// 						<< "FE time: (dual) convection b: "
// 						<< "created QGauss<1> quadrature"
// 						<< std::endl;
				}
			}

			AssertThrow(
				fe_quad_time_dual_convection.use_count(),
				dealii::ExcMessage(
					"FE time: (dual) convection b support points invalid"
				)
			);

			// create FE time
			{
				if (
					!parameter_set->fe.dual.convection.time_type.compare("dG")
				) {
					slab.time.dual.fe =
					std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
						*fe_quad_time_dual_convection
					);
				}

				AssertThrow(
					slab.time.dual.fe.use_count(),
					dealii::ExcMessage("dual convection FE time not known")
				);
			}

	// 			DTM::pout
	// 				<< "slab.time.dual.fe = "
	// 				<< slab.time.primal.fe->get_name()
	// 				<< std::endl;
		}

		slab.time.dual.mapping = std::make_shared< dealii::MappingQ<1> > (
			parameter_set->fe.dual.convection.r
		);
	}

	// init temporal "grids" of each slab
	{
		unsigned int n{1};
		for (auto &slab : slabs) {
			slab.t_m =
				(n-1) * parameter_set->time.fluid.tau_n
				+ parameter_set->time.fluid.t0;
			slab.t_n =
				n * parameter_set->time.fluid.tau_n
				+ parameter_set->time.fluid.t0;
			++n;
			
			slab.refine_in_time=false;
		}
		
		auto &last_slab = slabs.back();
		if ( std::abs(last_slab.t_n - parameter_set->time.fluid.T) >=
			std::numeric_limits< double >::epsilon()*parameter_set->time.fluid.T) {
			last_slab.t_n = parameter_set->time.fluid.T;
		}
	}
}

template<int dim>
void
Grid<dim>::
refine_slab_in_time(
	typename fluid::types::spacetime::dwr::slabs<dim>::iterator slab) {
#ifdef DEBUG
	// check if iterator slab is in the container slabs of this object
	{
		auto _slab{slabs.begin()};
		auto _ends{slabs.end()};
		bool check{false};
		for ( ; _slab != _ends; ++_slab ) {
			if (slab == _slab) {
				check=true;
				break;
			}
		}
		Assert(
			check,
			dealii::ExcMessage("your given iterator slab to be refined could not be found in this->slabs object")
		);
	}
#endif
	
	// emplace a new slab element in front of the iterator
	slabs.emplace(
		slab
	);
	
	// init new slab ("space-time" tria)
	std::prev(slab)->t_m=slab->t_m;
	std::prev(slab)->t_n=slab->t_m + slab->tau_n()/2.;
	slab->t_m=std::prev(slab)->t_n;
	
	////////////////////////////////////////////////////////////////////////////
	// old slab: init new tria and dof of the
	slab->refine_in_time=false;
	
	Assert(slab->time.primal.dof.use_count(), dealii::ExcNotInitialized());
	slab->time.primal.dof->clear();
	slab->time.primal.dof = nullptr;
	
	Assert(slab->time.dual.dof.use_count(), dealii::ExcNotInitialized());
	slab->time.dual.dof->clear();
	slab->time.dual.dof = nullptr;
	
	auto n_active_cells_time{slab->time.tria->n_active_cells()};
	
	slab->time.tria = std::make_shared< dealii::Triangulation<1> > ();
	dealii::GridGenerator::hyper_cube(
		*slab->time.tria,
		slab->t_m, slab->t_n,
		false
	);
	
	for ( ; slab->time.tria->n_active_cells() < n_active_cells_time ; ) {
		slab->time.tria->refine_global(1);
	}
	
	slab->time.primal.dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);
	
	slab->time.dual.dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);

	////////////////////////////////////////////////////////////////////////////
	// new slab: init
	std::prev(slab)->refine_in_time=false;
	
	// space
	std::prev(slab)->space.tria = std::make_shared< dealii::Triangulation<dim> > (
		typename dealii::Triangulation<dim>::MeshSmoothing(
			slab->space.tria->get_mesh_smoothing()
		)
	);
	
	slab->space.tria->reset_all_manifolds();
	std::prev(slab)->space.tria->copy_triangulation(*slab->space.tria);

	//////////////////////////////////////////
	// init primal grid components of new slab
	std::prev(slab)->space.primal.dof = std::make_shared< dealii::DoFHandler<dim> > (
		*std::prev(slab)->space.tria
	);
	
	// create primal fe system (FESystem does not allow to copy)
	{
		// create fe quadratures for support points
		// convection (Fluid velocity) (\boldsymbol b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_b;
		{
			if ( !(parameter_set->
				fe.primal.convection.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_b = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.primal.convection.p + 1)
				);
			}
		}

		Assert(
			fe_quad_b.use_count(),
			dealii::ExcMessage("FE: (primal) convection b support points invalid")
		);

		// pressure
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_p;
		{
			if ( !(parameter_set->
				fe.primal.pressure.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_p = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.primal.pressure.p + 1)
				);
			}
		}

		Assert(
			fe_quad_p.use_count(),
			dealii::ExcMessage("FE: (primal) pressure p support points invalid")
		);

		// create convection FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_b;
		{
			if (
				!parameter_set->fe.primal.convection.space_type.compare("cG")
			) {

				fe_b = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_b
				);
			}

			Assert(
				fe_b.use_count(),
				dealii::ExcMessage("primal convection FE not known")
			);
		}

		// create pressure FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_p;
		{
			if (
				!parameter_set->fe.primal.pressure.space_type.compare("cG")
			) {

				fe_p = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_p
				);
			}

			Assert(
				fe_p.use_count(),
				dealii::ExcMessage("primal pressure FE not known")
			);
		}

		// create FE System
		std::prev(slab)->space.primal.fe =
		std::make_shared< dealii::FESystem<dim> > (
			// Fluid FE (dim+1)
			dealii::FESystem<dim> (
				// convection FE (component 0 ... 1*dim-1)
				*fe_b, dim,
				// pressure FE (component 1*dim)
				*fe_p, 1
			), 1
		);
	}

	std::prev(slab)->space.primal.constraints =
		std::make_shared< dealii::AffineConstraints<double> > ();

	std::prev(slab)->space.primal.sp_block_L =
		std::make_shared< dealii::BlockSparsityPattern >();

	std::prev(slab)->space.primal.sp_L =
		std::make_shared< dealii::SparsityPattern >();

	std::prev(slab)->space.primal.mapping =
	std::make_shared< dealii::MappingQ<dim> > (
		std::max(
			static_cast<unsigned int> (1),
			std::max(
				parameter_set->fe.primal.convection.p,
				parameter_set->fe.primal.pressure.p
			)
		)
	);

	////////////////////////////////////////
	// init dual grid components of new slab
	std::prev(slab)->space.dual.dof = std::make_shared< dealii::DoFHandler<dim> > (
		*std::prev(slab)->space.tria
	);
	
	// create dual fe system (FESystem does not allow to copy)
	{
		// create fe quadratures for support points
		// convection (Fluid velocity) (\boldsymbol z_b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_b;
		{
			if ( !(parameter_set->
				fe.dual.convection.space_type_support_points
				.compare("Gauss-Lobatto")) ) {
				
				fe_quad_z_b = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.dual.convection.p + 1)
				);
			}
		}
		
		Assert(
			fe_quad_z_b.use_count(),
			dealii::ExcMessage("FE: (dual) convection b support points invalid")
		);
		
		// pressure
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_p;
		{
			if ( !(parameter_set->
				fe.dual.pressure.space_type_support_points
				.compare("Gauss-Lobatto")) ) {
				
				fe_quad_z_p = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.dual.pressure.p + 1)
				);
			}
		}
		
		Assert(
			fe_quad_z_p.use_count(),
			dealii::ExcMessage("FE: (dual) pressure p support points invalid")
		);
		
		// create convection FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_z_b;
		{
			if (
				!parameter_set->fe.dual.convection.space_type.compare("cG")
			) {
				
				fe_z_b = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_z_b
				);
			}
			
			Assert(
				fe_z_b.use_count(),
				dealii::ExcMessage("dual convection FE not known")
			);
		}
		
		// create pressure FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_z_p;
		{
			if (
				!parameter_set->fe.dual.pressure.space_type.compare("cG")
			) {
				
				fe_z_p = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_z_p
				);
			}
			
			Assert(
				fe_z_p.use_count(),
				dealii::ExcMessage("dual pressure FE not known")
			);
		}
		
		// create FE System
		std::prev(slab)->space.dual.fe =
		std::make_shared< dealii::FESystem<dim> > (
			// Fluid FE (dim+1)
			dealii::FESystem<dim> (
				// convection FE (component 0 ... 1*dim-1)
				*fe_z_b, dim,
				// pressure FE (component 1*dim)
				*fe_z_p, 1
			), 1
		);
	}
	
	std::prev(slab)->space.dual.constraints =
		std::make_shared< dealii::AffineConstraints<double> > ();
	
	std::prev(slab)->space.dual.sp_block_L =
		std::make_shared< dealii::BlockSparsityPattern >();
	
	std::prev(slab)->space.dual.sp_L =
		std::make_shared< dealii::SparsityPattern >();
	
	std::prev(slab)->space.dual.mapping =
	std::make_shared< dealii::MappingQ<dim> > (
		std::max(
			static_cast<unsigned int> (1),
			std::max(
				parameter_set->fe.dual.convection.p,
				parameter_set->fe.dual.pressure.p
			)
		)
	);
	
	////////////////////////////////////////////////////////////////////////////
	// time
	std::prev(slab)->time.tria = std::make_shared< dealii::Triangulation<1> > ();
	dealii::GridGenerator::hyper_cube(
		*std::prev(slab)->time.tria,
		std::prev(slab)->t_m, std::prev(slab)->t_n,
		false
	);
	
	for ( ; std::prev(slab)->time.tria->n_active_cells() < n_active_cells_time ; ) {
		std::prev(slab)->time.tria->refine_global(1);
	}
	
	// init primal grid components of new slab
	std::prev(slab)->time.primal.dof = std::make_shared< dealii::DoFHandler<1> > (
		*std::prev(slab)->time.tria
	);
	
	// create primal fe system (time)
	{
		// create fe quadratures for support points
		// convection (b) (also used for pressure)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_convection;
		{
			if ( !(parameter_set->
				fe.primal.convection.time_type_support_points
				.compare("Gauss")) ) {
				
				fe_quad_time_convection =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.primal.convection.r + 1)
				);
			}
		}
		
		Assert(
			fe_quad_time_convection.use_count(),
			dealii::ExcMessage(
				"FE time: (primal) convection b support points invalid"
			)
		);
		
		// create FE time
		{
			if (
				!parameter_set->fe.primal.convection.time_type.compare("dG")
			) {
				std::prev(slab)->time.primal.fe =
				std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
					*fe_quad_time_convection
				);
			}
			
			Assert(
				slab->time.primal.fe.use_count(),
				dealii::ExcMessage("primal convection FE time not known")
			);
		}
	}
	
	std::prev(slab)->time.primal.mapping = std::make_shared< dealii::MappingQ<1> > (
		parameter_set->fe.primal.convection.r
	);

	// init dual grid components of new slab
	std::prev(slab)->time.dual.dof = std::make_shared< dealii::DoFHandler<1> > (
		*std::prev(slab)->time.tria
	);

	// create dual fe system (time)
	{
		// create fe quadratures for support points
		// convection (b) (also used for pressure)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_dual_convection;
		{
			if ( !(parameter_set->
				fe.dual.convection.time_type_support_points
				.compare("Gauss")) ) {

				fe_quad_time_dual_convection =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.dual.convection.r + 1)
				);
			}
		}

		Assert(
			fe_quad_time_dual_convection.use_count(),
			dealii::ExcMessage(
				"FE time: (dual) convection b support points invalid"
			)
		);

		// create FE time
		{
			if (
				!parameter_set->fe.dual.convection.time_type.compare("dG")
			) {
				std::prev(slab)->time.dual.fe =
				std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
					*fe_quad_time_dual_convection
				);
			}

			Assert(
				slab->time.dual.fe.use_count(),
				dealii::ExcMessage("dual convection FE time not known")
			);
		}
	}

	std::prev(slab)->time.dual.mapping = std::make_shared< dealii::MappingQ<1> > (
		parameter_set->fe.dual.convection.r
	);
}


/// Generate tria on each slab.
template<int dim>
void
Grid<dim>::
generate() {
	auto slab(this->slabs.begin());
	auto ends(this->slabs.end());
	
	for (; slab != ends; ++slab) {
		{
			Assert(parameter_set.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.tria.use_count(), dealii::ExcNotInitialized());
			fluid::TriaGenerator<dim> tria_generator;
			tria_generator.generate(
				parameter_set->TriaGenerator,
				parameter_set->TriaGenerator_Options,
				slab->space.tria
			);
		}
		
		{
			Assert(slab->time.tria.use_count(), dealii::ExcNotInitialized());
			
			dealii::GridGenerator::hyper_cube(
				*slab->time.tria,
				slab->t_m, slab->t_n,
				false
			);
		}
	}
}


/// Global refinement.
template<int dim>
void
Grid<dim>::
refine_global(
	const unsigned int &space_n,
	const unsigned int &time_n) {
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
	for (; slab != ends; ++slab) {
		slab->space.tria->refine_global(space_n);
		slab->time.tria->refine_global(time_n);
	}
}


/// Set (boundary) manifolds
template<int dim>
void
Grid<dim>::
set_manifolds() {
	// base class: do nothing
}


/// Set boundary indicators
template<int dim>
void
Grid<dim>::
set_boundary_indicators() {
	// base class does not implement this function
	Assert(false, dealii::ExcNotImplemented());
}


template<int dim>
void
Grid<dim>::
distribute() {
	// Distribute the degrees of freedom (dofs)
	DTM::pout
		<< "grid: distribute the degrees of freedom (dofs) on slabs"
		<< std::endl;
	
	auto slab(slabs.begin());
	auto ends(slabs.end());
	
// 	unsigned int n{1};
	for (; slab != ends; ++slab) {
		////////////////////////////////////////////////////////////////////////
		// space
		// distribute primal dofs, create constraints and sparsity pattern sp
		{
			Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.primal.fe.use_count(), dealii::ExcNotInitialized());
			slab->space.primal.dof->distribute_dofs(*(slab->space.primal.fe));
			
			dealii::DoFRenumbering::component_wise(*slab->space.primal.dof);
			
			////////////////////////////////////////////////////////////////////
			// create partitionings for the initialisation of vectors
			//
			
			// get all dofs componentwise
			std::vector< dealii::types::global_dof_index > dofs_per_component(
				slab->space.primal.dof->get_fe_collection().n_components(), 0
			);
			
			dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
				*slab->space.primal.dof,
				true
			);
			
			// set specific values of dof counts
			dealii::types::global_dof_index N_b_offset, N_b; // convection
			dealii::types::global_dof_index N_p_offset, N_p; // pressure
			
			// dof count convection: vector-valued primitive FE
			N_b = 0;
			for (unsigned int d{0}; d < dim; ++d) {
				N_b += dofs_per_component[0*dim+d];
			}
			
			// dof count pressure
			N_p = dofs_per_component[dim];
			
			// set specific global dof offset values
			N_b_offset = 0;
			N_p_offset = N_b;
			
			DTM::pout << "\tN_b (convection dofs) = " << N_b << std::endl;
			DTM::pout << "\tN_p (pressure   dofs) = " << N_p << std::endl;
			
			DTM::pout << "\tn_dofs (primal)       = "
				<< slab->space.primal.dof->n_dofs()
				<< std::endl;
			
			// create dof partitionings
			
			// dof partitioning
			slab->space.primal.locally_owned_dofs =
				std::make_shared< dealii::IndexSet > ();
			*slab->space.primal.locally_owned_dofs =
				slab->space.primal.dof->locally_owned_dofs();
			
			slab->space.primal.partitioning_locally_owned_dofs =
				std::make_shared< std::vector< dealii::IndexSet > >();
			
			slab->space.primal.partitioning_locally_owned_dofs->push_back(
				slab->space.primal.locally_owned_dofs->get_view(
					N_b_offset, N_b_offset+N_b
				)
			);
			
			slab->space.primal.partitioning_locally_owned_dofs->push_back(
				slab->space.primal.locally_owned_dofs->get_view(
					N_p_offset, N_p_offset+N_p
				)
			);
			
			// deal.II BlockVector<double> reinit with partitioning is not possible!
			// create block_sizes data type
			Assert(
				slab->space.primal.partitioning_locally_owned_dofs.use_count(),
				dealii::ExcNotInitialized()
			);
			slab->space.primal.block_sizes =
			std::make_shared < std::vector< dealii::types::global_dof_index > > (
				slab->space.primal.partitioning_locally_owned_dofs->size()
			);
			
			for (unsigned int block_component{0};
				block_component < slab->space.primal.partitioning_locally_owned_dofs->size();
				++block_component) {
				Assert(
					(*slab->space.primal.partitioning_locally_owned_dofs)[block_component].n_elements(),
					dealii::ExcMessage("Error: (*slab->space.primal.partitioning_locally_owned_dofs)[block_component].n_elements() == 0")
				);
				
				(*slab->space.primal.block_sizes)[block_component] =
					(*slab->space.primal.partitioning_locally_owned_dofs)[block_component].n_elements();
			}
			
			// setup constraints (e.g. hanging nodes)
			Assert(
				slab->space.primal.constraints.use_count(),
				dealii::ExcNotInitialized()
			);
			{
				slab->space.primal.constraints->clear();
				slab->space.primal.constraints->reinit();
				
				Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
				dealii::DoFTools::make_hanging_node_constraints(
					*slab->space.primal.dof,
					*slab->space.primal.constraints
				);
				
				slab->space.primal.constraints->close();
			}
			
			// create sparsity pattern
			Assert(slab->space.primal.dof.use_count(), dealii::ExcNotInitialized());
			{
				// logical coupling tables for basis function and corresponding blocks
				dealii::Table<2,dealii::DoFTools::Coupling> coupling_block_L(
					((dim)+1), // rows [phi, psi]
					((dim)+1)  // cols [b, p]
				);
				{
					// loop 0: init with non-coupling
					for (unsigned int i{0}; i < dim+1; ++i)
					for (unsigned int j{0}; j < dim+1; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::none;
					}
					
					// (Dyn.) Fluid: saddle-point block structure:
					//     [ K_bb | B_bp ], K_bb = M_bb + A_bb
					// L = [-------------]
					//     [ B_pb | 0    ]
					
					// loop 1a: init with coupling for phi-phi (matrix K_bb)
					for (unsigned int i{0}; i < dim; ++i)
					for (unsigned int j{0}; j < dim; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
					// loop 1b: init with coupling for phi-psi (matrix B_bp)
					for (unsigned int i{0}; i < dim; ++i)
					for (unsigned int j{dim}; j < dim+1; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
					// loop 2a: init with coupling for psi-phi (matrix B_pb)
					for (unsigned int i{dim}; i < dim+1; ++i)
					for (unsigned int j{0}; j < dim; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
				}
				
				// create sparsity patterns from block couplings
				
				// fluid: "L" block matrix
				dealii::BlockDynamicSparsityPattern dsp;
				dsp.reinit(
					*slab->space.primal.partitioning_locally_owned_dofs
				);
				
				dealii::DoFTools::make_sparsity_pattern(
					*slab->space.primal.dof,
					coupling_block_L,
					dsp,
					*slab->space.primal.constraints,
					true // true => keep constrained entry in sparsity pattern
// 					dealii::Utilities::MPI::this_mpi_process(mpi_comm)
				);
				
				dsp.compress();
				
				Assert(
					slab->space.primal.sp_block_L.use_count(),
					dealii::ExcNotInitialized()
				);
				slab->space.primal.sp_block_L->copy_from(dsp);
			}
			
// 			////////////////////////////////////////////////////////////////////////////
// 			// TODO:    remove the following lines:
// 			// WARNING: only working on 1 mpi process only!
// 			// debug output of the sparsity patterns:
// #ifdef DEBUG
// 			std::ofstream out("sp_L.gpl");
// 			slab->space.primal.sp_block_L->print_gnuplot(out);
// #endif
		}

		////////////////////////////////////////////////////////////////////////
		// space
		// distribute dual dofs, create constraints and sparsity pattern sp
		{
			Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->space.dual.fe.use_count(), dealii::ExcNotInitialized());
			slab->space.dual.dof->distribute_dofs(*(slab->space.dual.fe));

			dealii::DoFRenumbering::component_wise(*slab->space.dual.dof);

			////////////////////////////////////////////////////////////////////
			// create partitionings for the initialisation of vectors
			//

			// get all dofs componentwise
			std::vector< dealii::types::global_dof_index > dofs_per_component(
				slab->space.dual.dof->get_fe_collection().n_components(), 0
			);

			dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
				*slab->space.dual.dof,
				true
			);

			// set specific values of dof counts
			dealii::types::global_dof_index N_b_offset, N_b; // convection
			dealii::types::global_dof_index N_p_offset, N_p; // pressure

			// dof count convection: vector-valued primitive FE
			N_b = 0;
			for (unsigned int d{0}; d < dim; ++d) {
				N_b += dofs_per_component[0*dim+d];
			}

			// dof count pressure
			N_p = dofs_per_component[dim];

			// set specific global dof offset values
			N_b_offset = 0;
			N_p_offset = N_b;

			DTM::pout << "\tN_b (convection dofs) = " << N_b << std::endl;
			DTM::pout << "\tN_p (pressure   dofs) = " << N_p << std::endl;

			DTM::pout << "\tn_dofs (dual)         = "
				<< slab->space.dual.dof->n_dofs()
				<< std::endl;

			// create dof partitionings

			// dof partitioning
			slab->space.dual.locally_owned_dofs =
				std::make_shared< dealii::IndexSet > ();
			*slab->space.dual.locally_owned_dofs =
				slab->space.dual.dof->locally_owned_dofs();

			slab->space.dual.partitioning_locally_owned_dofs =
				std::make_shared< std::vector< dealii::IndexSet > >();

			slab->space.dual.partitioning_locally_owned_dofs->push_back(
				slab->space.dual.locally_owned_dofs->get_view(
					N_b_offset, N_b_offset+N_b
				)
			);

			slab->space.dual.partitioning_locally_owned_dofs->push_back(
				slab->space.dual.locally_owned_dofs->get_view(
					N_p_offset, N_p_offset+N_p
				)
			);

			// deal.II BlockVector<double> reinit with partitioning is not possible!
			// create block_sizes data type
			Assert(
				slab->space.dual.partitioning_locally_owned_dofs.use_count(),
				dealii::ExcNotInitialized()
			);
			slab->space.dual.block_sizes =
			std::make_shared < std::vector< dealii::types::global_dof_index > > (
				slab->space.dual.partitioning_locally_owned_dofs->size()
			);

			for (unsigned int block_component{0};
				block_component < slab->space.dual.partitioning_locally_owned_dofs->size();
				++block_component) {
				Assert(
					(*slab->space.dual.partitioning_locally_owned_dofs)[block_component].n_elements(),
					dealii::ExcMessage("Error: (*slab->space.dual.partitioning_locally_owned_dofs)[block_component].n_elements() == 0")
				);

				(*slab->space.dual.block_sizes)[block_component] =
					(*slab->space.dual.partitioning_locally_owned_dofs)[block_component].n_elements();
			}

			// setup constraints (e.g. hanging nodes)
			Assert(
				slab->space.dual.constraints.use_count(),
				dealii::ExcNotInitialized()
			);
			{
				slab->space.dual.constraints->clear();
				slab->space.dual.constraints->reinit();

				Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
				dealii::DoFTools::make_hanging_node_constraints(
					*slab->space.dual.dof,
					*slab->space.dual.constraints
				);

				slab->space.dual.constraints->close();
			}

			// create sparsity pattern
			Assert(slab->space.dual.dof.use_count(), dealii::ExcNotInitialized());
			{
				// logical coupling tables for basis function and corresponding blocks
				dealii::Table<2,dealii::DoFTools::Coupling> coupling_block_L(
					((dim)+1), // rows [phi, psi]
					((dim)+1)  // cols [b, p]
				);
				{
					// loop 0: init with non-coupling
					for (unsigned int i{0}; i < dim+1; ++i)
					for (unsigned int j{0}; j < dim+1; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::none;
					}

					// (Dyn.) Fluid: saddle-point block structure:
					//     [ K_bb | B_bp ], K_bb = M_bb + A_bb
					// L = [-------------]
					//     [ B_pb | 0    ]

					// loop 1a: init with coupling for phi-phi (matrix K_bb)
					for (unsigned int i{0}; i < dim; ++i)
					for (unsigned int j{0}; j < dim; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
					// loop 1b: init with coupling for phi-psi (matrix B_bp)
					for (unsigned int i{0}; i < dim; ++i)
					for (unsigned int j{dim}; j < dim+1; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
					// loop 2a: init with coupling for psi-phi (matrix B_pb)
					for (unsigned int i{dim}; i < dim+1; ++i)
					for (unsigned int j{0}; j < dim; ++j) {
						coupling_block_L[i][j] = dealii::DoFTools::always;
					}
				}

				// create sparsity patterns from block couplings

				// fluid: "L" block matrix
				dealii::BlockDynamicSparsityPattern dsp;
				dsp.reinit(
					*slab->space.dual.partitioning_locally_owned_dofs
				);

				dealii::DoFTools::make_sparsity_pattern(
					*slab->space.dual.dof,
					coupling_block_L,
					dsp,
					*slab->space.dual.constraints,
					true // true => keep constrained entry in sparsity pattern
// 					dealii::Utilities::MPI::this_mpi_process(mpi_comm)
				);

				dsp.compress();

				Assert(
					slab->space.dual.sp_block_L.use_count(),
					dealii::ExcNotInitialized()
				);
				slab->space.dual.sp_block_L->copy_from(dsp);
			}

// 			////////////////////////////////////////////////////////////////////////////
// 			// TODO:    remove the following lines:
// 			// WARNING: only working on 1 mpi process only!
// 			// debug output of the sparsity patterns:
// #ifdef DEBUG
// 			std::ofstream out("sp_L.gpl");
// 			slab->space.dual.sp_block_L->print_gnuplot(out);
// #endif
		}

		
		////////////////////////////////////////////////////////////////////////
		// time
		// distribute primal dofs
		{
			Assert(slab->time.primal.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->time.primal.fe.use_count(), dealii::ExcNotInitialized());
			slab->time.primal.dof->distribute_dofs(*(slab->time.primal.fe));
		}

		////////////////////////////////////////////////////////////////////////
		// time
		// distribute dual dofs
		{
			Assert(slab->time.dual.dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->time.dual.fe.use_count(), dealii::ExcNotInitialized());
			slab->time.dual.dof->distribute_dofs(*(slab->time.dual.fe));
		}
		
		////////////////////////////////////////////////////////////////////////////
		// space-time
		//
		
		// primal dof partitioning
		{
			slab->spacetime.primal.locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (
				slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()
			);
			
			for (dealii::types::global_dof_index time_dof_index{0};
				time_dof_index < slab->time.primal.dof->n_dofs(); ++time_dof_index) {
				
				slab->spacetime.primal.locally_owned_dofs->add_indices(
					*slab->space.primal.locally_owned_dofs,
					time_dof_index * slab->space.primal.dof->n_dofs() // offset
				);
			}
		}
		
		// primal constraints
		{
			slab->spacetime.primal.constraints =
				std::make_shared< dealii::AffineConstraints<double> > ();
			
			Assert(
				slab->spacetime.primal.constraints.use_count(),
				dealii::ExcNotInitialized()
			);
			
			slab->spacetime.primal.constraints->clear();
			slab->spacetime.primal.constraints->reinit();
			
			slab->spacetime.primal.constraints->merge(
				*slab->space.primal.constraints
			);
			
			if (slab->time.primal.dof->n_dofs() > 1) {
				auto constraints = std::make_shared< dealii::AffineConstraints<double> > (
					*slab->space.primal.constraints
				);
				for (dealii::types::global_dof_index d{1};
					d < slab->time.primal.dof->n_dofs(); ++d) {
					constraints->shift(
						slab->space.primal.dof->n_dofs()
					);
					
					slab->spacetime.primal.constraints->merge(
						*constraints
					);
				}
			}
			
			slab->spacetime.primal.constraints->close();
			
// 			DTM::pout
// 				<< "Number of constraints: "
// 				<< slab->spacetime.primal.constraints->n_constraints()
// 				<< std::endl;
			
// #ifdef DEBUG
// 			std::cout << "space-time constraints" << std::endl;
// 			slab->spacetime.primal.constraints->print(std::cout);
// 			std::cout << std::endl << std::endl;
// 			
// 			// TODO: remove the following:
// 			std::ofstream out("constraints.log");
// 			slab->spacetime.primal.constraints->print(out);
// #endif
		}
		
		// primal sparsity pattern (downwind test function in time)
		{
			// sparsity pattern block for space-time cell-volume integrals
			// on the slab
			//
			// Step 1: transfer
			// (Dyn.) Fluid: saddle-point block structure:
			//
			//     [ K_bb | B_bp ], K_bb = M_bb + A_bb
			// L = [-------------]
			//     [ B_pb | 0    ]
			// into single block (classical) sparsity pattern.
			//
			// NOTE: Step 1 can be omitted if the assembly is done directly into
			//       a classical SparsityPattern. Anyhow, we keep the block
			//       structure here for a single time dof to make the space-time
			//       sparsity pattern more optimal (only convection has a temporal
			//       derivative).
			//
			// Step 2: create sparsity pattern for space-time tensor product.
			
			// Step 1:
			{
				Assert(
					slab->space.primal.sp_block_L.use_count(),
					dealii::ExcNotInitialized()
				);
				
				Assert(
					slab->space.primal.block_sizes.use_count(),
					dealii::ExcNotInitialized()
				);
				
				dealii::DynamicSparsityPattern dsp_transfer_L(
					slab->space.primal.dof->n_dofs(), // test
					slab->space.primal.dof->n_dofs()  // trial
				);
				
				// K_bb block
				for ( auto &sp_entry : slab->space.primal.sp_block_L->block(0,0) ) {
					dsp_transfer_L.add(
						sp_entry.row(),
						sp_entry.column()
					);
				}
				
				// E_bp block
				for ( auto &sp_entry : slab->space.primal.sp_block_L->block(0,1) ) {
					dsp_transfer_L.add(
						sp_entry.row(),
						sp_entry.column() + (*slab->space.primal.block_sizes)[0]
					);
				}
				
				// E_pb block
				for ( auto &sp_entry : slab->space.primal.sp_block_L->block(1,0) ) {
					dsp_transfer_L.add(
						sp_entry.row() + (*slab->space.primal.block_sizes)[0],
						sp_entry.column()
					);
				}
				
				slab->space.primal.sp_L = std::make_shared< dealii::SparsityPattern > ();
				Assert(slab->space.primal.sp_L.use_count(), dealii::ExcNotInitialized());
				slab->space.primal.sp_L->copy_from(dsp_transfer_L);
			}
			
			// Step 2:
			
			dealii::DynamicSparsityPattern dsp(
				slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs(), // test
				slab->space.primal.dof->n_dofs() * slab->time.primal.dof->n_dofs()  // trial
			);
			
			{
				// sparsity pattern block for coupling between time cells
				// within the slab by downwind the test functions and a jump
				// term (trace operator) on the trial functions
				{
					Assert(
						slab->space.primal.sp_block_L.use_count(),
						dealii::ExcNotInitialized()
					);
					
					// TODO: use sp_L, once you have more time derivatives!
					// NOTE: here we have only ((phi * \partial_t b)).
					auto cell{slab->space.primal.sp_block_L->block(0,0).begin()};
					auto endc{slab->space.primal.sp_block_L->block(0,0).end()};
					
					dealii::types::global_dof_index offset{0};
					for ( ; cell != endc; ++cell) {
						for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {
							
							offset =
								slab->space.primal.dof->n_dofs() *
								nn * slab->time.primal.fe->dofs_per_cell;
							
							if (nn)
							for (unsigned int i{0}; i < slab->time.primal.fe->dofs_per_cell; ++i)
							for (unsigned int j{0}; j < slab->time.primal.fe->dofs_per_cell; ++j) {
								dsp.add(
									// downwind test
									cell->row() + offset
									+ slab->space.primal.dof->n_dofs() * i,
									// trial functions of K^-
									cell->column() + offset
									- slab->space.primal.dof->n_dofs() * slab->time.primal.fe->dofs_per_cell
									+ slab->space.primal.dof->n_dofs() * j
								);
							}
						}
					}
				}
				
				// sparsity pattern block for space-time cell-volume integrals
				// on the slab
				{
					Assert(
						slab->space.primal.sp_L.use_count(),
						dealii::ExcNotInitialized()
					);
					
					auto cell{slab->space.primal.sp_L->begin()};
					auto endc{slab->space.primal.sp_L->end()};
					
					dealii::types::global_dof_index offset{0};
					for ( ; cell != endc; ++cell) {
						for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {
							
							offset =
								slab->space.primal.dof->n_dofs() *
								nn * slab->time.primal.fe->dofs_per_cell;
							
							for (unsigned int i{0}; i < slab->time.primal.fe->dofs_per_cell; ++i)
							for (unsigned int j{0}; j < slab->time.primal.fe->dofs_per_cell; ++j) {
								dsp.add(
									// test
									cell->row() + offset
									+ slab->space.primal.dof->n_dofs() * i,
									// trial
									cell->column() + offset
									+ slab->space.primal.dof->n_dofs() * j
								);
							}
						}
					}
				}
			}
			
			slab->spacetime.primal.sp = std::make_shared< dealii::SparsityPattern > ();
			Assert(slab->spacetime.primal.sp.use_count(), dealii::ExcNotInitialized());
			
// 			slab->spacetime.primal.constraints->condense(dsp);
			
			slab->spacetime.primal.sp->copy_from(dsp);
		}

		////////////////////////////////////////////////////////////////////////////
		// space-time
		//

		// dual dof partitioning
		{
			slab->spacetime.dual.locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (
				slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()
			);

			for (dealii::types::global_dof_index time_dof_index{0};
				time_dof_index < slab->time.dual.dof->n_dofs(); ++time_dof_index) {

				slab->spacetime.dual.locally_owned_dofs->add_indices(
					*slab->space.dual.locally_owned_dofs,
					time_dof_index * slab->space.dual.dof->n_dofs() // offset
				);
			}
		}

		// dual constraints
		{
			slab->spacetime.dual.constraints =
				std::make_shared< dealii::AffineConstraints<double> > ();

			Assert(
				slab->spacetime.dual.constraints.use_count(),
				dealii::ExcNotInitialized()
			);

			slab->spacetime.dual.constraints->clear();
			slab->spacetime.dual.constraints->reinit();

			slab->spacetime.dual.constraints->merge(
				*slab->space.dual.constraints
			);

			if (slab->time.dual.dof->n_dofs() > 1) {
				auto constraints = std::make_shared< dealii::AffineConstraints<double> > (
					*slab->space.dual.constraints
				);
				for (dealii::types::global_dof_index d{1};
					d < slab->time.dual.dof->n_dofs(); ++d) {
					constraints->shift(
						slab->space.dual.dof->n_dofs()
					);

					slab->spacetime.dual.constraints->merge(
						*constraints
					);
				}
			}

			slab->spacetime.dual.constraints->close();

// 			DTM::pout
// 				<< "Number of constraints: "
// 				<< slab->spacetime.dual.constraints->n_constraints()
// 				<< std::endl;

// #ifdef DEBUG
// 			std::cout << "space-time constraints" << std::endl;
// 			slab->spacetime.dual.constraints->print(std::cout);
// 			std::cout << std::endl << std::endl;
//
// 			// TODO: remove the following:
// 			std::ofstream out("constraints.log");
// 			slab->spacetime.dual.constraints->print(out);
// #endif
		}

		// dual sparsity pattern (downwind test function in time)
		{
			// sparsity pattern block for space-time cell-volume integrals
			// on the slab
			//
			// Step 1: transfer
			// (Dyn.) Fluid: saddle-point block structure:
			//
			//     [ K_bb | B_bp ], K_bb = M_bb + A_bb
			// L = [-------------]
			//     [ B_pb | 0    ]
			// into single block (classical) sparsity pattern.
			//
			// NOTE: Step 1 can be omitted if the assembly is done directly into
			//       a classical SparsityPattern. Anyhow, we keep the block
			//       structure here for a single time dof to make the space-time
			//       sparsity pattern more optimal (only convection has a temporal
			//       derivative).
			//
			// Step 2: create sparsity pattern for space-time tensor product.

			// Step 1:
			{
				Assert(
					slab->space.dual.sp_block_L.use_count(),
					dealii::ExcNotInitialized()
				);

				Assert(
					slab->space.dual.block_sizes.use_count(),
					dealii::ExcNotInitialized()
				);

				dealii::DynamicSparsityPattern dsp_transfer_L(
					slab->space.dual.dof->n_dofs(), // test
					slab->space.dual.dof->n_dofs()  // trial
				);

				// K_bb block
				for ( auto &sp_entry : slab->space.dual.sp_block_L->block(0,0) ) {
					dsp_transfer_L.add(
						sp_entry.row(),
						sp_entry.column()
					);
				}

				// E_bp block
				for ( auto &sp_entry : slab->space.dual.sp_block_L->block(0,1) ) {
					dsp_transfer_L.add(
						sp_entry.row(),
						sp_entry.column() + (*slab->space.dual.block_sizes)[0]
					);
				}

				// E_pb block
				for ( auto &sp_entry : slab->space.dual.sp_block_L->block(1,0) ) {
					dsp_transfer_L.add(
						sp_entry.row() + (*slab->space.dual.block_sizes)[0],
						sp_entry.column()
					);
				}

				slab->space.dual.sp_L = std::make_shared< dealii::SparsityPattern > ();
				Assert(slab->space.dual.sp_L.use_count(), dealii::ExcNotInitialized());
				slab->space.dual.sp_L->copy_from(dsp_transfer_L);
			}

			// Step 2:

			dealii::DynamicSparsityPattern dual_dsp(
				slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs(), // test
				slab->space.dual.dof->n_dofs() * slab->time.dual.dof->n_dofs()  // trial
			);

			{
				// sparsity pattern block for coupling between time cells
				// within the slab by downwind the test functions and a jump
				// term (trace operator) on the trial functions
				{
					Assert(
						slab->space.dual.sp_block_L.use_count(),
						dealii::ExcNotInitialized()
					);

					// TODO: use sp_L, once you have more time derivatives!
					// NOTE: here we have only ((phi * \partial_t b)).
					auto cell{slab->space.dual.sp_block_L->block(0,0).begin()};
					auto endc{slab->space.dual.sp_block_L->block(0,0).end()};

					dealii::types::global_dof_index offset{0};
					for ( ; cell != endc; ++cell) {
						for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {

							offset =
								slab->space.dual.dof->n_dofs() *
								nn * slab->time.dual.fe->dofs_per_cell;

							if (nn)
							for (unsigned int i{0}; i < slab->time.dual.fe->dofs_per_cell; ++i)
							for (unsigned int j{0}; j < slab->time.dual.fe->dofs_per_cell; ++j) {
								dual_dsp.add(
									// downwind test
									cell->row() + offset
									+ slab->space.dual.dof->n_dofs() * i,
									// trial functions of K^-
									cell->column() + offset
									- slab->space.dual.dof->n_dofs() * slab->time.dual.fe->dofs_per_cell
									+ slab->space.dual.dof->n_dofs() * j
								);
							}
						}
					}
				}

				// sparsity pattern block for space-time cell-volume integrals
				// on the slab
				{
					Assert(
						slab->space.dual.sp_L.use_count(),
						dealii::ExcNotInitialized()
					);

					auto cell{slab->space.dual.sp_L->begin()};
					auto endc{slab->space.dual.sp_L->end()};

					dealii::types::global_dof_index offset{0};
					for ( ; cell != endc; ++cell) {
						for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {

							offset =
								slab->space.dual.dof->n_dofs() *
								nn * slab->time.dual.fe->dofs_per_cell;

							for (unsigned int i{0}; i < slab->time.dual.fe->dofs_per_cell; ++i)
							for (unsigned int j{0}; j < slab->time.dual.fe->dofs_per_cell; ++j) {
								dual_dsp.add(
									// test
									cell->row() + offset
									+ slab->space.dual.dof->n_dofs() * i,
									// trial
									cell->column() + offset
									+ slab->space.dual.dof->n_dofs() * j
								);
							}
						}
					}
				}
			}

			slab->spacetime.dual.sp = std::make_shared< dealii::SparsityPattern > ();
			Assert(slab->spacetime.dual.sp.use_count(), dealii::ExcNotInitialized());

// 			slab->spacetime.dual.constraints->condense(dual_dsp);

			slab->spacetime.dual.sp->copy_from(dual_dsp);
		}
		
// 		////////////////////////////////////////////////////////////////////////
// 		++n;
	} // end for-loop slab
	
	DTM::pout << std::endl;
}

} // namespaces

#include "Grid.inst.in"
