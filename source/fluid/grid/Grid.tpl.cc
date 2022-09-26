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
 * @date 2019-08-27, merge into DWR, UK
 * @date 2019-01-28, space-time, UK
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
#include <fluid/QRightBox.tpl.hh>

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
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <ideal.II/dofs/SlabDoFTools.hh>
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
		// clear LOW dof handlers
		if ( !parameter_set->fe.primal_order.compare("low") ||
			 !parameter_set->fe.dual_order.compare("low") ||
			 (slab->space.low.fe_info->dof.use_count() > 0) ||
			 (slab->time.low.fe_info->dof.use_count() > 0))
		{
			Assert(slab->space.low.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->time.low.fe_info->dof.use_count(), dealii::ExcNotInitialized());

			slab->space.low.fe_info->dof->clear();
			slab->time.low.fe_info->dof->clear();
		}

		// clear HIGH dof handlers
		if ( !parameter_set->fe.primal_order.compare("high") ||
			 !parameter_set->fe.dual_order.compare("high") ||
			 (slab->space.high.fe_info->dof.use_count() > 0) ||
			 (slab->time.high.fe_info->dof.use_count() > 0))
		{
			Assert(slab->space.high.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->time.high.fe_info->dof.use_count(), dealii::ExcNotInitialized());

			slab->space.high.fe_info->dof->clear();
			slab->time.high.fe_info->dof->clear();
		}

		// clear PU dof handlers
		if ( (slab->space.pu.fe_info->dof.use_count() > 0) ||
			 (slab->time.pu.fe_info->dof.use_count() > 0))
		{
			Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			Assert(slab->time.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());

			slab->space.pu.fe_info->dof->clear();
			slab->time.pu.fe_info->dof->clear();
		}

		if (parameter_set->dwr.functional.mean_vorticity){
			if ( slab->space.vorticity.fe_info->dof.use_count() > 0 ||
			     slab->time.vorticity.fe_info->dof.use_count() > 0){
				Assert(slab->space.vorticity.fe_info->dof.use_count(), dealii::ExcNotInitialized());
				Assert(slab->time.vorticity.fe_info->dof.use_count(), dealii::ExcNotInitialized());

				slab->space.vorticity.fe_info->dof->clear();
				slab->time.vorticity.fe_info->dof->clear();
			}
		}
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
		slab.space.tria = std::make_shared< dealii::parallel::distributed::Triangulation<dim> >(
			mpi_comm,
			typename dealii::parallel::distributed::Triangulation<dim>::MeshSmoothing(
				dealii::parallel::distributed::Triangulation<dim>::maximum_smoothing
			),
			dealii::parallel::distributed::Triangulation<dim>::Settings::no_automatic_repartitioning
		);
		
		////////////////////////////////////////////////////////////////////////
		// time
		
		////////////////////
		// common components
		//
		slab.time.tria = std::make_shared< dealii::Triangulation<1> >();
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
initialize_low_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////
	// space

	/////////////////////////
	// low grid components
	//
	Assert(slab->space.tria.use_count(), dealii::ExcNotInitialized());
	slab->space.low.fe_info->dof = std::make_shared< dealii::DoFHandler<dim> > (
		*slab->space.tria
	);

	// check FE configuration parameters
	Assert(
		(!parameter_set->fe.low.convection.space_type.compare("cG")),
		dealii::ExcMessage("low convection (Fluid velocity) fe must be of the cG family")
	);

	Assert(
		(!parameter_set->fe.low.pressure.space_type.compare("cG")),
		dealii::ExcMessage("low pressure fe must be of the cG family")
	);

	if (parameter_set->fe.low.convection.p
		<= parameter_set->fe.low.pressure.p) {
		DTM::pout
			<< "WARNING: you try to use an unstable low FE"
			<< std::endl;
	}

	// create low fe system
	{
		// create fe quadratures for support points
		// convection (Fluid velocity) (\boldsymbol b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_b;
		{
			if ( !(parameter_set->
				fe.low.convection.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_b = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.low.convection.p + 1)
				);

				DTM::pout
					<< "FE: (low) convection b: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
			}
		}

		AssertThrow(
			fe_quad_b.use_count(),
			dealii::ExcMessage("FE: (low) convection b support points invalid")
		);

		// pressure
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_p;
		{
			if ( !(parameter_set->
				fe.low.pressure.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_p = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.low.pressure.p + 1)
				);

				DTM::pout
					<< "FE: (low) pressure p: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
			}
		}

		AssertThrow(
			fe_quad_p.use_count(),
			dealii::ExcMessage("FE: (low) pressure p support points invalid")
		);

		// create convection FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_b;
		{
			if (
				!parameter_set->fe.low.convection.space_type.compare("cG")
			) {

				fe_b = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_b
				);
			}

			AssertThrow(
				fe_b.use_count(),
				dealii::ExcMessage("low convection FE not known")
			);
		}

		// create pressure FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_p;
		{
			if (
				!parameter_set->fe.low.pressure.space_type.compare("cG")
			) {

				fe_p = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_p
				);
			}

			AssertThrow(
				fe_p.use_count(),
				dealii::ExcMessage("low pressure FE not known")
			);
		}

		// create FE System
		slab->space.low.fe_info->fe = std::make_shared< dealii::FESystem<dim> > (
			// Navier-Stokes FE (dim+1)
			dealii::FESystem<dim> (
				// convection FE (component 0 ... 1*dim-1)
				*fe_b, dim,
				// pressure FE (component 1*dim)
				*fe_p, 1
			), 1
		);

		DTM::pout
			<< "fluid: created "
			<< slab->space.low.fe_info->fe->get_name()
			<< std::endl;
	}

	slab->space.low.fe_info->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.low.fe_info->hanging_node_constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.low.fe_info->initial_constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.low.fe_info->mapping = std::make_shared< dealii::MappingQ<dim> > (
			1
	);

	////////////////////////////////////////////////////////////////////////
	// time

	/////////////////////////
	// low grid components
	//
	Assert(slab->time.tria.use_count(), dealii::ExcNotInitialized());
	slab->time.low.fe_info->dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);

	// check FE configuration parameters
	Assert(
		(!parameter_set->fe.low.convection.time_type.compare("dG")),
		dealii::ExcMessage("low convection time fe must be of the dG family")
	);

	// create low fe system (time)
	{
		// create fe quadratures for support points
		// convection (b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_convection;
		{
			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss")) ) {

				fe_quad_time_convection =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.low.convection.r + 1)
				);

 					DTM::pout
 						<< "FE time: (low) convection b: "
 						<< "created QGauss<1> quadrature"
 						<< std::endl;
			} else if ( !(parameter_set->
					fe.low.convection.time_type_support_points
					.compare("Gauss-Lobatto")) ){

				if (parameter_set->fe.low.convection.r < 1){
					fe_quad_time_convection =
							std::make_shared< QRightBox<1> > ();
						DTM::pout
							<< "FE time: (low) convection b: "
							<< "created QRightBox quadrature"
							<< std::endl;
				} else {
					fe_quad_time_convection =
							std::make_shared< dealii::QGaussLobatto<1> > (
									(parameter_set->fe.low.convection.r + 1)
							);

					DTM::pout
					<< "FE time: (low) convection b: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
				}
			}
		}

		AssertThrow(
			fe_quad_time_convection.use_count(),
			dealii::ExcMessage(
				"FE time: (low) convection b support points invalid"
			)
		);

		// create FE time
		{
			if (
				!parameter_set->fe.low.convection.time_type.compare("dG")
			) {
				slab->time.low.fe_info->fe =
				std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
					*fe_quad_time_convection
				);
			}
			
			AssertThrow(
				slab->time.low.fe_info->fe.use_count(),
				dealii::ExcMessage("low convection FE time not known")
			);
		}

// 			DTM::pout
// 				<< "slab->time.low.fe_info->fe = "
// 				<< slab->time.low.fe_info->fe->get_name()
// 				<< std::endl;
	}

	slab->time.low.fe_info->mapping = std::make_shared< dealii::MappingQ<1> > (
		parameter_set->fe.low.convection.r
	);
}

template<int dim>
void
Grid<dim>::
initialize_high_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////
	// space

	/////////////////////////
	// high grid components
	//
	Assert(slab->space.tria.use_count(), dealii::ExcNotInitialized());
	slab->space.high.fe_info->dof = std::make_shared< dealii::DoFHandler<dim> > (
		*slab->space.tria
	);

	// check FE configuration parameters
	Assert(
		(!parameter_set->fe.high.convection.space_type.compare("cG")),
		dealii::ExcMessage("high convection (Fluid velocity) fe must be of the cG family")
	);

	Assert(
		(!parameter_set->fe.high.pressure.space_type.compare("cG")),
		dealii::ExcMessage("high pressure fe must be of the cG family")
	);

	if (parameter_set->fe.high.convection.p
		<= parameter_set->fe.high.pressure.p) {
		DTM::pout
			<< "WARNING: you try to use an unstable high FE"
			<< std::endl;
	}

	// create high fe system
	{
		// create fe quadratures for support points
		// convection (Fluid velocity) (\boldsymbol z_b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_b;
		{
			if ( !(parameter_set->
				fe.high.convection.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_z_b = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.high.convection.p + 1)
				);

				DTM::pout
					<< "FE: (high) convection z_b: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
			}
		}

		AssertThrow(
			fe_quad_z_b.use_count(),
			dealii::ExcMessage("FE: (high) convection b support points invalid")
		);

		// pressure
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_z_p;
		{
			if ( !(parameter_set->
				fe.high.pressure.space_type_support_points
				.compare("Gauss-Lobatto")) ) {

				fe_quad_z_p = std::make_shared< dealii::QGaussLobatto<1> > (
					(parameter_set->fe.high.pressure.p + 1)
				);

				DTM::pout
					<< "FE: (high) pressure p: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
			}
		}

		AssertThrow(
			fe_quad_z_p.use_count(),
			dealii::ExcMessage("FE: (high) pressure p support points invalid")
		);

		// create convection FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_z_b;
		{
			if (
				!parameter_set->fe.high.convection.space_type.compare("cG")
			) {

				fe_z_b = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_z_b
				);
			}

			AssertThrow(
				fe_z_b.use_count(),
				dealii::ExcMessage("high convection FE not known")
			);
		}

		// create pressure FE
		std::shared_ptr<dealii::FiniteElement<dim>> fe_z_p;
		{
			if (
				!parameter_set->fe.high.pressure.space_type.compare("cG")
			) {

				fe_z_p = std::make_shared<dealii::FE_Q<dim>> (
					*fe_quad_z_p
				);
			}

			AssertThrow(
				fe_z_p.use_count(),
				dealii::ExcMessage("high pressure FE not known")
			);
		}

		// create FE System
		slab->space.high.fe_info->fe = std::make_shared< dealii::FESystem<dim> > (
			// Navier-Stokes FE (dim+1)
			dealii::FESystem<dim> (
				// convection FE (component 0 ... 1*dim-1)
				*fe_z_b, dim,
				// pressure FE (component 1*dim)
				*fe_z_p, 1
			), 1
		);

		DTM::pout
			<< "fluid: (high) created "
			<< slab->space.high.fe_info->fe->get_name()
			<< std::endl;
	}

	slab->space.high.fe_info->constraints = std::make_shared< dealii::AffineConstraints<double> > ();


	slab->space.high.fe_info->hanging_node_constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.high.fe_info->initial_constraints = std::make_shared< dealii::AffineConstraints<double> > ();


	slab->space.high.fe_info->mapping = std::make_shared< dealii::MappingQ<dim> > (
		1
	);

	////////////////////////////////////////////////////////////////////////
	// time

	/////////////////////////
	// high grid components
	//
	Assert(slab->time.tria.use_count(), dealii::ExcNotInitialized());
	slab->time.high.fe_info->dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);

	// check FE configuration parameters
	Assert(
		(!parameter_set->fe.high.convection.time_type.compare("dG")),
		dealii::ExcMessage("high convection time fe must be of the dG family")
	);

	// create high fe system (time)
	{
		// create fe quadratures for support points
		// convection (b)
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_high_convection;
		{
			if ( !(parameter_set->
				fe.high.convection.time_type_support_points
				.compare("Gauss")) ) {

				fe_quad_time_high_convection =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.high.convection.r + 1)
				);

					DTM::pout
						<< "FE time: (high) convection b: "
						<< "created QGauss<1> quadrature"
						<< std::endl;
			} else if ( !(parameter_set->
					fe.high.convection.time_type_support_points
					.compare("Gauss-Lobatto")) ){

				if (parameter_set->fe.high.convection.r < 1){
					fe_quad_time_high_convection =
							std::make_shared< QRightBox<1> > ();
						DTM::pout
							<< "FE time: (high) convection b: "
							<< "created QRightBox quadrature"
							<< std::endl;
				} else {
					fe_quad_time_high_convection =
							std::make_shared< dealii::QGaussLobatto<1> > (
									(parameter_set->fe.high.convection.r + 1)
							);

					DTM::pout
					<< "FE time: (high) convection b: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
				}
			}
		}

		AssertThrow(
			fe_quad_time_high_convection.use_count(),
			dealii::ExcMessage(
				"FE time: (high) convection b support points invalid"
			)
		);

		// create FE time
		{
			if (
				!parameter_set->fe.high.convection.time_type.compare("dG")
			) {
				slab->time.high.fe_info->fe =
				std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
					*fe_quad_time_high_convection
				);
			}

			AssertThrow(
				slab->time.high.fe_info->fe.use_count(),
				dealii::ExcMessage("high convection FE time not known")
			);
		}

// 			DTM::pout
// 				<< "slab->time.high.fe_info->fe = "
// 				<< slab->time.high.fe_info->fe->get_name()
// 				<< std::endl;
	}

	slab->time.high.fe_info->mapping = std::make_shared< dealii::MappingQ<1> > (
		parameter_set->fe.high.convection.r
	);
}

template<int dim>
void
Grid<dim>::
initialize_pu_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////
	// space

	/////////////////////////
	// pu grid components
	//
	Assert(slab->space.tria.use_count(), dealii::ExcNotInitialized());
	slab->space.pu.fe_info->dof = std::make_shared< dealii::DoFHandler<dim> > (
		*slab->space.tria
	);

	// FE configuration parameters for PU:
	// cG(1) in space

	// create pu fe system
	{
		// create fe quadrature for support points
		std::shared_ptr< dealii::Quadrature<1> > fe_quad;
		{
			fe_quad = std::make_shared< dealii::QGaussLobatto<1> > (2);

			DTM::pout
				<< "FE: (PU) pressure p: "
				<< "created QGaussLobatto<1> quadrature"
				<< std::endl;
		}

		// create FE
		slab->space.pu.fe_info->fe = std::make_shared<dealii::FE_Q<dim>> (
			*fe_quad
		);

		DTM::pout
			<< "stokes: (PU) created "
			<< slab->space.pu.fe_info->fe->get_name()
			<< std::endl;
	}

	slab->space.pu.fe_info->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.pu.fe_info->mapping = std::make_shared< dealii::MappingQ<dim> > (1);

	////////////////////////////////////////////////////////////////////////
	// time

	/////////////////////////
	// pu grid components
	//
	Assert(slab->time.tria.use_count(), dealii::ExcNotInitialized());
	slab->time.pu.fe_info->dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);

	// FE configuration parameters for PU:
	// dG(0) in time

	// create pu fe system (time)
	{
		// create fe quadrature for support points
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time;
		{
			fe_quad_time = std::make_shared< dealii::QGauss<1> > (1);

// 					DTM::pout
// 						<< "FE time: (PU): "
// 						<< "created QGauss<1> quadrature"
// 						<< std::endl;
		}

		// create FE time
		{
			slab->time.pu.fe_info->fe =
			std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
				*fe_quad_time
			);
		}

// 			DTM::pout
// 				<< "slab->time.pu.fe_info->fe = "
// 				<< slab->time.pu.fe_info->fe->get_name()
// 				<< std::endl;
	}

	slab->time.pu.fe_info->mapping = std::make_shared< dealii::MappingQ<1> > (1);
}

template<int dim>
void
Grid<dim>::
initialize_vorticity_grid_components_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(parameter_set.use_count(), dealii::ExcNotInitialized());

	////////////////////////////////////////////////////////////////////////
	// space

	/////////////////////////
	// pu grid components
	//
	Assert(slab->space.tria.use_count(), dealii::ExcNotInitialized());
	slab->space.vorticity.fe_info->dof = std::make_shared< dealii::DoFHandler<dim> > (
		*slab->space.tria
	);

	// FE configuration parameters for PU:
	// cG(1) in space

	// create pu fe system
	{
		DTM::pout << "WARNING: vorticity so far only uses low order!" << std::endl;
		std::shared_ptr< dealii::Quadrature<1> > fe_quad;
		if ( !(parameter_set->
					fe.low.convection.space_type_support_points
					.compare("Gauss-Lobatto")) ) {

					fe_quad = std::make_shared< dealii::QGaussLobatto<1> > (
						(parameter_set->fe.low.convection.p + 1)
					);

					DTM::pout
						<< "FE: (low) convection b: "
						<< "created QGaussLobatto<1> quadrature"
						<< std::endl;
				}

		// create FE
		slab->space.vorticity.fe_info->fe = std::make_shared<dealii::FE_Q<dim>> (
			*fe_quad
		);

		DTM::pout
			<< "FE: (vorticity) created "
			<< slab->space.vorticity.fe_info->fe->get_name()
			<< std::endl;
	}

//	slab->space.vorticity.fe_info->constraints = std::make_shared< dealii::AffineConstraints<double> > ();

	slab->space.vorticity.fe_info->mapping = std::make_shared< dealii::MappingQ<dim> > (1);

	////////////////////////////////////////////////////////////////////////
	// time

	/////////////////////////
	// vorticity grid components
	//
	Assert(slab->time.tria.use_count(), dealii::ExcNotInitialized());
	slab->time.vorticity.fe_info->dof = std::make_shared< dealii::DoFHandler<1> > (
		*slab->time.tria
	);

	// FE configuration parameters for PU:
	// dG(0) in time

	// create vorticity fe system (time)
	{
		// create fe quadratures for support points
		std::shared_ptr< dealii::Quadrature<1> > fe_quad_time_vorticity;
		{
			if ( !(parameter_set->
				fe.low.convection.time_type_support_points
				.compare("Gauss")) ) {

				fe_quad_time_vorticity =
				std::make_shared< dealii::QGauss<1> > (
					(parameter_set->fe.low.convection.r + 1)
				);

					DTM::pout
						<< "FE time: (low) convection b: "
						<< "created QGauss<1> quadrature"
						<< std::endl;
			} else if ( !(parameter_set->
					fe.low.convection.time_type_support_points
					.compare("Gauss-Lobatto")) ){

				if (parameter_set->fe.low.convection.r < 1){
					fe_quad_time_vorticity =
							std::make_shared< QRightBox<1> > ();
						DTM::pout
							<< "FE time: (low) convection b: "
							<< "created QRightBox quadrature"
							<< std::endl;
				} else {
					fe_quad_time_vorticity =
							std::make_shared< dealii::QGaussLobatto<1> > (
									(parameter_set->fe.low.convection.r + 1)
							);

					DTM::pout
					<< "FE time: (low) convection b: "
					<< "created QGaussLobatto<1> quadrature"
					<< std::endl;
				}
			}
		}


		// create FE time
		{
			slab->time.vorticity.fe_info->fe =
			std::make_shared< dealii::FE_DGQArbitraryNodes<1> > (
				*fe_quad_time_vorticity
			);
		}

	}

	slab->time.vorticity.fe_info->mapping = std::make_shared< dealii::MappingQ<1> > (1);
}

template<int dim>
bool
Grid<dim>::
split_slab_in_time(
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

	unsigned int M = slab->time.tria->n_active_cells();

	//Check whether slab should be split. If not we are done
	if (M <= parameter_set->time.fluid.max_intervals_per_slab)
		return false;

	//Get step sizes of current time tria
	std::vector<double> step_sizes(M);
	unsigned int i = 0;
	for (auto &cell : slab->time.tria->active_cell_iterators()){
		step_sizes[i] = cell->bounding_box().side_length(0);
		i++;
	}

    unsigned int split_M = M/2;
    std::vector<double> step_sizes_1(split_M);
    std::vector<double> step_sizes_2(M-split_M);

    for (i = 0 ; i < split_M ; i++){
    	step_sizes_1[i] = step_sizes[i];
    }

    for (i = 0 ; i < (M-split_M) ; i++) {
    	step_sizes_2[i] = step_sizes[i+split_M];
    }

    // emplace a new slab element in front of the iterator
	slabs.emplace(
		slab
	);

	// init new slab ("space-time" tria)
	std::prev(slab)->t_m=slab->t_m;
	std::prev(slab)->t_n=slab->t_m + slab->tau_n()/2.;
	slab->t_m=std::prev(slab)->t_n;

	//init triangulations
	slab->time.tria = std::make_shared< dealii::Triangulation<1> > ();
	dealii::Point<1> p1_1(slab->t_m);
	dealii::Point<1> p2_1(slab->t_n);

	std::vector<std::vector<double>> spacing_1;
	spacing_1.push_back(step_sizes_1);
	dealii::GridGenerator::subdivided_hyper_rectangle(
			*slab->time.tria,
			spacing_1,
			p1_1,
			p2_1
	);


	//init previous tria
	std::prev(slab)->refine_in_time=false;

	// space
	std::prev(slab)->space.tria = std::make_shared< dealii::parallel::distributed::Triangulation<dim> > (
			mpi_comm,
			typename dealii::parallel::distributed::Triangulation<dim>::MeshSmoothing(
				dealii::parallel::distributed::Triangulation<dim>::smoothing_on_refinement
			),
			dealii::parallel::distributed::Triangulation<dim>::Settings::no_automatic_repartitioning
		);
//	slab->space.tria->reset_all_manifolds();
	std::prev(slab)->space.tria->copy_triangulation(*coarse_tria);
	std::prev(slab)->space.tria->load(slab->space.tria->get_p4est());

	std::prev(slab)->time.tria = std::make_shared< dealii::Triangulation<1> > ();
	dealii::Point<1> p1_2(std::prev(slab)->t_m);
	dealii::Point<1> p2_2(std::prev(slab)->t_n);

	std::vector<std::vector<double>> spacing_2;
	spacing_2.push_back(step_sizes_2);
	dealii::GridGenerator::subdivided_hyper_rectangle(
			*std::prev(slab)->time.tria,
			spacing_2,
			p1_2,
			p2_2
	);

	///////////////////////////////////////////////
	// assign low or high to primal and dual

	// primal = low / high
	if ( !parameter_set->fe.primal_order.compare("low") )
	{
		std::prev(slab)->space.primal.fe_info = std::prev(slab)->space.low.fe_info;
		std::prev(slab)->time.primal.fe_info = std::prev(slab)->time.low.fe_info;
	}
	else if ( !parameter_set->fe.primal_order.compare("high") )
	{
		std::prev(slab)->space.primal.fe_info = std::prev(slab)->space.high.fe_info;
		std::prev(slab)->time.primal.fe_info =std::prev(slab)->time.high.fe_info;
	}
	else
	{
		AssertThrow(false, dealii::ExcMessage("primal_order needs to be 'low' or 'high'."));
	}

	// dual = low / high / high-time
	if ( !parameter_set->fe.dual_order.compare("low") )
	{
		std::prev(slab)->space.dual.fe_info = std::prev(slab)->space.low.fe_info;
		std::prev(slab)->time.dual.fe_info = std::prev(slab)->time.low.fe_info;
	}
	else if ( !parameter_set->fe.dual_order.compare("high") )
	{
		std::prev(slab)->space.dual.fe_info = std::prev(slab)->space.high.fe_info;
		std::prev(slab)->time.dual.fe_info = std::prev(slab)->time.high.fe_info;
	}
	else if ( !parameter_set->fe.dual_order.compare("high-time") )
	{
		std::prev(slab)->space.dual.fe_info = std::prev(slab)->space.low.fe_info;
		std::prev(slab)->time.dual.fe_info = std::prev(slab)->time.high.fe_info;
	}
	else
	{
		AssertThrow(false, dealii::ExcMessage("dual_order needs to be 'low' or 'high' or 'high-time'."));
	}

	return true;
}

template<int dim>
void
Grid<dim>::
refine_slab_in_time(
	typename fluid::types::spacetime::dwr::slabs<dim>::iterator slab) {
	
	Assert(false,dealii::ExcMessage("This function should not be in use anymore!"));
}


/// Generate tria on each slab.
template<int dim>
void
Grid<dim>::
generate() {


	coarse_tria = std::make_shared< dealii::Triangulation<dim> > (
			typename dealii::Triangulation<dim>::MeshSmoothing(
				dealii::Triangulation<dim>::smoothing_on_refinement
			)
	);
	fluid::TriaGenerator<dim> tria_generator;
	tria_generator.generate(
		parameter_set->TriaGenerator,
		parameter_set->TriaGenerator_Options,
		coarse_tria
	);

	auto slab(this->slabs.begin());
	auto ends(this->slabs.end());
	slab->space.tria->copy_triangulation(*coarse_tria);
	dealii::GridGenerator::hyper_cube(
		*slab->time.tria,
		slab->t_m, slab->t_n,
		false
	);
	slab++;
	for (; slab != ends; ++slab) {

//		if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive"))
//		{
//			slab->space.tria = slabs.begin()->space.tria;
//		}
//		else{
			slab->space.tria->copy_triangulation(*coarse_tria);
//		}
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


	//do refinement
	{
		auto slab(slabs.begin());
		auto ends(slabs.end());


//	if (!parameter_set->dwr.refine_and_coarsen.spacetime.strategy.compare("adaptive"))
//	{
//		slabs.begin()->space.tria->refine_global(space_n);
//		for (; slab != ends; ++slab) {
//			slab->time.tria->refine_global(time_n);
//		}
//	}
//	else {
		for (; slab != ends; ++slab) {
			slab->time.tria->refine_global(time_n);
			slab->space.tria->refine_global(space_n);
		}
	}

	//check if slabs should be split
    //depending on parameters this might have to happen multiple times
	bool was_split = true;
	while (was_split){
		auto slab(slabs.begin());
		auto ends(slabs.end());
        was_split = false;
		for (; slab != ends; ++slab) {
			if (split_slab_in_time(slab))
				was_split = true;
		}
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
set_dirichlet_function(std::shared_ptr< dealii::TensorFunction<1,dim,double> > fun){
	dirichlet_function = fun;
}


template<int dim>
void
Grid<dim>::
distribute() {
	// distribute is now being performed slab wise
	Assert(false, dealii::ExcNotImplemented());
}

template<int dim>
void
Grid<dim>::
distribute_low_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	////////////////////////////////////////////////////////////////////////
	// space
	// distribute low dofs and create constraints
	{
		Assert(slab->space.low.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->space.low.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->space.low.fe_info->dof->distribute_dofs(*(slab->space.low.fe_info->fe));

		dealii::DoFRenumbering::component_wise(*slab->space.low.fe_info->dof);

		////////////////////////////////////////////////////////////////////
		// create partitionings for the initialisation of vectors
		//

		// get all dofs componentwise
		std::vector< dealii::types::global_dof_index > dofs_per_component(
			slab->space.low.fe_info->dof->get_fe_collection().n_components(), 0
		);

		dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
			*slab->space.low.fe_info->dof,
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

		DTM::pout << "\tn_dofs (low)       = "
			<< slab->space.low.fe_info->dof->n_dofs()
			<< std::endl;

		// create dof partitionings

		// dof partitioning
		slab->space.low.fe_info->locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (slab->space.low.fe_info->dof->locally_owned_dofs());

		slab->space.low.fe_info->locally_relevant_dofs =
					std::make_shared< dealii::IndexSet > ();

		dealii::DoFTools::extract_locally_relevant_dofs(*slab->space.low.fe_info->dof,
													    *slab->space.low.fe_info->locally_relevant_dofs);


		// setup constraints (e.g. hanging nodes)
		Assert(
			slab->space.low.fe_info->constraints.use_count(),
			dealii::ExcNotInitialized()
		);
		{
			slab->space.low.fe_info->constraints->clear();
			slab->space.low.fe_info->constraints->reinit(*slab->space.low.fe_info->locally_relevant_dofs);


			slab->space.low.fe_info->hanging_node_constraints->clear();
			slab->space.low.fe_info->hanging_node_constraints->reinit(*slab->space.low.fe_info->locally_relevant_dofs);


			slab->space.low.fe_info->initial_constraints->clear();
			slab->space.low.fe_info->initial_constraints->reinit(*slab->space.low.fe_info->locally_relevant_dofs);


			interpolate_dirichlet_bc(slab->space.low.fe_info->dof,
									 slab->space.low.fe_info->constraints,
									 slab->t_m,false);


			interpolate_dirichlet_bc(slab->space.low.fe_info->dof,
									 slab->space.low.fe_info->initial_constraints,
									 slab->t_m, true);


			Assert(slab->space.low.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.low.fe_info->dof,
				*slab->space.low.fe_info->constraints
			);
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.low.fe_info->dof,
				*slab->space.low.fe_info->hanging_node_constraints
			);
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.low.fe_info->dof,
				*slab->space.low.fe_info->initial_constraints
			);
			slab->space.low.fe_info->constraints->close();
			slab->space.low.fe_info->hanging_node_constraints->close();
			slab->space.low.fe_info->initial_constraints->close();

		}
	}

	////////////////////////////////////////////////////////////////////////
	// time
	// distribute low dofs
	{
		Assert(slab->time.low.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.low.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->time.low.fe_info->dof->distribute_dofs(*(slab->time.low.fe_info->fe));
	}
}

template<int dim>
void
Grid<dim>::
distribute_high_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	////////////////////////////////////////////////////////////////////////
	// space
	// distribute dual dofs and create constraints
	{
		Assert(slab->space.high.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->space.high.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->space.high.fe_info->dof->distribute_dofs(*(slab->space.high.fe_info->fe));

		dealii::DoFRenumbering::component_wise(*slab->space.high.fe_info->dof);

		////////////////////////////////////////////////////////////////////
		// create partitionings for the initialisation of vectors
		//

		// get all dofs componentwise
		std::vector< dealii::types::global_dof_index > dofs_per_component(
			slab->space.high.fe_info->dof->get_fe_collection().n_components(), 0
		);

		dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
			*slab->space.high.fe_info->dof,
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

		DTM::pout << "\tn_dofs (high)         = "
			<< slab->space.high.fe_info->dof->n_dofs()
			<< std::endl;

		// create dof partitionings

		// dof partitioning
		slab->space.high.fe_info->locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (slab->space.high.fe_info->dof->locally_owned_dofs());

		slab->space.high.fe_info->locally_relevant_dofs =
					std::make_shared< dealii::IndexSet > ();

		dealii::DoFTools::extract_locally_relevant_dofs(*slab->space.high.fe_info->dof,
				 	 	 	 	 	 	 	 	 	 	*slab->space.high.fe_info->locally_relevant_dofs);


		// setup constraints (e.g. hanging nodes)
		Assert(
			slab->space.high.fe_info->constraints.use_count(),
			dealii::ExcNotInitialized()
		);
		{
			slab->space.high.fe_info->constraints->clear();
			slab->space.high.fe_info->constraints->reinit(*slab->space.high.fe_info->locally_relevant_dofs);

			slab->space.high.fe_info->hanging_node_constraints->clear();
			slab->space.high.fe_info->hanging_node_constraints->reinit(*slab->space.high.fe_info->locally_relevant_dofs);


			slab->space.high.fe_info->initial_constraints->clear();
			slab->space.high.fe_info->initial_constraints->reinit(*slab->space.high.fe_info->locally_relevant_dofs);
			interpolate_dirichlet_bc(slab->space.high.fe_info->dof,
					                 slab->space.high.fe_info->constraints,
									 slab->t_m,false);

			interpolate_dirichlet_bc(slab->space.high.fe_info->dof,
									 slab->space.high.fe_info->initial_constraints,
									 slab->t_m,true);

			Assert(slab->space.high.fe_info->dof.use_count(), dealii::ExcNotInitialized());

			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.high.fe_info->dof,
				*slab->space.high.fe_info->constraints
			);
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.high.fe_info->dof,
				*slab->space.high.fe_info->hanging_node_constraints
			);
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.high.fe_info->dof,
				*slab->space.high.fe_info->initial_constraints
			);
			slab->space.high.fe_info->constraints->close();
			slab->space.high.fe_info->hanging_node_constraints->close();
			slab->space.high.fe_info->initial_constraints->close();
		}
	}

	////////////////////////////////////////////////////////////////////////
	// time
	// distribute high dofs
	{
		Assert(slab->time.high.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.high.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->time.high.fe_info->dof->distribute_dofs(*(slab->time.high.fe_info->fe));
	}
}

template<int dim>
void
Grid<dim>::
distribute_pu_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	////////////////////////////////////////////////////////////////////////
	// space
	// distribute pu dofs and create constraints
	{
		Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->space.pu.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->space.pu.fe_info->dof->distribute_dofs(*(slab->space.pu.fe_info->fe));

		DTM::pout << "\tn_dofs (PU)         = "
			<< slab->space.pu.fe_info->dof->n_dofs()
			<< std::endl;

		// create dof partitionings

		// dof partitioning
		slab->space.pu.fe_info->locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (slab->space.pu.fe_info->dof->locally_owned_dofs());

		slab->space.pu.fe_info->locally_relevant_dofs =
			std::make_shared< dealii::IndexSet > ();

		dealii::DoFTools::extract_locally_relevant_dofs(*slab->space.pu.fe_info->dof,
				*slab->space.pu.fe_info->locally_relevant_dofs);

		// setup constraints (e.g. hanging nodes)
		Assert(
			slab->space.pu.fe_info->constraints.use_count(),
			dealii::ExcNotInitialized()
		);
		{
			slab->space.pu.fe_info->constraints->clear();
			slab->space.pu.fe_info->constraints->reinit(*slab->space.pu.fe_info->locally_relevant_dofs);

			Assert(slab->space.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
			dealii::DoFTools::make_hanging_node_constraints(
				*slab->space.pu.fe_info->dof,
				*slab->space.pu.fe_info->constraints
			);

			slab->space.pu.fe_info->constraints->close();
		}
	}

	////////////////////////////////////////////////////////////////////////
	// time
	// distribute pu dofs
	{
		Assert(slab->time.pu.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.pu.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->time.pu.fe_info->dof->distribute_dofs(*(slab->time.pu.fe_info->fe));
	}
}


template<int dim>
void
Grid<dim>::
distribute_vorticity_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	////////////////////////////////////////////////////////////////////////
	// space
	// distribute pu dofs and create constraints
	{
		Assert(slab->space.vorticity.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->space.vorticity.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->space.vorticity.fe_info->dof->distribute_dofs(*(slab->space.vorticity.fe_info->fe));

		DTM::pout << "\tn_dofs (vorticity)         = "
			<< slab->space.vorticity.fe_info->dof->n_dofs()
			<< std::endl;

		// create dof partitionings

		// dof partitioning
		slab->space.vorticity.fe_info->locally_owned_dofs =
			std::make_shared< dealii::IndexSet > (slab->space.vorticity.fe_info->dof->locally_owned_dofs());

		slab->space.vorticity.fe_info->locally_relevant_dofs =
			std::make_shared< dealii::IndexSet > ();

		dealii::DoFTools::extract_locally_relevant_dofs(*slab->space.vorticity.fe_info->dof,
				*slab->space.vorticity.fe_info->locally_relevant_dofs);

//		// setup constraints (e.g. hanging nodes)
//		Assert(
//			slab->space.vorticity.fe_info->constraints.use_count(),
//			dealii::ExcNotInitialized()
//		);
//		{
//			slab->space.vorticity.fe_info->constraints->clear();
//			slab->space.vorticity.fe_info->constraints->reinit(*slab->space.pu.fe_info->locally_relevant_dofs);
//
//			Assert(slab->space.vorticity.fe_info->dof.use_count(), dealii::ExcNotInitialized());
//			dealii::DoFTools::make_hanging_node_constraints(
//				*slab->space.vorticity.fe_info->dof,
//				*slab->space.vorticity.fe_info->constraints
//			);
//
//			slab->space.vorticity.fe_info->constraints->close();
//		}
	}

	////////////////////////////////////////////////////////////////////////
	// time
	// distribute vorticity dofs
	{
		Assert(slab->time.vorticity.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		Assert(slab->time.vorticity.fe_info->fe.use_count(), dealii::ExcNotInitialized());
		slab->time.vorticity.fe_info->dof->distribute_dofs(*(slab->time.vorticity.fe_info->fe));
	}
}
template<int dim>
void
Grid<dim>::
interpolate_dirichlet_bc(std::shared_ptr<dealii::DoFHandler<dim>> dof,
			                              std::shared_ptr<dealii::AffineConstraints<double>> constraints,
										  double tm,
										  bool inhom){




	auto component_mask_convection =
	dealii::ComponentMask (
			(dim+1), true
	);
	component_mask_convection.set(dim,false);

	if ( inhom ){
		dirichlet_function->set_time(tm);
		std::shared_ptr< dealii::VectorFunctionFromTensorFunction<dim>>
		bc_fun = std::make_shared< dealii::VectorFunctionFromTensorFunction<dim> > (
					*dirichlet_function,
					0, (dim+1)
		);

		bc_fun ->set_time(tm);
		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c3+
						fluid::types::space::boundary_id::prescribed_convection_c2+
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				*bc_fun,
				*constraints,
				component_mask_convection
		);
		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c2+
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				*bc_fun,
				*constraints,
				component_mask_convection
		);
		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				*bc_fun,
				*constraints,
				component_mask_convection
		);
	}
	else {

		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c3+
						fluid::types::space::boundary_id::prescribed_convection_c2+
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				dealii::ZeroFunction<dim>(dim+1),
				*constraints,
				component_mask_convection
		);
		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c2+
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				dealii::ZeroFunction<dim>(dim+1),
				*constraints,
				component_mask_convection
		);
		dealii::VectorTools::interpolate_boundary_values(
				*dof,
				static_cast< dealii::types::boundary_id> (
						fluid::types::space::boundary_id::prescribed_convection_c1
				),
				dealii::ZeroFunction<dim>(dim+1),
				*constraints,
				component_mask_convection
		);
	}
	dealii::VectorTools::interpolate_boundary_values(
			*dof,
			static_cast< dealii::types::boundary_id> (
					fluid::types::space::boundary_id::prescribed_no_slip
			),
			dealii::ZeroFunction<dim>(dim+1),
			*constraints,
			component_mask_convection
	);
	dealii::VectorTools::interpolate_boundary_values(
			*dof,
			static_cast< dealii::types::boundary_id> (
					fluid::types::space::boundary_id::prescribed_no_slip+
					fluid::types::space::boundary_id::prescribed_obstacle
			),
			dealii::ZeroFunction<dim>(dim+1),
			*constraints,
			component_mask_convection
	);
}

template<int dim>
void
Grid<dim>::
create_sparsity_pattern_primal_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> L_spacetime,
		std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> L_space)  {
	////////////////////////////////////////////////////////////////////////
	// space
	{
		// create sparsity pattern
		Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());
//		slab->space.primal.sp_block_L = std::make_shared< dealii::SparsityPattern >(); // std::make_shared< dealii::BlockSparsityPattern >();
		slab->space.primal.sp_L = std::make_shared< dealii::SparsityPattern >();

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

				// (Dyn.) Stokes: saddle-point block structure:
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

			// stokes: "L" block matrix
			dealii::DynamicSparsityPattern dsp(slab->space.primal.fe_info->dof->n_dofs()); //dealii::BlockDynamicSparsityPattern dsp;
//			dsp.reinit(
//				slab->space.primal.fe_info->dof->n_dofs() //*slab->space.primal.fe_info->partitioning_locally_owned_dofs
//			);

			dealii::DoFTools::make_sparsity_pattern(
				*slab->space.primal.fe_info->dof,
				coupling_block_L,
				dsp,
				*slab->space.primal.fe_info->constraints,
				true // true => keep constrained entry in sparsity pattern
 				,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
			);

			//copy undistributed pattern for spacetime???
			slab->space.primal.sp_L->copy_from(dsp);

//			dealii::SparsityTools::distribute_sparsity_pattern(
//					dsp,
//					*slab->space.primal.fe_info->locally_owned_dofs,
//					mpi_comm,
//					*slab->space.primal.fe_info->locally_relevant_dofs
//			);

//			dsp.compress();
//
//			L_space->reinit(*slab->space.primal.fe_info->locally_owned_dofs,
//			        		*slab->space.primal.fe_info->locally_owned_dofs,
//							dsp);
//			Assert(
//				slab->space.primal.sp_block_L.use_count(),
//				dealii::ExcNotInitialized()
//			);
//			slab->space.primal.sp_block_L->copy_from(dsp);
		}
	}

	//pattern for initial constraints (for projection only)
	{
			// create sparsity pattern
			Assert(slab->space.primal.fe_info->dof.use_count(), dealii::ExcNotInitialized());

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

					// (Dyn.) Stokes: saddle-point block structure:
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

				// stokes: "L" block matrix
				dealii::DynamicSparsityPattern dsp(slab->space.primal.fe_info->dof->n_dofs()); //dealii::BlockDynamicSparsityPattern dsp;

				dealii::DoFTools::make_sparsity_pattern(
					*slab->space.primal.fe_info->dof,
					coupling_block_L,
					dsp,
					*slab->space.primal.fe_info->initial_constraints,
					false // true => keep constrained entry in sparsity pattern
	 				,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
				);

				dealii::SparsityTools::distribute_sparsity_pattern(
						dsp,
						*slab->space.primal.fe_info->locally_owned_dofs,
						mpi_comm,
						*slab->space.primal.fe_info->locally_relevant_dofs
				);

				L_space->reinit(*slab->space.primal.fe_info->locally_owned_dofs,
				        		*slab->space.primal.fe_info->locally_owned_dofs,
								dsp);
			}
		}
	////////////////////////////////////////////////////////////////////////////
	// space-time
	//




	// primal sparsity pattern (downwind test function in time)
	{
		// sparsity pattern block for space-time cell-volume integrals
		// on the slab
		//
		// Step 1: transfer
		// (Dyn.) Stokes: saddle-point block structure:
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

		// Step 2:

		dealii::DynamicSparsityPattern dsp(
			slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs(), // test
			slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->dof->n_dofs()  // trial
		);

		{
			// sparsity pattern block for coupling between time cells
			// within the slab by downwind the test functions and a jump
			// term (trace operator) on the trial functions
			{
				Assert(
					slab->space.primal.sp_L.use_count(),
					dealii::ExcNotInitialized()
				);

				// TODO: use sp_L, once you have more time derivatives!
				// NOTE: here we have only ((phi * \partial_t b)).
				auto cell{slab->space.primal.sp_L->begin()};
				auto endc{slab->space.primal.sp_L->end()};

				dealii::types::global_dof_index offset{0};
				for ( ; cell != endc; ++cell) {
					for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {

						offset =
								slab->space.primal.fe_info->dof->n_dofs() *
								nn * slab->time.primal.fe_info->fe->dofs_per_cell;

						if (nn)
							for (unsigned int i{0}; i < slab->time.primal.fe_info->fe->dofs_per_cell; ++i)
								for (unsigned int j{0}; j < slab->time.primal.fe_info->fe->dofs_per_cell; ++j) {
									dsp.add(
											// downwind test
											cell->row() + offset
											+ slab->space.primal.fe_info->dof->n_dofs() * i,
											// trial functions of K^-
											cell->column() + offset
											- slab->space.primal.fe_info->dof->n_dofs() * slab->time.primal.fe_info->fe->dofs_per_cell
											+ slab->space.primal.fe_info->dof->n_dofs() * j
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
							slab->space.primal.fe_info->dof->n_dofs() *
							nn * slab->time.primal.fe_info->fe->dofs_per_cell;

						for (unsigned int i{0}; i < slab->time.primal.fe_info->fe->dofs_per_cell; ++i)
						for (unsigned int j{0}; j < slab->time.primal.fe_info->fe->dofs_per_cell; ++j) {
							dsp.add(
								// test
								cell->row() + offset
								+ slab->space.primal.fe_info->dof->n_dofs() * i,
								// trial
								cell->column() + offset
								+ slab->space.primal.fe_info->dof->n_dofs() * j
							);
						}
					}
				}
			}
		}


		Assert(slab->spacetime.primal.locally_owned_dofs.use_count(),
				dealii::ExcNotInitialized());

		Assert(slab->spacetime.primal.locally_relevant_dofs.use_count(),
						dealii::ExcNotInitialized());
//		{
//		  static dealii::Utilities::MPI::CollectiveMutex      mutex;
//		  dealii::Utilities::MPI::CollectiveMutex::ScopedLock lock(mutex, mpi_comm);
//		  // [ critical code to be guarded]

		dealii::SparsityTools::distribute_sparsity_pattern(
				dsp,
				*slab->spacetime.primal.locally_owned_dofs,
				mpi_comm,
				*slab->spacetime.primal.locally_relevant_dofs
		);

		L_spacetime->reinit(*slab->spacetime.primal.locally_owned_dofs,
        		  *slab->spacetime.primal.locally_owned_dofs,
				  dsp);
//		}
	}
	slab->space.primal.sp_L=nullptr;
}

template<int dim>
void
Grid<dim>::
create_sparsity_pattern_dual_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> L) {
	////////////////////////////////////////////////////////////////////////
	// space
	{
		// create sparsity pattern
		Assert(slab->space.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
		slab->space.dual.sp_block_L = std::make_shared< dealii::SparsityPattern >();
		slab->space.dual.sp_L = std::make_shared< dealii::SparsityPattern >();
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

				// (Dyn.) Stokes: saddle-point block structure:
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

			// stokes: "L" block matrix
			dealii::DynamicSparsityPattern dsp(slab->space.dual.fe_info->dof->n_dofs());;
//			dsp.reinit(
//				*slab->space.dual.fe_info->partitioning_locally_owned_dofs
//			);

			dealii::DoFTools::make_sparsity_pattern(
				*slab->space.dual.fe_info->dof,
				coupling_block_L,
				dsp,
				*slab->space.dual.fe_info->constraints,
				true // true => keep constrained entry in sparsity pattern
				,dealii::Utilities::MPI::this_mpi_process(mpi_comm)
			);

			slab->space.dual.sp_L->copy_from(dsp);

			dealii::SparsityTools::distribute_sparsity_pattern(
					dsp,
					*slab->space.dual.fe_info->locally_owned_dofs,
					mpi_comm,
					*slab->space.dual.fe_info->locally_relevant_dofs);
//			dsp.compress();

			Assert(
				slab->space.dual.sp_block_L.use_count(),
				dealii::ExcNotInitialized()
			);
			slab->space.dual.sp_block_L->copy_from(dsp);
		}
	}

	////////////////////////////////////////////////////////////////////////////
	// space-time
	//

	// dual sparsity pattern (upwind trial function in time)
	{
		// sparsity pattern block for space-time cell-volume integrals
		// on the slab
		//
		// Step 1: transfer
		// (Dyn.) Stokes: saddle-point block structure:
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

		// Step 2:

		dealii::DynamicSparsityPattern dual_dsp(
			slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs(), // test
			slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->dof->n_dofs()  // trial
		);

		// get all dofs componentwise
		std::vector< dealii::types::global_dof_index > dofs_per_component(
			slab->space.dual.fe_info->dof->get_fe_collection().n_components(), 0
		);

		dofs_per_component = dealii::DoFTools::count_dofs_per_fe_component(
			*slab->space.dual.fe_info->dof,
			true
		);

		// set specific values of dof counts
		dealii::types::global_dof_index N_b; // convection

		// dof count convection: vector-valued primitive FE
		N_b = 0;
		for (unsigned int d{0}; d < dim; ++d) {
			N_b += dofs_per_component[d];
		}

		{
			// sparsity pattern block for coupling between time cells
			// within the slab by upwinding the trial functions and a jump
			// term (trace operator) on the test functions
			{
				Assert(
					slab->space.dual.sp_block_L.use_count(),
					dealii::ExcNotInitialized()
				);

				// TODO: use sp_L, once you have more time derivatives!
				// NOTE: here we have only ((phi * \partial_t b)).
				auto cell{slab->space.dual.sp_L->begin()}; // block(0,0)
				auto endc{slab->space.dual.sp_L->end()}; // block(0,0)

				dealii::types::global_dof_index offset{0};
				for ( ; cell != endc; ++cell) {
					for (dealii::types::global_dof_index nn{0};
							nn < slab->time.tria->n_global_active_cells(); ++nn) {

						offset =
								slab->space.dual.fe_info->dof->n_dofs() *
								nn * slab->time.dual.fe_info->fe->dofs_per_cell;

						if (nn)
							for (unsigned int i{0}; i < slab->time.dual.fe_info->fe->dofs_per_cell; ++i)
								for (unsigned int j{0}; j < slab->time.dual.fe_info->fe->dofs_per_cell; ++j) {
									dual_dsp.add(
											// test
											cell->row() + offset
											- slab->space.dual.fe_info->dof->n_dofs() * slab->time.dual.fe_info->fe->dofs_per_cell
											+ slab->space.dual.fe_info->dof->n_dofs() * i,
											// trial
											cell->column() + offset
											+ slab->space.dual.fe_info->dof->n_dofs() * j
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
							slab->space.dual.fe_info->dof->n_dofs() *
							nn * slab->time.dual.fe_info->fe->dofs_per_cell;

						for (unsigned int i{0}; i < slab->time.dual.fe_info->fe->dofs_per_cell; ++i)
						for (unsigned int j{0}; j < slab->time.dual.fe_info->fe->dofs_per_cell; ++j) {
							dual_dsp.add(
								// test
								cell->row() + offset
								+ slab->space.dual.fe_info->dof->n_dofs() * i,
								// trial
								cell->column() + offset
								+ slab->space.dual.fe_info->dof->n_dofs() * j
							);
						}
					}
				}
			}
		}

		dealii::SparsityTools::distribute_sparsity_pattern(
				dual_dsp,
				*slab->spacetime.dual.locally_owned_dofs,
				mpi_comm,
				*slab->spacetime.dual.locally_relevant_dofs
		);

		L->reinit(
				*slab->spacetime.dual.locally_owned_dofs,
				*slab->spacetime.dual.locally_owned_dofs,
				dual_dsp
		);

	}
}

template<int dim>
void
Grid<dim>::
clear_primal_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(
		slab->time.primal.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	// space
	DTM::pout << "         slab->space.primal.fe_info->dof->memory_consumption() = " << std::right << std::setw(20) << (slab->space.primal.fe_info->dof->memory_consumption() / 1000) << " KB" << std::endl;
	DTM::pout << "         slab->space.primal.fe_info->fe->memory_consumption()  = " << std::setw(20) << (slab->space.primal.fe_info->fe->memory_consumption()  / 1000) << " KB" << std::endl;
	DTM::pout << "         slab->space.primal.fe_info->constraints->memory_consumption()  = " << std::setw(20) << (slab->space.primal.fe_info->constraints->memory_consumption()  / 1000) << " KB" << std::endl;
	slab->space.primal.sp_block_L = nullptr;
	slab->space.primal.sp_L = nullptr;

	// time
	DTM::pout << "         slab->time.primal.fe_info->dof->memory_consumption()  = " << std::setw(20) << (slab->time.primal.fe_info->dof->memory_consumption() / 1000) << " KB" << std::endl;
	DTM::pout << "         slab->time.primal.fe_info->fe->memory_consumption()   = " << std::setw(20) << (slab->time.primal.fe_info->fe->memory_consumption() / 1000)  << " KB" << std::endl;

	// space-time
	slab->spacetime.primal.constraints = nullptr;
	slab->spacetime.primal.sp = nullptr;
}

template<int dim>
void
Grid<dim>::
clear_dual_on_slab(const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
	Assert(
		slab->time.dual.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	// space
	slab->space.dual.sp_block_L = nullptr;
	slab->space.dual.sp_L = nullptr;

	// space-time
	slab->spacetime.dual.constraints = nullptr;
	slab->spacetime.dual.sp = nullptr;
}

} // namespaces

#include "Grid.inst.in"
