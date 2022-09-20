/**
 * @file ErrorEstimator.tpl.cc
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 *
 * @date 2022-09-20, clean up and semi-mixed order, JR
 * @date 2022-09-07, parallelization, JPT
 * @date 2022-05-16, added NSE nonlinearity, JR
 * @date 2022-05-02, added to fluid, JR
 * @date 2022-02-07, started working on Stokes, JR
 * @date 2020-02-07, back-merge from dwr-stokes-condiffrea, UK
 * @date 2020-02-06, new implementation with IE/II in space-time, UK
 * @date 2020-01-09, included from dwr-diffusion to dwr-condiffrea, UK
 * @date 2019-11-11, add primal AND dual residual for error indicators, MPB
 * @date 2018-11-15, add inhomogeneous Neumann, MPB, UK
 * @date 2018-07-19, bugfix: irregular face, UK, MPB
 * @date 2018-03-16, ErrorEstimator class for heat (final), UK, MPB
 * @date 2018-03-13, new development ErrorEstimator class for heat (begin), UK, MPB
 * @date 2018-03-13, fork from DTM++/dwr-poisson, UK
 *
 * @date 2017-11-08, ErrorEstimator class (Poisson), UK, MPB
 * @date 2016-08-16, ErrorEstimator class (Poisson), UK
 * @date 2016-08-11, Poisson / DWR from deal.II/step-14 and DTM++, UK
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
#include <fluid/ErrorEstimator/ErrorEstimator.tpl.hh>
#include <fluid/types/boundary_id.hh>

// DTM++ includes
#include <DTM++/base/LogStream.hh>

// DEAL.II includes
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/numerics/vector_tools.h>

#include <ideal.II/dofs/SlabDoFTools.hh>

// namespace dwr {
namespace fluid {

namespace cGp_dGr { // primal
namespace cGq_dGs { // dual

namespace Assembly {

namespace Scratch {

/// (Struct-) Constructor.
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(
	// space
	const dealii::DoFHandler<dim> &dof_high,
	const dealii::DoFHandler<dim> &dof_pu,
	const dealii::FiniteElement<dim> &fe_high,
	const dealii::FiniteElement<dim> &fe_pu,
	const dealii::Mapping<dim> &mapping,
	const dealii::Quadrature<dim> &quad,
	const dealii::UpdateFlags &uflags) :
	fe_values_high(mapping, fe_high, quad, uflags),
	fe_values_pu(mapping, fe_pu, quad, uflags),
	dof_high(dof_high),
	dof_pu(dof_pu),
	local_dof_indices_high(fe_high.dofs_per_cell),
	local_dof_indices_pu(fe_pu.dofs_per_cell),
	//
	chi(fe_pu.dofs_per_cell),
	grad_chi(fe_pu.dofs_per_cell),
	//
	phi_convection(fe_high.dofs_per_cell),
	grad_phi_convection(fe_high.dofs_per_cell),
	symgrad_phi_convection(fe_high.dofs_per_cell),
	div_phi_convection(fe_high.dofs_per_cell),
	phi_pressure(fe_high.dofs_per_cell),
	//
	local_u_kh_m(fe_high.dofs_per_cell),
	local_u_kh_p(fe_high.dofs_per_cell),
	local_u_k_m(fe_high.dofs_per_cell),
	local_u_k_p(fe_high.dofs_per_cell),
	local_z_p(fe_high.dofs_per_cell),
	local_z_k_rho_k_p(fe_high.dofs_per_cell),
	local_z_k_rho_h_p(fe_high.dofs_per_cell),
	local_z_kh_p(fe_high.dofs_per_cell),
	//
	local_u_kh_tq(fe_high.dofs_per_cell),
	local_dtu_kh_tq(fe_high.dofs_per_cell),
	local_u_k_tq(fe_high.dofs_per_cell),
	local_dtu_k_tq(fe_high.dofs_per_cell),
	//
	local_z_tq(fe_high.dofs_per_cell),
	local_z_k_rho_k_tq(fe_high.dofs_per_cell),
	local_z_k_rho_h_tq(fe_high.dofs_per_cell),
	local_z_kh_tq(fe_high.dofs_per_cell),
	value_viscosity(0.),
	value_z_z_k_pressure(0.),
	value_z_k_z_kh_pressure(0.),
	value_div_z_z_k_convection(0.),
	value_div_z_k_z_kh_convection(0.),
	value_u_k_pressure(0.),
	value_u_kh_pressure(0.),
	value_div_u_k_convection(0.),
	value_div_u_kh_convection(0.),
	JxW(0.),
	q(0),
	d(0),
	j(0)
{}


template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &scratch) :
	fe_values_high(
		scratch.fe_values_high.get_mapping(),
		scratch.fe_values_high.get_fe(),
		scratch.fe_values_high.get_quadrature(),
		scratch.fe_values_high.get_update_flags()
	),
	fe_values_pu(
		scratch.fe_values_pu.get_mapping(),
		scratch.fe_values_pu.get_fe(),
		scratch.fe_values_pu.get_quadrature(),
		scratch.fe_values_pu.get_update_flags()
	),
	dof_high(scratch.dof_high),
	dof_pu(scratch.dof_pu),
	local_dof_indices_high(scratch.local_dof_indices_high),
	local_dof_indices_pu(scratch.local_dof_indices_pu),
	//
	chi(scratch.chi),
	grad_chi(scratch.grad_chi),
	//
	phi_convection(scratch.phi_convection),
	grad_phi_convection(scratch.grad_phi_convection),
	symgrad_phi_convection(scratch.symgrad_phi_convection),
	div_phi_convection(scratch.div_phi_convection),
	phi_pressure(scratch.phi_pressure),
	//
	local_u_kh_m(scratch.local_u_kh_m),
	local_u_kh_p(scratch.local_u_kh_p),
	local_u_k_m(scratch.local_u_k_m),
	local_u_k_p(scratch.local_u_k_p),
	local_z_p(scratch.local_z_p),
	local_z_k_rho_k_p(scratch.local_z_k_rho_k_p),
	local_z_k_rho_h_p(scratch.local_z_k_rho_h_p),
	local_z_kh_p(scratch.local_z_kh_p),
	//
	local_u_kh_tq(scratch.local_u_kh_tq),
	local_dtu_kh_tq(scratch.local_dtu_kh_tq),
	local_u_k_tq(scratch.local_u_k_tq),
	local_dtu_k_tq(scratch.local_dtu_k_tq),
	//
	local_z_tq(scratch.local_z_tq),
	local_z_k_rho_k_tq(scratch.local_z_k_rho_k_tq),
	local_z_k_rho_h_tq(scratch.local_z_k_rho_h_tq),
	local_z_kh_tq(scratch.local_z_kh_tq),
	//
	value_viscosity(scratch.value_viscosity),
	//
	value_jump_u_kh_convection(scratch.value_jump_u_kh_convection),
	value_jump_u_k_convection(scratch.value_jump_u_k_convection),
	value_z_z_k_convection(scratch.value_z_z_k_convection),
	value_z_k_z_kh_convection(scratch.value_z_k_z_kh_convection),
	value_z_z_k_pressure(scratch.value_z_z_k_pressure),
	value_z_k_z_kh_pressure(scratch.value_z_k_z_kh_pressure),
	//
	value_grad_z_z_k_convection(scratch.value_grad_z_z_k_convection),
	value_grad_z_k_z_kh_convection(scratch.value_grad_z_k_z_kh_convection),
	value_symgrad_z_z_k_convection(scratch.value_symgrad_z_z_k_convection),
	value_symgrad_z_k_z_kh_convection(scratch.value_symgrad_z_k_z_kh_convection),
	value_div_z_z_k_convection(scratch.value_div_z_z_k_convection),
	value_div_z_k_z_kh_convection(scratch.value_div_z_k_z_kh_convection),
	//
	value_grad_u_k_convection(scratch.value_grad_u_k_convection),
	value_grad_u_kh_convection(scratch.value_grad_u_kh_convection),
	value_symgrad_u_k_convection(scratch.value_symgrad_u_k_convection),
	value_symgrad_u_kh_convection(scratch.value_symgrad_u_kh_convection),
	value_u_k_pressure(scratch.value_u_k_pressure),
	value_u_kh_pressure(scratch.value_u_kh_pressure),
	value_u_k_convection(scratch.value_u_k_convection),
	value_u_kh_convection(scratch.value_u_kh_convection),
	value_div_u_k_convection(scratch.value_div_u_k_convection),
	value_div_u_kh_convection(scratch.value_div_u_kh_convection),
	//
	JxW(scratch.JxW),
	q(scratch.q),
	d(scratch.d),
	j(scratch.j) {
}


template<int dim>
ErrorEstimates<dim>::ErrorEstimates(
	const dealii::DoFHandler<dim>    &dof_high,
	const dealii::DoFHandler<dim>    &dof_pu,
	const dealii::FiniteElement<dim> &fe_high,
	const dealii::FiniteElement<dim> &fe_pu,
	const dealii::Mapping<dim>       &mapping,
	const dealii::Quadrature<dim>    &quad_cell,
	const dealii::UpdateFlags        &uflags_cell) :
	cell(dof_high, dof_pu, fe_high, fe_pu, mapping, quad_cell, uflags_cell) {
}

template<int dim>
ErrorEstimates<dim>::ErrorEstimates(const ErrorEstimates &scratch) :
	cell(scratch.cell) {
}

}

namespace CopyData {
/// (Struct-) Constructor
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const dealii::FiniteElement<dim> &fe) :
	local_eta_h_vector(fe.dofs_per_cell),
	local_eta_k_vector(fe.dofs_per_cell),
	local_dof_indices_pu(fe.dofs_per_cell) {
}

/// (Struct-) Copy constructor
template<int dim>
ErrorEstimateOnCell<dim>::ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata) :
	local_eta_h_vector(copydata.local_eta_h_vector),
	local_eta_k_vector(copydata.local_eta_k_vector),
	local_dof_indices_pu(copydata.local_dof_indices_pu) {
}

template<int dim>
ErrorEstimates<dim>::ErrorEstimates(const dealii::FiniteElement<dim> &fe) :
    cell(fe) {
}

template<int dim>
ErrorEstimates<dim>::ErrorEstimates(const ErrorEstimates &copydata) :
	cell(copydata.cell) {
}

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////



template<int dim>
void
ErrorEstimator<dim>::
estimate_on_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &u,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &um,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &z,

	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_s,
	const typename DTM::types::storage_data_trilinos_vectors<1>::iterator &eta_t
) {
	typedef
		dealii::
		FilteredIterator<const typename dealii::DoFHandler<dim>::active_cell_iterator>
		CellFilter;

	n_slab++;
	primal.locally_owned_dofs    = slab->space.primal.fe_info->locally_owned_dofs;
	primal.locally_relevant_dofs = slab->space.primal.fe_info->locally_relevant_dofs;

	dual.locally_owned_dofs      = slab->space.dual.fe_info->locally_owned_dofs;
	dual.locally_relevant_dofs   = slab->space.dual.fe_info->locally_relevant_dofs;

	// INFO: replace_linearization_points and replace_weights are bing hard coded in the error estimator
	//       They have a negligible impact on the effectivity index and hard coding them maskes the estimator more readable.

	primal.um_on_tn = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();

	pu.constraints = std::make_shared< dealii::AffineConstraints<double> > (); // slab->space.pu.fe_info->constraints;

	// init eta_s, eta_t
	{
		error_estimator.x_h = eta_s->x[0];
		error_estimator.x_k = eta_t->x[0];
	}


	// check low.solution_can_be_patchwise_interpolated
	{
		// check if all cells of the coarse level = 0 in tria have children
		auto low_cell = slab->space.low.fe_info->dof->begin(0);
		auto low_endc = slab->space.low.fe_info->dof->end(0);

		low.solution_can_be_patchwise_interpolated = true;

		for ( ; low_cell != low_endc; ++low_cell) {
			if (!(low_cell->has_children() || low_cell->is_artificial())) {
				low.solution_can_be_patchwise_interpolated = false;
				DTM::pout
					<< "WARNING: your tria on this slab: "
					<< std::endl
					<< "         ( " << slab->t_m << " , " << slab->t_n << " )"
					<< std::endl
					<< "         is not at least once globally refined."
					<< std::endl
					<< "         Patchwise higher order interpolation in space is not"
					<< std::endl
					<< "         possible, keeping the order instead."
					<< std::endl << std::endl;

				break;
			}
		}
	}

	////////////////////////////////////////////////////////////////////////
	// integrate or evaluate, respectively, on slab for
	// - eta_s : spatial error contributions
	// - eta_t : temporal error contributions
	//
	// Order: first in time (outer quadrature), then in space (inner quadrature)
	////////////////////////////////////////////////////////////////////////

	// prepare vector local dof ind. (in low time)
	std::vector< dealii::types::global_dof_index >
	low_local_dof_indices_time(
		slab->time.low.fe_info->fe->dofs_per_cell
	);

	Assert(slab->space.high.fe_info.use_count(), dealii::ExcNotInitialized());
	// quadratures for integration in space
	dealii::QGauss<dim> quad_cell(
		std::max(
			std::max(
				slab->space.high.fe_info->fe->base_element(0).base_element(0).tensor_degree(),
				slab->space.high.fe_info->fe->base_element(0).base_element(1).tensor_degree()
			),
			static_cast<unsigned int> (1)
		) + 3
	);

	/////////////////////////////////////////////////////////////////////////////
	// GETTING THE SPACE-TIME VECTORS: U_k, U_kh, Z, Z_k^{ρ_k}, Z_k^{ρ_h}, Z_kh
	//

	//////////////////////
	// GETTING Z
	//
	auto high_z_slab = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	if (!dual_order.compare("high")) // dual = [time = high, space = high]
		high_z_slab = z->x[0]; // for mixed order
	else if (!dual_order.compare("high-time")) // dual = [time = high, space = low]
	{
		// computation of z_kh^(2,2) from z_kh^(2,1)
		get_patchwise_high_order_interpolated_space_slab( // for semi-mixed order
			slab,
			z->x[0],
			high_z_slab
		);
	}
	else
		AssertThrow(false, dealii::ExcNotImplemented());

	//////////////////////
	// GETTING Z_k^{ρ_k}
	//
	auto high_z_k_rho_k_slab = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	if (!dual_order.compare("high") || !dual_order.compare("high-time")) // dual = { [time = high, space = high], [time = high, space = low] }
	{
		// computation of z_kh^(1,2) from z_kh^(2,2)
		get_back_interpolated_time_slab_w( // for mixed order
			slab,
			slab->space.high.fe_info->dof,
			high_z_slab,
			high_z_k_rho_k_slab
		);
	}
	else
		AssertThrow(false, dealii::ExcNotImplemented());

	//////////////////////
	// GETTING Z_k^{ρ_h}
	//   --> same as Z_k^{ρ_k}
	auto high_z_k_rho_h_slab = high_z_k_rho_k_slab;

	//////////////////////
	// GETTING Z_kh
	//
	auto high_z_kh_slab = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();

	if (!dual_order.compare("high")) // dual = [time = high, space = high]
	{
		// computation of z_kh^(2,1) from z_kh^(2,2)
		get_back_interpolated_space_slab( // for mixed order
			slab,
			high_z_k_rho_h_slab,
			high_z_kh_slab
		);
	}
	else if (!dual_order.compare("high-time")) // dual = [time = high, space = low]
	{
		get_back_interpolated_space_slab( // for semi-mixed order
			slab,
			high_z_k_rho_h_slab,
			high_z_kh_slab
		);
	}
	else
		AssertThrow(false, dealii::ExcNotImplemented());

	///////////////////////
	// GETTING U_k == U_kh
	//

	// interpolate u_kh^(1,1) in time
	auto low_interpolated_time_u_slab = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	get_interpolated_time_slab_w(
		slab,
		slab->space.low.fe_info->dof,
		u->x[0],
		low_interpolated_time_u_slab
	);

	// interpolate u_kh^(2,1) in space
	auto high_u_slab = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	get_interpolated_space_slab(
		slab,
		low_interpolated_time_u_slab,
		high_u_slab
	);

	////////////////////////////////////////////////////////////////////////
	// left jump (between slabs prepare)
	//

	auto tmp_owned = std::make_shared<dealii::TrilinosWrappers::MPI::Vector >();

	// interpolate primal solution u^-(t_m) to high solution space
	auto primal_um_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	primal_um_on_tm->reinit(
			*slab->space.primal.fe_info->locally_owned_dofs,
			*slab->space.primal.fe_info->locally_relevant_dofs,
			mpi_comm);

	// TODO: for-loop over pointers to simplify code
	high_u_kh_m_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	high_u_kh_m_on_tm->reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);


	high_u_k_m_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	high_u_k_m_on_tm->reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);



	high_u_k_p_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	high_u_k_p_on_tm->reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);


	high_u_kh_p_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_u_kh_p_on_tm -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);


	high_z_kh_p_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_kh_p_on_tm -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);

	high_z_k_rho_k_p_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_k_rho_k_p_on_tm  -> reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm);

	high_z_k_rho_h_p_on_tm = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_k_rho_h_p_on_tm  -> reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm);

	high_z_p_on_tm  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_p_on_tm  -> reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm);

	high_z_on_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_on_tq -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);

	high_z_kh_on_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_kh_on_tq  -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);

	high_z_k_rho_k_on_tq = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_k_rho_k_on_tq  -> reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm);

	high_z_k_rho_h_on_tq = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_z_k_rho_h_on_tq  -> reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm);
	high_u_kh_on_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_u_kh_on_tq  -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);

	high_dt_u_kh_on_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_dt_u_kh_on_tq  -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			*slab->space.high.fe_info->locally_relevant_dofs,
			mpi_comm);

	auto high_owned = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
	high_owned -> reinit(
			*slab->space.high.fe_info->locally_owned_dofs,
			mpi_comm);


	if (slab == grid->slabs.begin()) {
		// slab n == 1: interpolate initial value function u_0 to high solution space
		Assert(primal_um_on_tm.use_count(), dealii::ExcNotInitialized());
		Assert(primal_um_on_tm->size(), dealii::ExcNotInitialized());
		tmp_owned ->reinit(
			*slab->space.primal.fe_info->locally_owned_dofs,
			mpi_comm
		);
		*tmp_owned = 0.;
		// our case: u_0 = 0
//			function.u_0->set_time(slab->t_m);
//			dealii::VectorTools::interpolate(
//				*slab->space.primal.fe_info->mapping,
//				*slab->space.primal.fe_info->dof,
//				*function.u_0,
//				*tmp_owned
//			);


		// call hanging nodes to make the result continuous again
		// (Note: after the first dwr-loop the initial grid could have hanging nodes)
		slab->space.primal.fe_info->hanging_node_constraints->distribute(*tmp_owned);
		*primal_um_on_tm = *tmp_owned;
	}
	else {
		// slab n > 1
		Assert(primal.um_on_tn.use_count(), dealii::ExcNotInitialized());

		if (std::prev(slab)->space.primal.fe_info->dof->n_dofs() != primal.um_on_tn->size()){

			primal.um_on_tn->reinit(
					*std::prev(slab)->space.primal.fe_info->locally_owned_dofs,
					*std::prev(slab)->space.primal.fe_info->locally_relevant_dofs,
					mpi_comm);
		}
		// primal_um_on_tm has actually already been stored with a divergence free projection in um
		*primal_um_on_tm = *um->x[0];
	}

	// compute high_u_kh_m_on_tm and high_u_k_m_on_tm from primal_um_on_tm
	Assert(!primal_order.compare("low"), dealii::ExcNotImplemented());
	{
		// interpolate (space): low -> high
		dealii::FETools::interpolate(
			// primal/low solution
			*slab->space.primal.fe_info->dof,
			*primal_um_on_tm,
			// high solution
			*slab->space.high.fe_info->dof,
			*slab->space.high.fe_info->hanging_node_constraints,
			*high_owned
		);
		*high_u_kh_m_on_tm = *high_owned;
	}

	// replacing linearization point
	// ρ_k(u_k,.) ~ ρ_k(u_kh,.), i.e. use u_kh as u_k
	high_u_k_m_on_tm = high_u_kh_m_on_tm;

	////////////////////////////////////////////////////////////////////////
	// loop over all cells in time on this slab
	//
	auto cell_time = slab->time.high.fe_info->dof->begin_active();
	auto endc_time = slab->time.high.fe_info->dof->end();

	auto low_cell_time = slab->time.low.fe_info->dof->begin_active();

	for ( ; cell_time != endc_time; ++cell_time, ++low_cell_time) {
		cell_time_index = cell_time->index();

		dealii::DoFHandler<1>::active_cell_iterator primal_cell_time;
		primal_cell_time = low_cell_time;

		dealii::DoFHandler<1>::active_cell_iterator dual_cell_time;
		if (!dual_order.compare("low")) // dual == low
		{
			dual_cell_time = low_cell_time;
		}
		else // dual == high || dual == high-time
		{
			dual_cell_time = cell_time;
		}

		Assert(
			cell_time->center()[0] == low_cell_time->center()[0],
			dealii::ExcInternalError()
		);

		Assert(
			low_local_dof_indices_time.size() ==
			slab->time.low.fe_info->fe->dofs_per_cell,
			dealii::ExcNotInitialized()
		);

		low_cell_time->get_dof_indices(
			low_local_dof_indices_time
		);

		// face values: [t_m, t_n]
		dealii::FEValues<1> fe_face_values_time(
			*slab->time.high.fe_info->mapping,
			*slab->time.high.fe_info->fe,
			dealii::QGaussLobatto<1>(2),
			dealii::update_quadrature_points
		);

		fe_face_values_time.reinit(cell_time);

		////////////////////////////////////////////////////////////////////
		function.viscosity->set_time(fe_face_values_time.quadrature_point(0)[0]);

		////////////////////////////////////////////////////////////////////
		// jump in t_m
		//
		// inside slab: set primal_um_on_tm / high_u_kh_m_on_tm / high_u_k_m_on_tm
		if (cell_time->index() >= 1) {
//			Assert(primal.um_on_tn.use_count(), dealii::ExcNotInitialized());
//			Assert(
//				primal.um_on_tn->size() == slab->space.primal.fe_info->dof->n_dofs(),
//				dealii::ExcMessage(
//					"primal.um_on_tn->size() != slab->space.primal.fe_info->dof->n_dofs()"
//				)
//			);

			Assert(primal_um_on_tm.use_count(), dealii::ExcNotInitialized());
			Assert(primal_um_on_tm->size(), dealii::ExcNotInitialized());

//			primal_um_on_tm = *primal.um_on_tn;

			// compute high_u_kh_m_on_tm and high_u_k_m_on_tm from primal_um_on_tm
			// interpolate (space): low -> high
			dealii::FETools::interpolate(
				// primal/low solution
				*slab->space.primal.fe_info->dof,
				*primal_um_on_tm,
				// high solution
				*slab->space.high.fe_info->dof,
				*slab->space.high.fe_info->hanging_node_constraints,
				*high_owned
			);
			*high_u_kh_m_on_tm = *high_owned;

			// replacing linearization point:
			// ρ_k(u_k,.) ~ ρ_k(u_kh,.), i.e. use u_kh as u_k
			high_u_k_m_on_tm = high_u_kh_m_on_tm;
		} // end (inside slab)

		//////////////////////////////////////////
		// get the primal solution at t_m^+
		//
		get_w_t(
			slab->time.high.fe_info->fe,
			slab->time.high.fe_info->mapping,
			slab->space.high.fe_info->dof,
			cell_time,
			high_u_slab,
			fe_face_values_time.quadrature_point(0)[0],
			high_u_kh_p_on_tm
		);

		// replace linearization point:
		// ρ_k(u_k,.) ~ ρ_k(u_kh,.), i.e. use u_kh as u_k
		high_u_k_p_on_tm = high_u_kh_p_on_tm;

		//////////////////////////////////////////
		// get the dual solution at t_m^+
		//
		get_w_t(
			slab->time.high.fe_info->fe,
			slab->time.high.fe_info->mapping,
			slab->space.high.fe_info->dof,
			cell_time,
			high_z_slab,
			fe_face_values_time.quadrature_point(0)[0],
			high_z_p_on_tm
		);
		get_w_t(
			slab->time.high.fe_info->fe,
			slab->time.high.fe_info->mapping,
			slab->space.high.fe_info->dof,
			cell_time,
			high_z_k_rho_k_slab,
			fe_face_values_time.quadrature_point(0)[0],
			high_z_k_rho_k_p_on_tm
		);
		get_w_t(
			slab->time.high.fe_info->fe,
			slab->time.high.fe_info->mapping,
			slab->space.high.fe_info->dof,
			cell_time,
			high_z_k_rho_h_slab,
			fe_face_values_time.quadrature_point(0)[0],
			high_z_k_rho_h_p_on_tm
		);
		get_w_t(
			slab->time.high.fe_info->fe,
			slab->time.high.fe_info->mapping,
			slab->space.high.fe_info->dof,
			cell_time,
			high_z_kh_slab,
			fe_face_values_time.quadrature_point(0)[0],
			high_z_kh_p_on_tm
		);

		////////////////////////////////////////////////////////////////////
		// integrate in space on t_m:
		//
		// WorkStream
		// assemble cell_time on t_m problem
		//
		dealii::WorkStream::run(
			CellFilter(
				dealii::IteratorFilters::LocallyOwnedCell(), slab->space.high.fe_info->dof->begin_active()
			),
			CellFilter(
				dealii::IteratorFilters::LocallyOwnedCell(), slab->space.high.fe_info->dof->end()
			),
			std::bind (
				&ErrorEstimator<dim>::assemble_local_error_tm,
				this,
				std::placeholders::_1,
				std::placeholders::_2,
				std::placeholders::_3
			),
			std::bind (
				&ErrorEstimator<dim>::copy_local_error,
				this,
				std::placeholders::_1
			),
			Assembly::Scratch::ErrorEstimates<dim> (
				*slab->space.high.fe_info->dof,
				*slab->space.pu.fe_info->dof,
				*slab->space.high.fe_info->fe,
				*slab->space.pu.fe_info->fe,
				*slab->space.high.fe_info->mapping,
				quad_cell,
				//
				dealii::update_values |
				dealii::update_quadrature_points |
				dealii::update_JxW_values
			),
			Assembly::CopyData::ErrorEstimates<dim> (*slab->space.pu.fe_info->fe)
		);

		////////////////////////////////////////////////////////////////////
		// integrate time by quadrature
		// use high space to set up the time quadrature

		dealii::FEValues<1> fe_values_time(
			*slab->time.high.fe_info->mapping,
			*slab->time.high.fe_info->fe,
			dealii::QGauss<1> (
				slab->time.high.fe_info->fe->tensor_degree()+5
			),
			dealii::update_quadrature_points |
			dealii::update_JxW_values
		);

		fe_values_time.reinit(cell_time);

		for (unsigned int qt{0}; qt < fe_values_time.n_quadrature_points; ++qt) {
			cell_time_tau_n = cell_time->diameter();
			cell_time_JxW = fe_values_time.JxW(qt); // tau_n x w_q

			function.viscosity->set_time(fe_values_time.quadrature_point(qt)[0]);

			//////////////////////////////////////////
			// DELETE: get the dual solution at t_q
			//
			auto dual_z_on_tq = std::make_shared< dealii::TrilinosWrappers::MPI::Vector >();
			dual_z_on_tq->reinit(
				*slab->space.dual.fe_info->locally_owned_dofs,
				*slab->space.dual.fe_info->locally_relevant_dofs,
				mpi_comm
			);
			get_w_t(
				slab->time.dual.fe_info->fe,
				slab->time.dual.fe_info->mapping,
				slab->space.dual.fe_info->dof,
				dual_cell_time,
				z->x[0],
				fe_values_time.quadrature_point(qt)[0],
				dual_z_on_tq
			);

			////////////////////////////////////////////////////////
			// Load u_kh, z, z_k and z_kh from space-time vectors
			//
			get_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_u_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_u_kh_on_tq
			);

			get_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_z_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_z_on_tq
			);

			get_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_z_k_rho_k_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_z_k_rho_k_on_tq
			);

			get_dt_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_u_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_dt_u_kh_on_tq
			);

			get_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_z_k_rho_h_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_z_k_rho_h_on_tq
			);

			get_w_t(
				slab->time.high.fe_info->fe,
				slab->time.high.fe_info->mapping,
				slab->space.high.fe_info->dof,
				cell_time,
				high_z_kh_slab,
				fe_values_time.quadrature_point(qt)[0],
				high_z_kh_on_tq
			);

			////////////////////////////////////////////////////////////////
			// integrate in space on t_q:
			//
			// WorkStream
			// assemble cell_time on t_q problem
			//
			dealii::WorkStream::run(
				CellFilter(
					dealii::IteratorFilters::LocallyOwnedCell(), slab->space.high.fe_info->dof->begin_active()
				),
				CellFilter(
					dealii::IteratorFilters::LocallyOwnedCell(), slab->space.high.fe_info->dof->end()
				),
				std::bind (
					&ErrorEstimator<dim>::assemble_local_error,
					this,
					std::placeholders::_1,
					std::placeholders::_2,
					std::placeholders::_3
				),
				std::bind (
					&ErrorEstimator<dim>::copy_local_error,
					this,
					std::placeholders::_1
				),
				Assembly::Scratch::ErrorEstimates<dim> (
					*slab->space.high.fe_info->dof,
					*slab->space.pu.fe_info->dof,
					*slab->space.high.fe_info->fe,
					*slab->space.pu.fe_info->fe,
					*slab->space.high.fe_info->mapping,
					quad_cell,
					//
					dealii::update_values |
					dealii::update_gradients |
					dealii::update_quadrature_points |
					dealii::update_JxW_values
				),
				Assembly::CopyData::ErrorEstimates<dim> (*slab->space.pu.fe_info->fe)
			);
		} // t_q

		////////////////////////////////////////////////////////////////////
		// jump in t_n (either between next cell_time or (next slab or z_T))
		//
		// NOTE: only needed for the adjoint error estimator

		// TODO: Are the jump terms correct with more than one time interval per slab?
//		////////////////////////////////////////////////////////////////////
//		// evaluate solution u(t_n) on this cell_time
//		// for next cell_time or next slab
//		//
//
//		Assert(primal.um_on_tn.use_count(), dealii::ExcNotInitialized());
//		if (slab->space.primal.fe_info->dof->n_dofs() != primal.um_on_tn->size()) {
//			primal.um_on_tn->reinit(
//					*slab->space.primal.fe_info->locally_owned_dofs,
//					*slab->space.primal.fe_info->locally_relevant_dofs,
//					mpi_comm);
//		}
//
//		tmp_owned->reinit(
//				*slab->space.primal.fe_info->locally_owned_dofs,
//				mpi_comm);
//
//		std::cout << "assemble jump" << std::endl;
//		*primal.um_on_tn = 0;
//		{
//			dealii::FEValues<1> fe_face_values_time(
//				*slab->time.primal.fe_info->mapping,
//				*slab->time.primal.fe_info->fe,
//				dealii::QGaussLobatto<1>(2),
//				dealii::update_values
//			);
//
//			{
//				fe_face_values_time.reinit(primal_cell_time);
//
//				// evaluate solution for t_n of time cell
//				for (unsigned int jj{0};
//					jj < slab->time.primal.fe_info->fe->dofs_per_cell; ++jj){
//
//					dealii::IndexSet::ElementIterator lri =
//							slab->space.primal.fe_info->locally_owned_dofs->begin();
//
//					dealii::IndexSet::ElementIterator lre =
//							slab->space.primal.fe_info->locally_owned_dofs->end();
//
//				for ( ; lri != lre ; lri++ ){
//						(*tmp_owned)[*lri] += (*u->x[0])[
//							*lri
//							// time offset
//							+ slab->space.primal.fe_info->dof->n_dofs() *
//								(primal_cell_time->index() * slab->time.primal.fe_info->fe->dofs_per_cell)
//							// local in time dof
//							+ slab->space.primal.fe_info->dof->n_dofs() * jj
//						] * fe_face_values_time.shape_value(jj,1);
//					}
//				}
//				*primal.um_on_tn = *tmp_owned;
//			}
//		}
	} // for cell_time

	error_estimator.x_h->compress(dealii::VectorOperation::add);
	error_estimator.x_k->compress(dealii::VectorOperation::add);
	//
	////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////
	// prepare next slab
	//
	primal_um_on_tm = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
//
//

template<int dim>
void
ErrorEstimator<dim>::
get_w_t(
	std::shared_ptr< dealii::FiniteElement<1> > time_fe,
	std::shared_ptr< dealii::Mapping<1> > time_mapping,
	std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
	const typename dealii::DoFHandler<1>::active_cell_iterator &cell_time,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > w,
	const double &t,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > &w_t
) {
	Assert(w.use_count(), dealii::ExcNotInitialized());

	dealii::FEValues<1> fe_face_values_time(
		*time_mapping,
		*time_fe,
		dealii::QGaussLobatto<1>(2),
		dealii::update_quadrature_points
	);

	fe_face_values_time.reinit(cell_time);

#ifdef DEBUG
	{
		// check if t >= t_m of dual_cell_time
		Assert(
			(t >= fe_face_values_time.quadrature_point(0)[0] ),
			dealii::ExcInvalidState()
		);

		// check if <= t_n of dual_cell_time
		Assert(
			(t <= fe_face_values_time.quadrature_point(1)[0] ),
			dealii::ExcInvalidState()
		);
	}
#endif

	const dealii::IndexSet loe = w_t->locally_owned_elements();
	auto tmpvec = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
    tmpvec->reinit(
		loe,
		mpi_comm);
	// create special quadrature for fe eval on t
	// NOTE: cell_time->diameter() = tau_n of cell_time in 1d

	std::vector< dealii::Point<1> > output_time_points(1);
	output_time_points[0][0] =
		(t - fe_face_values_time.quadrature_point(0)[0])
		/ cell_time->diameter();

	// eval
	dealii::FEValues<1> fe_values_time(
		*time_mapping,
		*time_fe,
		dealii::Quadrature<1> (output_time_points),
		dealii::update_values
	);

	fe_values_time.reinit(cell_time);

	std::vector< dealii::types::global_dof_index > local_dof_indices(
		time_fe->dofs_per_cell
	);

	cell_time->get_dof_indices(local_dof_indices);

    *tmpvec = 0.;
    Assert(fe_values_time.n_quadrature_points==1,
			dealii::ExcInternalError());

	//only a single QP, so no loop needed
	unsigned int qt{0};

	// evaluate solution for t_q
	for (
		unsigned int jj{0};
		jj < time_fe->dofs_per_cell; ++jj) {

		dealii::IndexSet::ElementIterator lri = loe.begin();
		dealii::IndexSet::ElementIterator lre = loe.end();
		for ( ; lri != lre ; lri ++) {
			double tmp  = (*w)[
				*lri
				// time offset
				+ space_dof->n_dofs() *
					local_dof_indices[jj]
			] * fe_values_time.shape_value(jj,qt);

			(*tmpvec)[*lri] += tmp;
		}
	}
	*w_t = *tmpvec;
}


template<int dim>
void
ErrorEstimator<dim>::
get_dt_w_t(
	std::shared_ptr< dealii::FiniteElement<1> > time_fe,
	std::shared_ptr< dealii::Mapping<1> > time_mapping,
	std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
	const typename dealii::DoFHandler<1>::active_cell_iterator &cell_time,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > w,
	const double &t,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > &dt_w_t
) {
	Assert(w.use_count(), dealii::ExcNotInitialized());

	dealii::FEValues<1> fe_face_values_time(
		*time_mapping,
		*time_fe,
		dealii::QGaussLobatto<1>(2),
		dealii::update_quadrature_points
	);

	fe_face_values_time.reinit(cell_time);

#ifdef DEBUG
	{
		// check if t >= t_m of dual_cell_time
		Assert(
			(t >= fe_face_values_time.quadrature_point(0)[0] ),
			dealii::ExcInvalidState()
		);

		// check if <= t_n of dual_cell_time
		Assert(
			(t <= fe_face_values_time.quadrature_point(1)[0] ),
			dealii::ExcInvalidState()
		);
	}
#endif

	const dealii::IndexSet loe = dt_w_t->locally_owned_elements();
	auto tmpvec = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	tmpvec->reinit(
		loe,
		mpi_comm);
	// create special quadrature for fe eval on t
	// NOTE: cell_time->diameter() = tau_n of cell_time in 1d

	std::vector< dealii::Point<1> > output_time_points(1);
	output_time_points[0][0] =
		(t - fe_face_values_time.quadrature_point(0)[0])
		/ cell_time->diameter();

	// eval
	dealii::FEValues<1> fe_values_time(
		*time_mapping,
		*time_fe,
		dealii::Quadrature<1> (output_time_points),
		dealii::update_gradients
	);

	fe_values_time.reinit(cell_time);

	std::vector< dealii::types::global_dof_index > local_dof_indices(
		time_fe->dofs_per_cell
	);

	cell_time->get_dof_indices(local_dof_indices);

	Assert(fe_values_time.n_quadrature_points==1,
			dealii::ExcInternalError());

	*tmpvec = 0.;
	//only a single QP, so no loop needed
	unsigned int qt{0};

	// evaluate solution for t_q
	for (
		unsigned int jj{0};
		jj < time_fe->dofs_per_cell; ++jj) {

		dealii::IndexSet::ElementIterator lri = loe.begin();
		dealii::IndexSet::ElementIterator lre = loe.end();

		for ( ; lri != lre ; lri++ ){
			(*tmpvec)[*lri] += (*w)[
				*lri
				// time offset
				+ space_dof->n_dofs() *
					local_dof_indices[jj]
			  ] * fe_values_time.shape_grad(jj,qt)[0];
		}
	}

	*dt_w_t = *tmpvec;
}


template<int dim>
void
ErrorEstimator<dim>::
interpolate_space(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > w,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > &interpolated_space_w
) {
	Assert(w.use_count(), dealii::ExcNotInitialized());
	Assert(
		(w->size() == slab->space.low.fe_info->dof->n_dofs()),
		dealii::ExcMessage(
			"Internal dimensions error: "
			"w->size() =/= slab->space.low.fe_info->dof->n_dofs()"
		)
	);


	auto tmp = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
	tmp->reinit(interpolated_space_w->locally_owned_elements(),
			   mpi_comm);

	Assert(interpolated_space_w.use_count(), dealii::ExcNotInitialized());

	// interpolate low dof vector to high dof vector
	dealii::FETools::interpolate(
		// low solution
		*slab->space.low.fe_info->dof,
		*w,
		// high solution
		*slab->space.high.fe_info->dof,
		*slab->space.high.fe_info->hanging_node_constraints,
		*tmp
	);
	*interpolated_space_w = *tmp;
}


template<int dim>
void
ErrorEstimator<dim>::
patchwise_high_order_interpolate_space(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > w,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > higher_order_space_w
) {
	// I_h^[high]{ I_h^[2*p on 2*h patch]{ w(t) } } or
	// I_h^[high]{ w(t) }:
	//
	// Interpolate to high space
	// the higher order interpolation (2*p) of a vector w(t) from the low space.
	//
	// The interpolation is done on 2*h patches (one mesh level up).
	// Therefore, the spatial triangulation must be at least once globally refined.
	//
	// If the interpolation is not possible, then I_h^[high]{ w(t) }
	// is the result.

	Assert(w.use_count(), dealii::ExcNotInitialized());
	Assert(
		(w->size() == slab->space.low.fe_info->dof->n_dofs()),
		dealii::ExcMessage(
			"Internal dimensions error: "
			"w->size() =/= slab->space.low.fe_info->dof->n_dofs()"
		)
	);

	Assert(higher_order_space_w.use_count(), dealii::ExcNotInitialized());

	auto tmpvec = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
	tmpvec -> reinit(
			higher_order_space_w->locally_owned_elements(),
			mpi_comm
	);

	if (low.solution_can_be_patchwise_interpolated) {
		// patchwise higher order interpolation in space
		// this function is called 'extrapolate' in deal.II
		// but its name is 'probably not particularly well chosen'
		dealii::FETools::extrapolate(
			*slab->space.low.fe_info->dof,
			*w,
			// high solution
			*slab->space.high.fe_info->dof,
			*slab->space.high.fe_info->hanging_node_constraints,
			*tmpvec
		);
	}
	else {
		// interpolate low dof vector to high dof vector
		dealii::FETools::interpolate(
			// low solution
			*slab->space.low.fe_info->dof,
			*w,
			// high solution
			*slab->space.high.fe_info->dof,
			*slab->space.high.fe_info->hanging_node_constraints,
			*tmpvec
		);
	}
	*higher_order_space_w = *tmpvec;
}

template<int dim>
void
ErrorEstimator<dim>::
back_interpolate_space(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > w,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > back_interpolated_space_w) {
	// back interpolation:
	// IR_h_w^high(t) = I_h^[high]{ R_h^[low]{w^high(t)} }

	Assert(w.use_count(), dealii::ExcNotInitialized());
	Assert(
		(w->size() == slab->space.high.fe_info->dof->n_dofs()),
		dealii::ExcMessage(
			"Internal dimensions error: "
			"w->size() =/= slab->space.high.fe_info->dof->n_dofs()"
		)
	);

	Assert(back_interpolated_space_w.use_count(), dealii::ExcNotInitialized());

	auto tmp = std::make_shared<dealii::TrilinosWrappers::MPI::Vector>();
	tmp->reinit(back_interpolated_space_w->locally_owned_elements(),
			   mpi_comm);

	dealii::FETools::back_interpolate(
		// high space (input and output space)
		*slab->space.high.fe_info->dof,
		*slab->space.high.fe_info->hanging_node_constraints,
		// input vector
		*w,
		// low space (restriction space)
		*slab->space.low.fe_info->dof,
		*slab->space.low.fe_info->hanging_node_constraints,
		// output vector (high solution after back interpolation)
		*tmp
	);
	*back_interpolated_space_w = *tmp;
}

template<int dim>
void
ErrorEstimator<dim>::
get_patchwise_high_order_interpolated_space_slab(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > slab_w,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > &high_slab_w
) {
	// TODO: adapt this for multiple cells per slab, as in get_interpolated_time_slab_w

	// slab_w evaluated at temporal quadrature point
	auto slab_w_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	slab_w_tq->reinit(
		*slab->space.low.fe_info->locally_owned_dofs,
		*slab->space.low.fe_info->locally_relevant_dofs,
		mpi_comm
	);
	*slab_w_tq = 0.;

	auto slab_w_tq_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	slab_w_tq_tmp->reinit(
		*slab->space.low.fe_info->locally_owned_dofs,
		mpi_comm
	);

	// higher order reconstruction in space of slab_w evaluated at temporal quadrature point
	auto high_slab_w_tq  = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	high_slab_w_tq->reinit(
		*slab->space.high.fe_info->locally_owned_dofs,
		*slab->space.high.fe_info->locally_relevant_dofs,
		mpi_comm
	);
	*high_slab_w_tq = 0.;

	for (unsigned int ii{0}; ii < slab->time.high.fe_info->dof->n_dofs(); ++ii)
	{
		// get slab_w_tq
		//for (dealii::types::global_dof_index i{0}; i < slab->space.low.fe_info->dof->n_dofs(); ++i)
		for (auto i : *slab->space.low.fe_info->locally_owned_dofs)
			(*slab_w_tq_tmp)[i] = (*slab_w)[i + slab->space.low.fe_info->dof->n_dofs() * ii];
		*slab_w_tq = *slab_w_tq_tmp;

		// use higher order interpolation in space to go from slab_w_tq to high_slab_w_tq
		patchwise_high_order_interpolate_space(
			slab,
			slab_w_tq,
			high_slab_w_tq
		);

		// write high_slab_w_tq into high_slab_w
		//for (dealii::types::global_dof_index i{0}; i < slab->space.high.fe_info->dof->n_dofs(); ++i)
		for (auto i : *slab->space.high.fe_info->locally_owned_dofs)
			(*high_slab_w)[i + slab->space.high.fe_info->dof->n_dofs() * ii] = (*high_slab_w_tq)[i];
	}
}


template<int dim>
void
ErrorEstimator<dim>::
get_back_interpolated_time_slab_w(
	const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
	std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > slab_w,
	std::shared_ptr< dealii::TrilinosWrappers::MPI::Vector > back_interpolated_time_slab_w
) {
	// TODO: does this also work for multiple cells per slab?!
	// evaluate back interpolation in time for the whole slab solution z:
	// IR_tau_w^high = I_tau^[high]{ R_tau^[low]{z^high} }

	Assert(
		space_dof.use_count(),
		dealii::ExcNotInitialized()
	);
	Assert(
		slab->time.low.fe_info->dof.use_count(),
		dealii::ExcNotInitialized()
	);

	Assert(
		slab->time.high.fe_info->fe.use_count(),
		dealii::ExcNotInitialized()
	);

	Assert(
		slab_w.use_count(),
		dealii::ExcNotInitialized()
	);

	Assert(
		slab_w->size(),
		dealii::ExcNotInitialized()
	);


	Assert(
		back_interpolated_time_slab_w.use_count(),
		dealii::ExcNotInitialized()
	);

	dealii::IndexSet low_restricted_owned_dofs =
		idealii::SlabDoFTools::extract_locally_owned_dofs(
			space_dof,
			slab->time.low.fe_info->dof
		);

	dealii::IndexSet low_restricted_relevant_dofs =
		idealii::SlabDoFTools::extract_locally_owned_dofs(
			space_dof,
			slab->time.low.fe_info->dof
		);

	dealii::IndexSet high_restricted_owned_dofs =
		idealii::SlabDoFTools::extract_locally_owned_dofs(
			space_dof,
			slab->time.high.fe_info->dof
		);

	dealii::IndexSet high_restricted_relevant_dofs =
		idealii::SlabDoFTools::extract_locally_owned_dofs(
			space_dof,
			slab->time.high.fe_info->dof
		);

	// init intermediate vector
	auto low_restricted_time_slab_w = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	low_restricted_time_slab_w->reinit(
		low_restricted_owned_dofs,
		low_restricted_relevant_dofs,
		mpi_comm
	);
	*low_restricted_time_slab_w = 0.;

	auto low_restricted_tmp = std::make_shared< dealii::TrilinosWrappers::MPI::Vector> ();
	low_restricted_tmp->reinit(
		low_restricted_owned_dofs,
		mpi_comm
	);

	// init result vector
//	back_interpolated_time_slab_w = std::make_shared< dealii::TrilinosWrappers::MPI::Vector > ();
	back_interpolated_time_slab_w->reinit(
		high_restricted_owned_dofs,
		high_restricted_relevant_dofs,
		mpi_comm
	);
	*back_interpolated_time_slab_w = 0.;

	auto back_tmp = std::make_shared<dealii::TrilinosWrappers::MPI::Vector >();
	back_tmp->reinit(
		high_restricted_owned_dofs,
		mpi_comm
	);

	// prepare local dof mappings
	std::vector< dealii::types::global_dof_index >
	low_local_dof_indices_time(
		slab->time.low.fe_info->fe->dofs_per_cell
	);

	std::vector< dealii::types::global_dof_index >
	high_local_dof_indices_time(
		slab->time.high.fe_info->fe->dofs_per_cell
	);

	// prepare fe values for restriction and back interpolation
	dealii::FEValues<1> high_fe_values_time(
		*slab->time.high.fe_info->mapping,
		*slab->time.high.fe_info->fe,
		dealii::Quadrature<1>(slab->time.low.fe_info->fe->get_unit_support_points()),
		dealii::update_values
	);

	dealii::FEValues<1> low_fe_values_time(
		*slab->time.low.fe_info->mapping,
		*slab->time.low.fe_info->fe,
		dealii::Quadrature<1>(slab->time.high.fe_info->fe->get_unit_support_points()),
		dealii::update_values
	);

	// loop over all cell time on slab
	auto cell_time = slab->time.high.fe_info->dof->begin_active();
	auto endc_time = slab->time.high.fe_info->dof->end();

	auto low_cell_time = slab->time.low.fe_info->dof->begin_active();

	for ( ; cell_time != endc_time; ++cell_time, ++low_cell_time) {
		////////////////////////////////////////////////////////////////////////
		// check (low) cell time iterators and matching
		Assert(
			low_cell_time != slab->time.low.fe_info->dof->end(),
			dealii::ExcInternalError()
		);

		Assert(
			cell_time->center()[0] == low_cell_time->center()[0],
			dealii::ExcInternalError()
		);

		////////////////////////////////////////////////////////////////////////
		// high_local_dof_indices_time
		Assert(
			high_local_dof_indices_time.size() ==
			slab->time.high.fe_info->fe->dofs_per_cell,
			dealii::ExcNotInitialized()
		);

		cell_time->get_dof_indices(
			high_local_dof_indices_time
		);

		// low_local_dof_indices_time
		Assert(
			low_local_dof_indices_time.size() ==
			slab->time.low.fe_info->fe->dofs_per_cell,
			dealii::ExcNotInitialized()
		);

		low_cell_time->get_dof_indices(
			low_local_dof_indices_time
		);

		////////////////////////////////////////////////////////////////////////
		// reinit and check: high_fe_values_time
		high_fe_values_time.reinit(cell_time);

		Assert(
			high_local_dof_indices_time.size() ==
			high_fe_values_time.get_fe().dofs_per_cell,
			dealii::ExcInternalError()
		);

		Assert(
			high_fe_values_time.n_quadrature_points ==
			low_local_dof_indices_time.size(),
			dealii::ExcInternalError()
		);

		// reinit and check: low_fe_values_time
		low_fe_values_time.reinit(low_cell_time);

		Assert(
			low_local_dof_indices_time.size() ==
			low_fe_values_time.get_fe().dofs_per_cell,
			dealii::ExcInternalError()
		);

		Assert(
			low_fe_values_time.n_quadrature_points ==
			high_local_dof_indices_time.size(),
			dealii::ExcInternalError()
		);

		////////////////////////////////////////////////////////////////////////
		// z -> R_tau z
		//

		// NOTE: qt must correspond to local dof jj of low restriction
		Assert(
			high_fe_values_time.n_quadrature_points ==
			slab->time.low.fe_info->fe->dofs_per_cell,
			dealii::ExcInternalError()
		);

		for (unsigned int qt{0};
			qt < high_fe_values_time.n_quadrature_points; ++qt) {

			for (
				unsigned int jj{0};
				jj < high_fe_values_time.get_fe().dofs_per_cell; ++jj) {
					dealii::IndexSet::ElementIterator lri = space_dof->locally_owned_dofs().begin();
					dealii::IndexSet::ElementIterator lre = space_dof->locally_owned_dofs().end();
					for ( ; lri!=lre ; lri++ ) {
						(*low_restricted_tmp)[
							*lri
							// time offset
							+ space_dof->n_dofs() *
								low_local_dof_indices_time[qt]
						] += (*slab_w)[
							*lri
							// time offset
							+ space_dof->n_dofs() *
								high_local_dof_indices_time[jj]
						] * high_fe_values_time.shape_value(jj,qt);
					}
				}
			}

		*low_restricted_time_slab_w = *low_restricted_tmp;
		////////////////////////////////////////////////////////////////////////
		// R_tau z -> IR_tau_z
		//

		// NOTE: qt must correspond to local dof jj of high back interpolation
		Assert(
			low_fe_values_time.n_quadrature_points ==
			slab->time.high.fe_info->fe->dofs_per_cell,
			dealii::ExcInternalError()
		);

		for (unsigned int qt{0};
			qt < low_fe_values_time.n_quadrature_points; ++qt) {

			for (
				unsigned int jj{0};
				jj < low_fe_values_time.get_fe().dofs_per_cell; ++jj) {

				dealii::IndexSet::ElementIterator lri = space_dof->locally_owned_dofs().begin();
				dealii::IndexSet::ElementIterator lre = space_dof->locally_owned_dofs().end();
				for ( ; lri!=lre ; lri++ ) {
					(*back_tmp)[
					 *lri
					// time offset
					+ space_dof->n_dofs() *
						high_local_dof_indices_time[qt]
				] += (*low_restricted_time_slab_w)[
					*lri
					// time offset
					+ space_dof->n_dofs() *
						low_local_dof_indices_time[jj]
				] * low_fe_values_time.shape_value(jj,qt);
			}}
		}
	}
	*back_interpolated_time_slab_w = *back_tmp;
}

//
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// assemble functions:
//

template<int dim>
void
ErrorEstimator<dim>::
assemble_local_error_tm(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimates<dim> &scratch,
	Assembly::CopyData::ErrorEstimates<dim> &copydata) {

	////////////////////////////////////////////////////////////////////////
	// cell integrals:
	//
	assemble_error_on_cell_tm(cell, scratch.cell, copydata.cell);
}

template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_cell_tm(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
	Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata) {

	typename dealii::DoFHandler<dim>::active_cell_iterator cell_high(&cell->get_triangulation(),
															   cell->level(),
															   cell->index(),
															   &scratch.dof_high);

	typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&cell->get_triangulation(),
																	cell->level(),
																	cell->index(),
																   &scratch.dof_pu);


	// reinit scratch and data to current cell
	scratch.fe_values_high.reinit(cell_high);
	scratch.fe_values_pu.reinit(cell_pu);

	// fetch local dof data
	cell_high->get_dof_indices(scratch.local_dof_indices_high);

	for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
		++scratch.j) {

		//////////////
		// PART 1: u

		// 1)  u_kh := u_kh^(1,1) = u_kh^low = I_k(I_h(u_kh^high))
		// 1a) get u_kh(t_m^-)
		scratch.local_u_kh_m[scratch.j] =
			(*high_u_kh_m_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];
		// 1b) get u_kh(t_m^+)
		scratch.local_u_kh_p[scratch.j] =
			(*high_u_kh_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];

		// 2)  u_k := u_kh^(1,2) = I_2h(u_kh^low) = I_k(u_kh^high)
		// BUT: if we replace the linearization point, then here u_k is being replaced by u_kh --> this has already been done in the code before!
		// 2a) get u_k(t_m^-)
		scratch.local_u_k_m[scratch.j] =
			(*high_u_k_m_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];
		// 2b) get u_k(t_m^+)
		scratch.local_u_k_p[scratch.j] =
			(*high_u_k_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];

		//////////////
		// PART 2: z

		// 3)  z - z_k to evaluate temporal error ρ_k(.,z-z_k)

		// 3a) z
		scratch.local_z_p[scratch.j] =
			(*high_z_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];

		// 3b) z_k [for ρ_k]
		scratch.local_z_k_rho_k_p[scratch.j] =
			(*high_z_k_rho_k_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];

		// 4)  z_k - z_kh to evaluate spatial error ρ_h(.,z_k-z_kh)

		// 4a) z_k [for ρ_h]
		scratch.local_z_k_rho_h_p[scratch.j] =
			(*high_z_k_rho_h_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];

		// 4b) z_kh
		scratch.local_z_kh_p[scratch.j] =
			(*high_z_kh_p_on_tm)[ scratch.local_dof_indices_high[scratch.j] ];
	}
	
	// For the jump we always only need the convection components of all the vectors

	// initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    // dof mapping pu: local to global
    std::vector< dealii::types::global_dof_index > tmp_local_dof_indices_pu(scratch.fe_values_pu.get_fe().dofs_per_cell);
    cell_pu->get_dof_indices(tmp_local_dof_indices_pu);
    for (unsigned int i{0}; i < scratch.fe_values_pu.get_fe().dofs_per_cell; ++i)
		copydata.local_dof_indices_pu[i] =
			tmp_local_dof_indices_pu[i]
			+ cell_time_index * scratch.dof_pu.n_dofs();

	// assemble PU
	for (scratch.q=0; scratch.q < scratch.fe_values_pu.n_quadrature_points; ++scratch.q) {
		scratch.JxW = scratch.fe_values_pu.JxW(scratch.q);

		// loop over all basis functions to get the shape values
		for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.phi_convection[scratch.j] =
				scratch.fe_values_high[convection].value(scratch.j,scratch.q);
		}

		scratch.value_jump_u_kh_convection = 0;
		scratch.value_jump_u_k_convection = 0;
		scratch.value_z_z_k_convection = 0;
		scratch.value_z_k_z_kh_convection = 0;

		for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.value_jump_u_kh_convection +=
				(scratch.local_u_kh_p[scratch.j] - scratch.local_u_kh_m[scratch.j])
				* scratch.phi_convection[scratch.j];

			scratch.value_jump_u_k_convection +=
				(scratch.local_u_k_p[scratch.j] - scratch.local_u_k_m[scratch.j])
				* scratch.phi_convection[scratch.j];

			scratch.value_z_z_k_convection +=
				(scratch.local_z_p[scratch.j] - scratch.local_z_k_rho_k_p[scratch.j])
				* scratch.phi_convection[scratch.j];

			scratch.value_z_k_z_kh_convection +=
				(scratch.local_z_k_rho_h_p[scratch.j] - scratch.local_z_kh_p[scratch.j])
				* scratch.phi_convection[scratch.j];
		} // for j

		// shape values for partition of Unity
		for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
			 ++ scratch.j )
		{
			scratch.chi[scratch.j] =
				scratch.fe_values_pu.shape_value(scratch.j, scratch.q);
		}

		for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
			++ scratch.j )
		{
			copydata.local_eta_k_vector[scratch.j] -= (
				// [ v_k(t_m) ] * (z^v(t_m^+) - z^v_k(t_m^+))χ_i(t_m^+)
				scratch.value_jump_u_k_convection
				* scratch.value_z_z_k_convection
				* scratch.chi[scratch.j]
			) * scratch.JxW;

			copydata.local_eta_h_vector[scratch.j] -= (
				// [ v_kh(t_m) ] * (z^v_k(t_m^+) - z^v_kh(t_m^+))χ_i(t_m^+)
				scratch.value_jump_u_kh_convection
				* scratch.value_z_k_z_kh_convection
				* scratch.chi[scratch.j]
			) * scratch.JxW;
		}
	} // for q
}

//// NOTE: only need this for adjoint error estimator
//template<int dim>
//void
//ErrorEstimator<dim>::
//assemble_local_error_tn(
//	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
//	Assembly::Scratch::ErrorEstimates<dim> &scratch,
//	Assembly::CopyData::ErrorEstimates<dim> &copydata) {
//
//	////////////////////////////////////////////////////////////////////////
//	// cell integrals:
//	//
//	assemble_error_on_cell_tn(cell, scratch.cell, copydata.cell);
//}

//// NOTE: only need this for adjoint error estimator
//template<int dim>
//void
//ErrorEstimator<dim>::
//assemble_error_on_cell_tn(
//	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
//	Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
//	Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata) {
//
//}


template<int dim>
void
ErrorEstimator<dim>::
assemble_local_error(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimates<dim> &scratch,
	Assembly::CopyData::ErrorEstimates<dim> &copydata) {

	////////////////////////////////////////////////////////////////////////
	// cell integrals:
	//
	assemble_error_on_cell(cell, scratch.cell, copydata.cell);
}


template<int dim>
void
ErrorEstimator<dim>::
assemble_error_on_cell(
	const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
	Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
	Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata) {

	typename dealii::DoFHandler<dim>::active_cell_iterator cell_high(&cell->get_triangulation(),
															   cell->level(),
															   cell->index(),
															   &scratch.dof_high);

	typename dealii::DoFHandler<dim>::active_cell_iterator cell_pu(&cell->get_triangulation(),
																	cell->level(),
																	cell->index(),
																   &scratch.dof_pu);

	// reinit scratch and data to current cell
	scratch.fe_values_high.reinit(cell_high);
	scratch.fe_values_pu.reinit(cell_pu);

	// fetch local dof data
	cell_high->get_dof_indices(scratch.local_dof_indices_high);

	for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
		++scratch.j) {

		//////////////
		// PART 1: u

		// 1)  u_kh := u_kh^(1,1) = u_kh^low = I_k(I_h(u_kh^high))
		// 1a) get u_kh(t_q)
		scratch.local_u_kh_tq[scratch.j] =
			(*high_u_kh_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];
		// 1b) get ∂_t u_kh(t_q)
		scratch.local_dtu_kh_tq[scratch.j] =
			(*high_dt_u_kh_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

		// 2)  u_k := u_kh^(1,2) = I_2h(u_kh^low) = I_k(u_kh^high)
		// 2a) get u_k(t_q)
		scratch.local_u_k_tq[scratch.j] =
			(*high_u_k_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];
		// 2b) get ∂_t u_k(t_q)
		scratch.local_dtu_k_tq[scratch.j] =
			(*high_dt_u_k_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

		//////////////
		// PART 2: z

		// 3)  z - z_k to evaluate temporal error ρ_k(.,z-z_k)

		// 3a) z
		scratch.local_z_tq[scratch.j] =
			(*high_z_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

		// 3b) z_k [for ρ_k]
		scratch.local_z_k_rho_k_tq[scratch.j] =
			(*high_z_k_rho_k_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

		// 4)  z_k - z_kh to evaluate spatial error ρ_h(.,z_k-z_kh)

		// 4a) z_k [for ρ_h]
		scratch.local_z_k_rho_h_tq[scratch.j] =
			(*high_z_k_rho_h_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

		// 4b) z_kh
		scratch.local_z_kh_tq[scratch.j] =
			(*high_z_kh_on_tq)[ scratch.local_dof_indices_high[scratch.j] ];

	}
	
	// initialize copydata
    copydata.local_eta_h_vector = 0.;
    copydata.local_eta_k_vector = 0.;
    // dof mapping pu: local to global
    std::vector< dealii::types::global_dof_index > tmp_local_dof_indices_pu(scratch.fe_values_pu.get_fe().dofs_per_cell);
    cell_pu->get_dof_indices(tmp_local_dof_indices_pu);
    for (unsigned int i{0}; i < scratch.fe_values_pu.get_fe().dofs_per_cell; ++i)
		copydata.local_dof_indices_pu[i] =
			tmp_local_dof_indices_pu[i]
			+ cell_time_index * scratch.dof_pu.n_dofs();

	// assemble PU
	for (scratch.q=0; scratch.q < scratch.fe_values_pu.n_quadrature_points; ++scratch.q) {
		scratch.JxW = scratch.fe_values_pu.JxW(scratch.q);

		// loop over all basis functions to get the shape values
		for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
			++scratch.j) {
			scratch.phi_convection[scratch.j] =
				scratch.fe_values_high[convection].value(scratch.j, scratch.q);

			scratch.grad_phi_convection[scratch.j] =
				scratch.fe_values_high[convection].gradient(scratch.j, scratch.q);

			scratch.symgrad_phi_convection[scratch.j] =
				scratch.fe_values_high[convection].symmetric_gradient(scratch.j, scratch.q);

			scratch.div_phi_convection[scratch.j] =
				scratch.fe_values_high[convection].divergence(scratch.j, scratch.q);

			scratch.phi_pressure[scratch.j] =
				scratch.fe_values_high[pressure].value(scratch.j, scratch.q);
		}

		scratch.value_viscosity = function.viscosity->value(
			scratch.fe_values_pu.quadrature_point(scratch.q), 0
		);

		////////////////////////////////////////////////////////////////////////
		// primal residuals

		///////////////////
		// 1. term: ∂_t v
		//

		// (∂_t v_k, [z^v-z^v_k]χ_i)
		scratch.value_dt_u_k_convection = 0;
		scratch.value_z_z_k_convection = 0;

		// (∂_t v_kh, [z^v_k-z^v_kh]χ_i)
		scratch.value_dt_u_kh_convection = 0;
		scratch.value_z_k_z_kh_convection = 0;

		////////////////////////////////////////
		// 2. term: symmetric/unsymmetric stress
		//
		if (symmetric_stress)
		{
			// symmetric stress tensor

			// 2ν (ϵ(v_k), ϵ([z^v-z^v_k]χ_i))       where ϵ(v) = 1/2(∇v + (∇v)^T)
			scratch.value_symgrad_u_k_convection = 0;
			scratch.value_symgrad_z_z_k_convection = 0;

			// 2ν (ϵ(v_kh), ϵ([z^v_k-z^v_kh]χ_i))   where ϵ(v) = 1/2(∇v + (∇v)^T)
			scratch.value_symgrad_u_kh_convection = 0;
			scratch.value_symgrad_z_k_z_kh_convection = 0;
		}
		else
		{
			// unsymmetric stress tensor

			// ν (∇ v_k, ∇{[z^v-z^v_k]χ_i})
			scratch.value_grad_u_k_convection = 0;
			scratch.value_grad_z_z_k_convection = 0;

			// ν (∇ v_kh, ∇{[z^v_k-z^v_kh]χ_i})
			scratch.value_grad_u_kh_convection = 0;
			scratch.value_grad_z_k_z_kh_convection = 0;
		}

		// -(p_k, ∇ · {[z^v-z^v_k]χ_i})
		scratch.value_u_k_pressure = 0;
		scratch.value_div_z_z_k_convection = 0;

		// -(p_kh, ∇ · {[z^v_k-z^v_kh]χ_i})
		scratch.value_u_kh_pressure = 0;
		scratch.value_div_z_k_z_kh_convection = 0;

		////////////////////////////////////////
		// 3. term: NSE nonlinearity
		//
		if (nonlinear)
		{
			// get ∇ v_k and ∇ v_kh
			if (symmetric_stress)
			{
				scratch.value_grad_u_k_convection = 0;
				scratch.value_grad_u_kh_convection = 0;
			}

			// get v_k and v_kh
			scratch.value_u_k_convection = 0;
			scratch.value_u_kh_convection = 0;
		}

		////////////////////////////////////////
		// 4. term: incompressibility condition
		//

		// (∇ ·v_k, [z^p-z^p_k]χ_i)
		scratch.value_div_u_k_convection = 0;
		scratch.value_z_z_k_pressure = 0;

		// (∇ ·v_kh, [z^p_k-z^p_kh]χ_i)
		scratch.value_div_u_kh_convection = 0;
		scratch.value_z_k_z_kh_pressure = 0;

		for (scratch.j=0; scratch.j < scratch.fe_values_high.get_fe().dofs_per_cell;
			++scratch.j) {

			///////////////////
			// 1. term: ∂_t v
			//

			// (∂_t v_k, [z^v-z^v_k]χ_i)
			scratch.value_dt_u_k_convection +=
				scratch.local_dtu_k_tq[scratch.j]
				* scratch.phi_convection[scratch.j];
			scratch.value_z_z_k_convection +=
				(scratch.local_z_tq[scratch.j] - scratch.local_z_k_rho_k_tq[scratch.j])
				* scratch.phi_convection[scratch.j];

			// (∂_t v_kh, [z^v_k-z^v_kh]χ_i)
			scratch.value_dt_u_kh_convection +=
				scratch.local_dtu_kh_tq[scratch.j]
				* scratch.phi_convection[scratch.j];
			scratch.value_z_k_z_kh_convection +=
				(scratch.local_z_k_rho_h_tq[scratch.j] - scratch.local_z_kh_tq[scratch.j])
				* scratch.phi_convection[scratch.j];

			////////////////////////////////////////
			// 2. term: symmetric/unsymmetric stress
			//

			if (symmetric_stress)
			{
				// symmetric stress tensor

				// 2ν (ϵ(v_k), ϵ([z^v-z^v_k]χ_i))       where ϵ(v) = 1/2(∇v + (∇v)^T)
				scratch.value_symgrad_u_k_convection +=
					scratch.local_u_k_tq[scratch.j]
					* scratch.symgrad_phi_convection[scratch.j];
				scratch.value_symgrad_z_z_k_convection +=
					(scratch.local_z_tq[scratch.j] - scratch.local_z_k_rho_k_tq[scratch.j])
					* scratch.symgrad_phi_convection[scratch.j];

				// 2ν (ϵ(v_kh), ϵ([z^v_k-z^v_kh]χ_i))   where ϵ(v) = 1/2(∇v + (∇v)^T)
				scratch.value_symgrad_u_kh_convection +=
					scratch.local_u_kh_tq[scratch.j]
					* scratch.symgrad_phi_convection[scratch.j];
				scratch.value_symgrad_z_k_z_kh_convection +=
					(scratch.local_z_k_rho_h_tq[scratch.j] - scratch.local_z_kh_tq[scratch.j])
					* scratch.symgrad_phi_convection[scratch.j];

			}
			else
			{
				// unsymmetric stress tensor

				// ν (∇ v_k, ∇{[z^v-z^v_k]χ_i})
				scratch.value_grad_u_k_convection +=
					scratch.local_u_k_tq[scratch.j]
					* scratch.grad_phi_convection[scratch.j];
				scratch.value_grad_z_z_k_convection +=
					(scratch.local_z_tq[scratch.j] - scratch.local_z_k_rho_k_tq[scratch.j])
					* scratch.grad_phi_convection[scratch.j];

				// ν (∇ v_kh, ∇{[z^v_k-z^v_kh]χ_i})
				scratch.value_grad_u_kh_convection +=
					scratch.local_u_kh_tq[scratch.j]
					* scratch.grad_phi_convection[scratch.j];
				scratch.value_grad_z_k_z_kh_convection +=
					(scratch.local_z_k_rho_h_tq[scratch.j] - scratch.local_z_kh_tq[scratch.j])
					* scratch.grad_phi_convection[scratch.j];
			}

			// -(p_k, ∇ · {[z^v-z^v_k]χ_i})
			scratch.value_u_k_pressure +=
				scratch.local_u_k_tq[scratch.j]
				* scratch.phi_pressure[scratch.j];
			scratch.value_div_z_z_k_convection +=
				(scratch.local_z_tq[scratch.j] - scratch.local_z_k_rho_k_tq[scratch.j])
				* scratch.div_phi_convection[scratch.j];

			// -(p_kh, ∇ · {[z^v_k-z^v_kh]χ_i})
			scratch.value_u_kh_pressure +=
				scratch.local_u_kh_tq[scratch.j]
				* scratch.phi_pressure[scratch.j];
			scratch.value_div_z_k_z_kh_convection +=
				(scratch.local_z_k_rho_h_tq[scratch.j] - scratch.local_z_kh_tq[scratch.j])
				* scratch.div_phi_convection[scratch.j];

			////////////////////////////////////////
			// 3. term: NSE nonlinearity
			//
			if (nonlinear)
			{
				// get ∇ v_k and ∇ v_kh
				if (symmetric_stress)
				{
					scratch.value_grad_u_k_convection +=
						scratch.local_u_k_tq[scratch.j]
						* scratch.grad_phi_convection[scratch.j];
					scratch.value_grad_u_kh_convection +=
						scratch.local_u_kh_tq[scratch.j]
						* scratch.grad_phi_convection[scratch.j];
				}

				// get v_k and v_kh
				scratch.value_u_k_convection +=
					scratch.local_u_k_tq[scratch.j]
					* scratch.phi_convection[scratch.j];
				scratch.value_u_kh_convection +=
					scratch.local_u_kh_tq[scratch.j]
					* scratch.phi_convection[scratch.j];
			}

			////////////////////////////////////////
			// 4. term: incompressibility condition
			//

			// (∇ ·v_k, [z^p-z^p_k]χ_i)
			scratch.value_div_u_k_convection +=
				scratch.local_u_k_tq[scratch.j]
				* scratch.div_phi_convection[scratch.j];
			scratch.value_z_z_k_pressure +=
				(scratch.local_z_tq[scratch.j] - scratch.local_z_k_rho_k_tq[scratch.j])
				* scratch.phi_pressure[scratch.j];

			// (∇ ·v_kh, [z^p_k-z^p_kh]χ_i)
			scratch.value_div_u_kh_convection +=
				scratch.local_u_kh_tq[scratch.j]
				* scratch.div_phi_convection[scratch.j];
			scratch.value_z_k_z_kh_pressure +=
				(scratch.local_z_k_rho_h_tq[scratch.j] - scratch.local_z_kh_tq[scratch.j])
				* scratch.phi_pressure[scratch.j];
		}

		// shape values for partition of Unity
		for (scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
			 ++ scratch.j )
		{
			scratch.chi[scratch.j] =
				scratch.fe_values_pu.shape_value(scratch.j, scratch.q);

			scratch.grad_chi[scratch.j] =
			  scratch.fe_values_pu.shape_grad(scratch.j, scratch.q);
		}

		// \int_{I_n} ... :

		for ( scratch.j = 0 ; scratch.j < scratch.fe_values_pu.get_fe().dofs_per_cell;
					++ scratch.j )
		{
			////////////////////////////////////////////////////////////
			// primal residual (time):    ρ_k(u_k ,[z-z_k]χ_i)    = ...
			//      ---> copydata.value_time
			// primal residual (space):   ρ_h(u_kh,[z_k-z_kh]χ_i) = ...
			//      ---> copydata.value_space
			//

			///////////////////
			// 1. term: ∂_t v
			//

			// (∂_t v_k, [z^v-z^v_k]χ_i)
			copydata.local_eta_k_vector[scratch.j] -= (
				1. //(1 / cell_time_tau_n) // mapping I_n -> widehat(I)
				* scratch.value_dt_u_k_convection
				* scratch.value_z_z_k_convection
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			// (∂_t v_kh, [z^v_k-z^v_kh]χ_i)
			copydata.local_eta_h_vector[scratch.j] -= (
				1. // (1 / cell_time_tau_n) // mapping I_n -> widehat(I)
				* scratch.value_dt_u_kh_convection
				* scratch.value_z_k_z_kh_convection
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			////////////////////////////////////////
			// 2. term: symmetric/unsymmetric stress
			//

			if (symmetric_stress)
			{
				// symmetric stress tensor

				// TODO: add terms with \nabla \chi_i
				// TODO: next terms...

				// 2ν (ϵ(v_k), ϵ([z^v-z^v_k]χ_i))       where ϵ(v) = 1/2(∇v + (∇v)^T)
				copydata.local_eta_k_vector[scratch.j] -= (
					2. * scratch.value_viscosity
					* scalar_product(
							scratch.value_symgrad_u_k_convection,
							scratch.value_symgrad_z_z_k_convection
					)
					* scratch.JxW * cell_time_JxW
				);

				// 2ν (ϵ(v_kh), ϵ([z^v_k-z^v_kh]χ_i))   where ϵ(v) = 1/2(∇v + (∇v)^T)
				copydata.local_eta_h_vector[scratch.j] -= (
					2. * scratch.value_viscosity
					* scalar_product(
							scratch.value_symgrad_u_kh_convection,
							scratch.value_symgrad_z_k_z_kh_convection
					)
					* scratch.JxW * cell_time_JxW
				);
			}
			else
			{
				// unsymmetric stress tensor

				// using the product rule:
				// ν(∇v_k, ∇[(z^v-z^v_k)χ_i])) = ν(∇v_k, ∇[z^v-z^v_k]χ_i) + ν(∇v_k, (z^v-z^v_k)⊗(∇χ_i))
				//  I) ν(∇v_k, ∇[z^v-z^v_k]χ_i)
				copydata.local_eta_k_vector[scratch.j] -= (
					scratch.value_viscosity
					* scalar_product(
							scratch.value_grad_u_k_convection,
							scratch.value_grad_z_z_k_convection
					)
					* scratch.chi[scratch.j]
					* scratch.JxW * cell_time_JxW
				);

				// II) ν(∇v_k, (z^v-z^v_k)⊗(∇χ_i))
				copydata.local_eta_k_vector[scratch.j] -= (
					scratch.value_viscosity
					* scalar_product(
							scratch.value_grad_u_k_convection,
							outer_product( // tensor product
									scratch.value_z_z_k_convection,
									scratch.grad_chi[scratch.j]
							)
					)
					* scratch.JxW * cell_time_JxW
				);

				// using the product rule:
				// ν(∇v_kh, ∇[(z^v_k-z^v_kh)χ_i])) = ν(∇v_kh, ∇[z^v_k-z^v_kh]χ_i) + ν(∇v_kh, (z^v_k-z^v_kh)⊗(∇χ_i))
				//  I) ν(∇v_kh, ∇[z^v_k-z^v_kh]χ_i)
				copydata.local_eta_h_vector[scratch.j] -= (
					scratch.value_viscosity
					* scalar_product(
							scratch.value_grad_u_kh_convection,
							scratch.value_grad_z_k_z_kh_convection
					)
					* scratch.chi[scratch.j]
					* scratch.JxW * cell_time_JxW
				);

				// II) ν(∇v_kh, (z^v_k-z^v_kh)⊗(∇χ_i))
				copydata.local_eta_h_vector[scratch.j] -= (
					scratch.value_viscosity
					* scalar_product(
							scratch.value_grad_u_kh_convection,
							outer_product( // tensor product
									scratch.value_z_k_z_kh_convection,
									scratch.grad_chi[scratch.j]
							)
					)
					* scratch.JxW * cell_time_JxW
				);
			}

			// using the product rule:
			// -(p_k, ∇·[(z^v-z^v_k)χ_i]) = -(p_k, ∇·[z^v-z^v_k]χ_i) -(p_k, (z^v-z^v_k) · (∇χ_i))
			//  I) -(p_k, ∇·[z^v-z^v_k]χ_i)
			copydata.local_eta_k_vector[scratch.j] += ( // plus here !
				scratch.value_u_k_pressure
				* scratch.value_div_z_z_k_convection
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			// II) -(p_k, (z^v-z^v_k) · (∇χ_i))
			copydata.local_eta_k_vector[scratch.j] += ( // plus here !
				scratch.value_u_k_pressure
				* scalar_product(
						scratch.value_z_z_k_convection,
						scratch.grad_chi[scratch.j]
				)
				* scratch.JxW * cell_time_JxW
			);

			// using the product rule:
			// -(p_kh, ∇·[(z^v_k-z^v_kh)χ_i]) = -(p_kh, ∇·[z^v_k-z^v_kh]χ_i) -(p_kh, (z^v_k-z^v_kh) · (∇χ_i))
			//  I) -(p_kh, ∇·[z^v_k-z^v_kh]χ_i)
			copydata.local_eta_h_vector[scratch.j] += ( // plus here !
				scratch.value_u_kh_pressure
				* scratch.value_div_z_k_z_kh_convection
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			// II) -(p_kh, (z^v_k-z^v_kh) · (∇χ_i))
			copydata.local_eta_h_vector[scratch.j] += ( // plus here !
				scratch.value_u_kh_pressure
				* scalar_product(
						scratch.value_z_k_z_kh_convection,
						scratch.grad_chi[scratch.j]
				)
				* scratch.JxW * cell_time_JxW
			);

			////////////////////////////////////////
			// 3. term: NSE nonlinearity
			//
			if (nonlinear)
			{
				// ((v_k ·∇)v_k, (z^v-z^v_k)χ_i)
				copydata.local_eta_k_vector[scratch.j] -= (
					scalar_product(
						(scratch.value_grad_u_k_convection * scratch.value_u_k_convection),
						scratch.value_z_z_k_convection
					)
					* scratch.chi[scratch.j]
					* scratch.JxW * cell_time_JxW
				);

				// ((v_kh ·∇)v_kh, (z^v_k-z^v_kh)χ_i)
				copydata.local_eta_h_vector[scratch.j] -= (
					scalar_product(
						(scratch.value_grad_u_kh_convection * scratch.value_u_kh_convection),
						scratch.value_z_k_z_kh_convection
					)
					* scratch.chi[scratch.j]
					* scratch.JxW * cell_time_JxW
				);
			}

			////////////////////////////////////////
			// 4. term: incompressibility condition
			//

			// (∇ ·v_k, [z^p-z^p_k]χ_i)
			copydata.local_eta_k_vector[scratch.j] -= (
				scratch.value_div_u_k_convection
				* scratch.value_z_z_k_pressure
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			// (∇ ·v_kh, [z^p_k-z^p_kh]χ_i)
			copydata.local_eta_h_vector[scratch.j] -= (
				scratch.value_div_u_kh_convection
				* scratch.value_z_k_z_kh_pressure
				* scratch.chi[scratch.j]
				* scratch.JxW * cell_time_JxW
			);

			//
			////////////////////////////////////////////////////////////////////////

			////////////////////////////////////////////////////////////////////////
			// dual residuals

			// Not implemented for Stokes, since we have an error identity
			// which contains only the primal residual.
			// For Navier-Stokes in the laminar regime the primal error estimator
			// should also be sufficient.
		}

		//
		////////////////////////////////////////////////////////////////////////
	} // for q
}

//
//
////////////////////////////////////////////////////////////////////////////////

template<int dim>
void
ErrorEstimator<dim>::copy_local_error(
	const Assembly::CopyData::ErrorEstimates<dim> &copydata) {

	pu.constraints->distribute_local_to_global(
		copydata.cell.local_eta_h_vector,
		copydata.cell.local_dof_indices_pu,
		*error_estimator.x_h
	);

	pu.constraints->distribute_local_to_global(
		copydata.cell.local_eta_k_vector,
		copydata.cell.local_dof_indices_pu,
		*error_estimator.x_k
	);
}

template<int dim>
void
ErrorEstimator<dim>::
init(
	MPI_Comm _mpi_comm,
	std::shared_ptr< dealii::Function<dim> > _viscosity,
	std::shared_ptr< fluid::Grid<dim> > _grid,
	bool use_symmetric_stress,
	bool _replace_linearization_point,
	bool _replace_weights,
	std::string _primal_order,
	std::string _dual_order,
	bool _nonlinear)
{
	mpi_comm = _mpi_comm;
	Assert(_viscosity.use_count(), dealii::ExcNotInitialized());
	function.viscosity = _viscosity;

	Assert(_grid.use_count(), dealii::ExcNotInitialized());
	grid = _grid;

	symmetric_stress = use_symmetric_stress;
	replace_linearization_points = _replace_linearization_point;
	replace_weights = _replace_weights;
	primal_order = _primal_order;
	dual_order = _dual_order;
	nonlinear = _nonlinear;

	// FEValuesExtractors
	convection = 0;
	pressure   = dim;

	n_slab = 0;
}

}}} // namespace

#include "ErrorEstimator.inst.in"
