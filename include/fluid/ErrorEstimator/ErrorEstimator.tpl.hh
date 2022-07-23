/**
 * @file ErrorEstimator.tpl.hh
 *
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Julian Roth (JR)
 *
 * @date 2022-05-02, added to fluid, JR
 * @date 2022-02-07, started working on Stokes, JR
 * @date 2020-02-07, back-merge from dwr-stokes-condiffrea, UK
 * @date 2020-02-06, new implementation with IE/II in space-time, UK
 * @date 2020-01-09, included from dwr-diffusion to dwr-condiffrea, MPB, UK
 * @date 2019-11-11, add primal AND dual residual for error indicators, MPB
 * @date 2018-11-15, add inhomogeneous Neumann, MPB, UK
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

#ifndef __ErrorEstimator_tpl_hh
#define __ErrorEstimator_tpl_hh

// PROJECT includes
#include <fluid/grid/Grid.tpl.hh>
#include <fluid/parameters/ParameterSet.hh>

// DTM++ includes
#include <DTM++/types/storage_data_vectors.tpl.hh>

// DEAL.II includes
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

// C++ includes
#include <memory>
#include <vector>
#include <string>

// namespace dwr {
namespace fluid {

namespace cGp_dGr { // primal
namespace cGq_dGs { // dual

////////////////////////////////////////////////////////////////////////////////
// TODO: split assembly scratch objects for jump tm, quad. t_q and jump t_n
////////////////////////////////////////////////////////////////////////////////

namespace Assembly {

namespace Scratch {

// NOTE: ErrorEstimateOnCell used for [*]_tm, t_q and [*]_tn
// TODO: split
template<int dim>
struct ErrorEstimateOnCell {
	ErrorEstimateOnCell(
		// space
		const dealii::DoFHandler<dim> &dof_high,
		const dealii::DoFHandler<dim> &dof_pu,
		const dealii::FiniteElement<dim> &fe_high,
		const dealii::FiniteElement<dim> &fe_pu,
		const dealii::Mapping<dim> &mapping,
		const dealii::Quadrature<dim> &quad,
		const dealii::UpdateFlags &uflags
	);

	ErrorEstimateOnCell(const ErrorEstimateOnCell &scratch);

	// data structures of current cell
	dealii::FEValues<dim>               fe_values_high;
	dealii::FEValues<dim>               fe_values_pu;

	const dealii::DoFHandler<dim> &dof_high;
	const dealii::DoFHandler<dim> &dof_pu;

	std::vector< dealii::types::global_dof_index > local_dof_indices_high;
	std::vector< dealii::types::global_dof_index > local_dof_indices_pu;

	//partition of unity shape functions
	std::vector<double>                 chi;
	std::vector<dealii::Tensor<1,dim> > grad_chi;

	// shape fun scratch:
	std::vector< dealii::Tensor<1,dim> >           phi_convection;
	std::vector< dealii::Tensor<2,dim> >           grad_phi_convection;
	std::vector< dealii::SymmetricTensor<2,dim> >  symgrad_phi_convection;
	std::vector<double>                            div_phi_convection;
	std::vector<double>                            phi_pressure;

	////////////////////////////////////////////////////////////////////////////
	// local dof scratch: (jump terms in time)

	// u_kh := u_kh^(1,1) = u_kh^low = I_k(I_h(u_kh^high))
	std::vector<double>                 local_u_kh_m; // u_kh(t^-)
	std::vector<double>                 local_u_kh_p; // u_kh(t^+)

	// u_k := u_kh^(1,2) = E_2h(u_kh^low) = I_k(u_kh^high)
	std::vector<double>                 local_u_k_m; // u_k(t^-)
	std::vector<double>                 local_u_k_p; // u_k(t^+)

	// NOTE: choices for z, z_k and z_kh depend on whether we linearize the weights (see also description in ErrorEstimator.tpl.cc)
	// e.g. it can be that local_z_k_rho_k_p != local_z_k_rho_h_p

	// z - z_k to evaluate temporal error ρ_k(.,z-z_k)
	std::vector<double> 					local_z_p;           // z(t^+)
	std::vector<double> 					local_z_k_rho_k_p;   // z_k(t^+) [for ρ_k]

	// z_k - z_kh to evaluate spatial error ρ_h(.,z_k-z_kh)
	std::vector<double>					local_z_k_rho_h_p;   // z_k(t^+) [for ρ_h]
	std::vector<double>					local_z_kh_p;    	 // z_kh(t^+)

	////////////////////////////////////////////////////////////////////////////
	// local dof scratch: (on a time quadrature point)

	// u_kh := u_kh^(1,1) = u_kh^low = I_k(I_h(u_kh^high))
	std::vector<double>                 local_u_kh_tq;      // u_kh(t_q)
	std::vector<double>                 local_dtu_kh_tq;    // ∂_t u_kh(t_q)

	// u_k := u_kh^(1,2) = E_2h(u_kh^low) = I_k(u_kh^high)
	std::vector<double>                 local_u_k_tq;      // u_k(t_q)
	std::vector<double>                 local_dtu_k_tq;    // ∂_t u_k(t_q)

	std::vector<double>                 local_z_tq;     		 // z(t_q)
	std::vector<double>                 local_z_k_rho_k_tq;      // z_k(t_q) [for ρ_k]
	std::vector<double>                 local_z_k_rho_h_tq;      // z_k(t_q) [for ρ_h]
	std::vector<double>                 local_z_kh_tq;      	// z_kh(t_q)

	////////////////////////////////////////////////////////////////////////////
	// function eval scratch:
	double value_viscosity;

	////////////////////////////////////////////////////////////////////////////
	// local solution eval

	// NOTE: I sometimes omit the PU when writing the adjoint error terms. --> I don't remember whether I actually did.

	// t_m:
	dealii::Tensor<1,dim> value_jump_u_kh_convection;   // [ v_kh(t) ]_t = v_kh(t^+) - v_kh(t^-)
	dealii::Tensor<1,dim> value_jump_u_k_convection;    // [ v_k(t) ]_t  = v_k(t^+)  - v_k(t^-)
	dealii::Tensor<1,dim> value_z_z_k_convection;    // z^v(t) - z^v_k(t)
	dealii::Tensor<1,dim> value_z_k_z_kh_convection; // z^v_k(t) - z^v_kh(t)
	double value_z_z_k_pressure;    				 // z^p(t) - z^p_k(t)
	double value_z_k_z_kh_pressure;	 			 	 // z^p_k(t) - z^p_kh(t)

	// NOTE: t_n only needed for adjoint error estimator

	// t_q: primal residuals:
	dealii::Tensor<1,dim> value_dt_u_k_convection;    					//    ∂_t v_k(t)
	dealii::Tensor<2,dim> value_grad_z_z_k_convection;    				//    ∇(z  -z_k )(t)
	dealii::Tensor<1,dim> value_dt_u_kh_convection;    					//    ∂_t v_kh(t)
	dealii::Tensor<2,dim> value_grad_z_k_z_kh_convection;				//    ∇(z_k-z_kh)(t)
	dealii::SymmetricTensor<2,dim> value_symgrad_z_z_k_convection;    	//    ϵ(z  -z_k )(t)  where ϵ(v) = 1/2(∇v + (∇v)^T)
	dealii::SymmetricTensor<2,dim> value_symgrad_z_k_z_kh_convection;	//    ϵ(z_k-z_kh)(t)  where ϵ(v) = 1/2(∇v + (∇v)^T)
	double value_div_z_z_k_convection;       		      				// ∇ · (z  -z_k )(t)
	double value_div_z_k_z_kh_convection;        	  	  				// ∇ · (z_k-z_kh)(t)
	dealii::Tensor<2,dim> value_grad_u_k_convection;      				// ∇v_k(t)
	dealii::Tensor<2,dim> value_grad_u_kh_convection; 	  				// ∇v_kh(t)
	dealii::SymmetricTensor<2,dim> value_symgrad_u_k_convection;    	// 1/2(∇v_k + (∇v_k)^T)(t)
	dealii::SymmetricTensor<2,dim> value_symgrad_u_kh_convection; 		// 1/2(∇v_kh + (∇v_kh)^T)(t)
	double value_u_k_pressure;                            				// ∇p_k(t)
	double value_u_kh_pressure;                          				// ∇p_kh(t)
	dealii::Tensor<1,dim> value_u_k_convection;    						//  v_k(t)
	dealii::Tensor<1,dim> value_u_kh_convection;						//  v_kh(t)
	double value_div_u_k_convection;					  				// ∇ · u_k(t)
	double value_div_u_kh_convection;					  			    // ∇ · u_kh(t)

	// t_q: dual residuals:
	// TODO

	////////////////////////////////////////////////////////////////////////////
	// other:
	double JxW;

	unsigned int q;
	unsigned int d;
	unsigned int j;
};

template<int dim>
struct ErrorEstimates {
	ErrorEstimates(
			const dealii::DoFHandler<dim>    &dof_high,
			const dealii::DoFHandler<dim>    &dof_pu,
			const dealii::FiniteElement<dim> &fe_high,
			const dealii::FiniteElement<dim> &fe_pu,
			const dealii::Mapping<dim>       &mapping,
			const dealii::Quadrature<dim>    &quad_cell,
			const dealii::UpdateFlags        &uflags_cell
	);

	ErrorEstimates(const ErrorEstimates &scratch);

	ErrorEstimateOnCell<dim> cell;
};

} // namespace Scratch

namespace CopyData {

/// Struct for copydata on local cell matrix.
template<int dim>
struct ErrorEstimateOnCell{
	ErrorEstimateOnCell(const dealii::FiniteElement<dim> &fe);
	ErrorEstimateOnCell(const ErrorEstimateOnCell &copydata);

	dealii::Vector<double> local_eta_h_vector;
	dealii::Vector<double> local_eta_k_vector;
	std::vector< dealii::types::global_dof_index > local_dof_indices_pu;
};


template<int dim>
struct ErrorEstimates {
	ErrorEstimates(const dealii::FiniteElement<dim> &fe);
	ErrorEstimates(const ErrorEstimates &copydata);

	ErrorEstimateOnCell<dim> cell;
};

} // namespace CopyData

} // namespace Assembly
////////////////////////////////////////////////////////////////////////////////


template<int dim>
class ErrorEstimator {
public:
	ErrorEstimator() = default;
	virtual ~ErrorEstimator() = default;

	virtual void estimate_on_slab(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		const typename DTM::types::storage_data_vectors<1>::iterator &u,
		const typename DTM::types::storage_data_vectors<1>::iterator &um,
		const typename DTM::types::storage_data_vectors<1>::iterator &z,

		const typename DTM::types::storage_data_vectors<1>::iterator &eta_s,
		const typename DTM::types::storage_data_vectors<1>::iterator &eta_t
	);

	void init(
		std::shared_ptr< dealii::Function<dim> > _viscosity,
		std::shared_ptr< fluid::Grid<dim> > _grid,
		bool use_symmetric_stress,
		bool _replace_linearization_point,
		bool _replace_weights,
		std::string _primal_order,
		std::string _dual_order,
		bool _nonlinear
	);

protected:
	/// evaluate solution w(t), where w is either the primal solution u or the dual solution z
	virtual void get_w_t(
		std::shared_ptr< dealii::FiniteElement<1> > time_fe,
		std::shared_ptr< dealii::Mapping<1> > time_mapping,
		std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
		const typename dealii::DoFHandler<1>::active_cell_iterator &cell_time,
		std::shared_ptr< dealii::Vector<double> > w,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &w_t
	);

	/// evaluate solution ∂_t w(t), where w is either the primal solution u or the dual solution z
	virtual void get_dt_w_t(
		std::shared_ptr< dealii::FiniteElement<1> > time_fe,
		std::shared_ptr< dealii::Mapping<1> > time_mapping,
		std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
		const typename dealii::DoFHandler<1>::active_cell_iterator &cell_time,
		std::shared_ptr< dealii::Vector<double> > w,
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &dt_w_t
	);

	/// interpolate vector w from low spatial space to high spatial space
	virtual void interpolate_space(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double> > w,
		std::shared_ptr< dealii::Vector<double> > &interpolated_space_w
	);

	/// patchwise high order interpolate vector w from low spatial space to high spatial space
	virtual void patchwise_high_order_interpolate_space(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double> > w,
		std::shared_ptr< dealii::Vector<double> > &higher_order_space_w
	);

	/// higher order reconstructed solution w(t) in time, where w is either the primal solution u or the dual solution z
	virtual void get_patchwise_higher_order_time_w_t(
		std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
		std::shared_ptr< dealii::Vector<double> > w2,
		const double &a, // slab start point
		const double &b, // slab   end point
		const double &t,
		std::shared_ptr< dealii::Vector<double> > &higher_order_time_w_t
	);

	/// restrict vector w from high spatial space to low spatial space and then interpolate back to high spatial space
	virtual void back_interpolate_space(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::Vector<double> > w,
		std::shared_ptr< dealii::Vector<double> > &back_interpolated_space_w
	);

	/// take the entire solution on the space slab_w, restrict it back in time and then interpolate it back in time again
	virtual void get_back_interpolated_time_slab_w(
		const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
		std::shared_ptr< dealii::DoFHandler<dim> > space_dof,
		std::shared_ptr< dealii::Vector<double> > slab_w,
		std::shared_ptr< dealii::Vector<double> > &back_interpolated_time_slab_w
	);

	////////////////////////////////////////////////////////////////////////////
	// assemble local functions:
	//

	virtual void assemble_local_error_tm(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimates<dim> &scratch,
		Assembly::CopyData::ErrorEstimates<dim> &copydata
	);

	virtual void assemble_error_on_cell_tm(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata
	);


	virtual void assemble_local_error_tn(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimates<dim> &scratch,
		Assembly::CopyData::ErrorEstimates<dim> &copydata
	);

	virtual void assemble_error_on_cell_tn(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata
	);


	virtual void assemble_local_error(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimates<dim> &scratch,
		Assembly::CopyData::ErrorEstimates<dim> &copydata
	);

	virtual void assemble_error_on_cell(
		const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
		Assembly::Scratch::ErrorEstimateOnCell<dim> &scratch,
		Assembly::CopyData::ErrorEstimateOnCell<dim> &copydata
	);

	virtual void copy_local_error(
		const Assembly::CopyData::ErrorEstimates<dim> &copydata
	);

	////////////////////////////////////////////////////////////////////////////
	// internal data structures:
	//

	std::shared_ptr< fluid::ParameterSet > parameter_set;
	std::shared_ptr< fluid::Grid<dim> > grid;

	bool nonlinear;

	struct {
		std::shared_ptr< dealii::Function<dim> > viscosity;
	} function;

	struct {
		std::shared_ptr< dealii::Vector<double> > um_on_tn; // u(t_n^-)
	} primal;

	struct {
		bool solution_can_be_patchwise_interpolated;
	} low;

    struct {
    	std::shared_ptr< dealii::AffineConstraints<double> > constraints;
    } pu;

	struct {
		std::shared_ptr<dealii::Vector<double> > x_h;
		std::shared_ptr<dealii::Vector<double> > x_k;
	} error_estimator;

	double cell_time_tau_n;
	double cell_time_JxW; // tau_n x w_q
	int cell_time_index;

	////////////////////////////////////////////////////////////////////////////
	// solution and dual solution vectors on tm, tq, tn
	//

	// u_kh(t_m^-) = u_kh^(1,1)(t_m^-) [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_kh_m_on_tm;
	// u_k(t_m^-) = u_kh^(1,2)(t_m^-)  [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_k_m_on_tm;

	// u_kh(t_m^+) = u_kh^(1,1)(t_m^+) [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_kh_p_on_tm;
	// u_k(t_m^+) = u_kh^(1,2)(t_m^+)  [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_k_p_on_tm;

	// z(t_m^+)              [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_p_on_tm;
	// z_k(t_m^+) for ρ_k    [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_k_rho_k_p_on_tm;
	// z_k(t_m^+) for ρ_h    [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_k_rho_h_p_on_tm;
	// z_kh(t_m^+)           [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_kh_p_on_tm;
	// z_kh(t_m^+)           [of low length]
	std::shared_ptr< dealii::Vector<double> > low_z_kh_p_on_tm;

	// u_k(t_q)                	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_k_on_tq;
	// ∂_t u_k(t_q)              [of high length]
	std::shared_ptr< dealii::Vector<double> > high_dt_u_k_on_tq;
	// u_kh(t_q)                 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_u_kh_on_tq;
	// ∂_t u_kh(t_q)           	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_dt_u_kh_on_tq;

	// z(t_q)                	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_on_tq;
	// z_k(t_q) for ρ_k       	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_k_rho_k_on_tq;
	// z_k(t_q) for ρ_h      	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_k_rho_h_on_tq;
	// z_kh(t_q)              	 [of high length]
	std::shared_ptr< dealii::Vector<double> > high_z_kh_on_tq;
	// z_kh(t_q)              	 [of low length]
	std::shared_ptr< dealii::Vector<double> > low_z_kh_on_tq;

	////////////////////////////////////////////////////////////////////////////
	// other data structures
	//

	dealii::FEValuesExtractors::Vector convection;
	dealii::FEValuesExtractors::Scalar pressure;

	bool symmetric_stress;
	bool replace_linearization_points = true;
	bool replace_weights = true;

	std::string primal_order;
	std::string dual_order;
};

}}} // namespace

#endif
