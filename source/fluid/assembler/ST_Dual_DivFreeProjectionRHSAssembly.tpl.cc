/**
 * @file ST_Dual_DivFreeProjectionRHSAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @authro Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @date 2022-06-02, dual divergence free projection, JR
 * @date 2022-05-18, divergence free projection, JR
 * @date 2021-11-05, cleanups, UK
 * @date 2021-11-05, initialvalue for ST fluid, JR
 *
 * @date 2020-01-08, supg, MPB
 * @date 2019-09-18, space-time initialvalue, UK
 * @date 2019-09-13, space-time force, UK
 * @date 2019-08-30, space-time diffusion, UK
 * @date 2019-01-28, space-time parabolic, UK
 * @date 2018-03-08, included from ewave, UK
 * @date 2017-10-23, ewave, UK
 * @date 2015-05-18, AWAVE/C++.11, UK
 * @date 2014-04-09, Tensor, UK
 * @date 2012-03-13, UK
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
#include <fluid/assembler/ST_Dual_DivFreeProjectionRHSAssembly.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

// C++ includes
#include <functional>

namespace projectionrhs {
namespace spacetime {
namespace dual {
namespace Operator {

namespace Assembly {
namespace Scratch {

template <int dim>
DualDivFreeProjectionRHSAssembly<dim>::DualDivFreeProjectionRHSAssembly(
    // space
    const dealii::FiniteElement<dim> &fe_space,
    const dealii::Mapping<dim> &mapping_space,
    const dealii::Quadrature<dim> &quad_space)
    :  // init space
      space_fe_values(mapping_space, fe_space, quad_space,
                      dealii::update_values | dealii::update_gradients |
                          dealii::update_JxW_values),
      space_grad_phi(fe_space.dofs_per_cell),
      space_phi(fe_space.dofs_per_cell),
      space_local_dof_indices(fe_space.dofs_per_cell) {}

template <int dim>
DualDivFreeProjectionRHSAssembly<dim>::DualDivFreeProjectionRHSAssembly(
    const DualDivFreeProjectionRHSAssembly &scratch)
    : space_fe_values(scratch.space_fe_values.get_mapping(),
                      scratch.space_fe_values.get_fe(),
                      scratch.space_fe_values.get_quadrature(),
                      scratch.space_fe_values.get_update_flags()),
      space_grad_phi(scratch.space_grad_phi),
      space_phi(scratch.space_phi),
      space_local_dof_indices(scratch.space_local_dof_indices),
      //
      zn(scratch.zn),
      grad_zn(scratch.grad_zn) {}

}  // namespace Scratch

namespace CopyData {

template <int dim>
DualDivFreeProjectionRHSAssembly<dim>::DualDivFreeProjectionRHSAssembly(
    const dealii::FiniteElement<dim> &fe_s)
    : vi_zn_vector(fe_s.dofs_per_cell), local_dof_indices(fe_s.dofs_per_cell) {}

template <int dim>
DualDivFreeProjectionRHSAssembly<dim>::DualDivFreeProjectionRHSAssembly(
    const DualDivFreeProjectionRHSAssembly &copydata)
    : vi_zn_vector(copydata.vi_zn_vector),
      local_dof_indices(copydata.local_dof_indices) {}

}  // namespace CopyData
}  // namespace Assembly

template <int dim>
void Assembler<dim>::set_gradient_projection(bool use_gradient_projection) {
  gradient_projection = use_gradient_projection;
}

template <int dim>
void Assembler<dim>::assemble(
    std::shared_ptr<dealii::Vector<double> > _zn,   // input
    std::shared_ptr<dealii::Vector<double> > _Mzn,  // output
    const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
  ////////////////////////////////////////////////////////////////////////////
  // check
  Assert(_zn.use_count(), dealii::ExcNotInitialized());
  Assert(_Mzn.use_count(), dealii::ExcNotInitialized());

  Assert(slab->space.dual.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->fe.use_count(), dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->constraints.use_count(),
         dealii::ExcNotInitialized());

  ////////////////////////////////////////////////////////////////////////////
  // init

  // FEValuesExtractors
  convection = 0;
  //	pressure = dim;

  zn = _zn;
  Mzn = _Mzn;

  space.dof = slab->space.dual.fe_info->dof;
  space.fe = slab->space.dual.fe_info->fe;
  space.mapping = slab->space.dual.fe_info->mapping;
  space.constraints = slab->space.dual.fe_info->constraints;

  ////////////////////////////////////////////////////////////////////////////
  // WorkStream assemble

  const dealii::QGauss<dim> quad_space(
      std::max(
          std::max(space.fe->base_element(0).base_element(0).tensor_degree(),
                   space.fe->base_element(0).base_element(1).tensor_degree()),
          static_cast<unsigned int>(1)) +
      2);

  typedef dealii::FilteredIterator<
      const typename dealii::DoFHandler<dim>::active_cell_iterator>
      CellFilter;

  // Using WorkStream to assemble.
  dealii::WorkStream::run(
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(),
                 space.dof->begin_active()),
      CellFilter(dealii::IteratorFilters::LocallyOwnedCell(), space.dof->end()),
      std::bind(&Assembler<dim>::local_assemble_cell, this,
                std::placeholders::_1, std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&Assembler<dim>::copy_local_to_global_cell, this,
                std::placeholders::_1),
      Assembly::Scratch::DualDivFreeProjectionRHSAssembly<dim>(
          *space.fe, *space.mapping, quad_space),
      Assembly::CopyData::DualDivFreeProjectionRHSAssembly<dim>(*space.fe));
}

/// Local assemble on cell.
template <int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::DualDivFreeProjectionRHSAssembly<dim> &scratch,
    Assembly::CopyData::DualDivFreeProjectionRHSAssembly<dim> &copydata) {
  cell->get_dof_indices(scratch.space_local_dof_indices);
  scratch.space_fe_values.reinit(cell);

  copydata.vi_zn_vector = 0;

  // dof mapping
  for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
    copydata.local_dof_indices[i] = scratch.space_local_dof_indices[i];
  }

  for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
       ++q) {
    scratch.zn = 0;
    scratch.grad_zn = 0;
    // prefetch
    for (unsigned int k{0}; k < space.fe->dofs_per_cell; ++k) {
      scratch.space_grad_phi[k] =
          scratch.space_fe_values[convection].gradient(k, q);
      scratch.space_phi[k] = scratch.space_fe_values[convection].value(k, q);
    }

    // get zn
    for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
      if (gradient_projection)
        scratch.grad_zn += (*zn)[scratch.space_local_dof_indices[j]] *
                           scratch.space_grad_phi[j];
      else
        scratch.zn +=
            (*zn)[scratch.space_local_dof_indices[j]] * scratch.space_phi[j];
    }

    // assemble (v, ϕ^v) or (∇v, ∇ϕ^v)
    for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
      copydata.vi_zn_vector(i) +=
          ((gradient_projection
                ? (scalar_product(scratch.space_grad_phi[i], scratch.grad_zn))
                : (scratch.space_phi[i] * scratch.zn)) *
           scratch.space_fe_values.JxW(q));
    }
  }  // x_q
}

template <int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::DualDivFreeProjectionRHSAssembly<dim> &copydata) {
  space.constraints->distribute_local_to_global(
      copydata.vi_zn_vector, copydata.local_dof_indices, *Mzn);
}

}  // namespace Operator
}  // namespace dual
}  // namespace spacetime
}  // namespace projectionrhs

#include "ST_Dual_DivFreeProjectionRHSAssembly.inst.in"
