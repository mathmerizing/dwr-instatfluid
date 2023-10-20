/**
 * @file ST_DivFreeProjectionAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 *
 * @date 2022-05-18, divergence free projection, JR
 * @date 2021-11-05, dynamics for fluid, JR
 * @date 2019-11-21, space-time fluid, UK
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
#include <fluid/assembler/ST_DivFreeProjectionAssembly.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>

// C++ includes
#include <functional>

namespace projection {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template <int dim>
DivFreeProjectionAssembly<
    dim>::DivFreeProjectionAssembly(  // @suppress("Class members should be
                                      // properly initialized") space
    const dealii::FiniteElement<dim> &fe_space,
    const dealii::Mapping<dim> &mapping_space,
    const dealii::Quadrature<dim> &quad_space)
    :  // init space
      space_fe_values(mapping_space, fe_space, quad_space,
                      dealii::update_values | dealii::update_gradients |
                          dealii::update_JxW_values |
                          dealii::update_quadrature_points),
      space_grad_phi(fe_space.dofs_per_cell),
      space_phi(fe_space.dofs_per_cell),
      space_div_phi(fe_space.dofs_per_cell),
      space_psi(fe_space.dofs_per_cell),
      space_local_dof_indices(fe_space.dofs_per_cell) {}

template <int dim>
DivFreeProjectionAssembly<dim>::DivFreeProjectionAssembly(
    const DivFreeProjectionAssembly &scratch)
    : space_fe_values(scratch.space_fe_values.get_mapping(),
                      scratch.space_fe_values.get_fe(),
                      scratch.space_fe_values.get_quadrature(),
                      scratch.space_fe_values.get_update_flags()),
      space_grad_phi(scratch.space_grad_phi),
      space_phi(scratch.space_phi),
      space_div_phi(scratch.space_div_phi),
      space_psi(scratch.space_psi),
      space_dofs_per_cell(scratch.space_dofs_per_cell),
      space_JxW(scratch.space_JxW),
      space_local_dof_indices(scratch.space_local_dof_indices),
      //
      viscosity(scratch.viscosity) {}

}  // namespace Scratch

namespace CopyData {

template <int dim>
DivFreeProjectionAssembly<dim>::DivFreeProjectionAssembly(
    const dealii::FiniteElement<dim> &fe_s)
    : matrix(fe_s.dofs_per_cell, fe_s.dofs_per_cell),
      local_dof_indices(fe_s.dofs_per_cell) {}

template <int dim>
DivFreeProjectionAssembly<dim>::DivFreeProjectionAssembly(
    const DivFreeProjectionAssembly &copydata)
    : matrix(copydata.matrix), local_dof_indices(copydata.local_dof_indices) {}

}  // namespace CopyData
}  // namespace Assembly
////////////////////////////////////////////////////////////////////////////////

template <int dim>
void Assembler<dim>::set_functions(
    std::shared_ptr<dealii::Function<dim> > viscosity) {
  function.viscosity = viscosity;
}

template <int dim>
void Assembler<dim>::set_symmetric_stress(bool use_symmetric_stress) {
  symmetric_stress = use_symmetric_stress;
}

template <int dim>
void Assembler<dim>::set_gradient_projection(bool use_gradient_projection) {
  gradient_projection = use_gradient_projection;
}

template <int dim>
void Assembler<dim>::assemble(
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _L,
    const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
  ////////////////////////////////////////////////////////////////////////////
  // check
  Assert(dim == 2 || dim == 3, dealii::ExcNotImplemented());

  Assert(_L.use_count(), dealii::ExcNotInitialized());

  Assert(slab->space.primal.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->fe.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->constraints.use_count(),
         dealii::ExcNotInitialized());

  //	Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());

  ////////////////////////////////////////////////////////////////////////////
  // init

  L = _L;

  space.dof = slab->space.primal.fe_info->dof;
  space.fe = slab->space.primal.fe_info->fe;
  space.mapping = slab->space.primal.fe_info->mapping;
  space.constraints = slab->space.primal.fe_info->initial_constraints;

  // FEValuesExtractors
  convection = 0;
  pressure = dim;

  // check fe: ((fluid)) = ((FE_Q^d, FE_Q))
  Assert((space.fe->n_base_elements() == 1),
         dealii::ExcMessage("fe not correct (fluid system)"));

  Assert((space.fe->base_element(0).n_base_elements() == 2),
         dealii::ExcMessage("fe: (fluid) not correct (FEQ+FEQ system)"));

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
      Assembly::Scratch::DivFreeProjectionAssembly<dim>(
          *space.fe, *space.mapping, quad_space),
      Assembly::CopyData::DivFreeProjectionAssembly<dim>(*space.fe));

  L->compress(dealii::VectorOperation::add);
}

/// Local assemble on cell.
template <int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::DivFreeProjectionAssembly<dim> &scratch,
    Assembly::CopyData::DivFreeProjectionAssembly<dim> &copydata) {
  cell->get_dof_indices(scratch.space_local_dof_indices);
  scratch.space_fe_values.reinit(cell);

  copydata.matrix = 0;

  // dof mapping
  for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
    copydata.local_dof_indices[i] = scratch.space_local_dof_indices[i];

  // prefetch data

  // assemble: volume
  for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
       ++q) {
    for (unsigned int k{0}; k < space.fe->dofs_per_cell; ++k) {
      scratch.space_grad_phi[k] =
          scratch.space_fe_values[convection].gradient(k, q);
      scratch.space_phi[k] = scratch.space_fe_values[convection].value(k, q);
      scratch.space_div_phi[k] =
          scratch.space_fe_values[convection].divergence(k, q);
      scratch.space_psi[k] = scratch.space_fe_values[pressure].value(k, q);
    }

    for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
      for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
        copydata.matrix(i, j) +=
            // convection A_bb
            ((gradient_projection
                  ? (scalar_product(scratch.space_grad_phi[i],
                                    //											scratch.viscosity
                                    //*
                                    scratch.space_grad_phi[j]))
                  : (scratch.space_phi[i] * scratch.space_phi[j])) *

             scratch.space_fe_values.JxW(q))
            // pressure B_bp
            - (scratch.space_div_phi[i] * scratch.space_psi[j] *
               scratch.space_fe_values.JxW(q))
            // div-free constraint B_pb
            + (scratch.space_psi[i] * scratch.space_div_phi[j] *
               scratch.space_fe_values.JxW(q));
      }
  }  // x_q
}

/// Copy local assembly to global matrix.
template <int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::DivFreeProjectionAssembly<dim> &copydata) {
  //	Assert(copydata.matrix.size(), dealii::ExcNotInitialized());

  space.constraints->distribute_local_to_global(copydata.matrix,
                                                copydata.local_dof_indices, *L);
}

}  // namespace Operator
}  // namespace spacetime
}  // namespace projection

#include "ST_DivFreeProjectionAssembly.inst.in"
