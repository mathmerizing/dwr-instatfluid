/**
 * @file ST_InitialValueAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 *
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-05, cleanups, UK
 * @date 2021-11-05, initialvalue for ST Stokes, JR
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
#include <fluid/assembler/ST_InitialValueAssembly.tpl.hh>

// deal.II includes
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

// C++ includes
#include <functional>

namespace initialvalue {
namespace spacetime {
namespace Operator {

namespace Assembly {
namespace Scratch {

template <int dim>
InitialValueAssembly<dim>::InitialValueAssembly(
    // space
    const dealii::FiniteElement<dim> &fe_space,
    const dealii::Mapping<dim> &mapping_space,
    const dealii::Quadrature<dim> &quad_space,
    // time
    const dealii::FiniteElement<1> &fe_time,
    const dealii::Mapping<1> &mapping_time,
    const dealii::Quadrature<1> &face_nodes)
    :  // init space
      space_fe_values(mapping_space, fe_space, quad_space,
                      dealii::update_values | dealii::update_JxW_values),
      space_local_dof_indices(fe_space.dofs_per_cell),
      // init time
      time_fe_face_values(mapping_time, fe_time, face_nodes,
                          dealii::update_values),
      time_local_dof_indices(fe_time.dofs_per_cell) {}

template <int dim>
InitialValueAssembly<dim>::InitialValueAssembly(
    const InitialValueAssembly &scratch)
    : space_fe_values(scratch.space_fe_values.get_mapping(),
                      scratch.space_fe_values.get_fe(),
                      scratch.space_fe_values.get_quadrature(),
                      scratch.space_fe_values.get_update_flags()),
      space_local_dof_indices(scratch.space_local_dof_indices),
      //
      time_fe_face_values(scratch.time_fe_face_values.get_mapping(),
                          scratch.time_fe_face_values.get_fe(),
                          scratch.time_fe_face_values.get_quadrature(),
                          scratch.time_fe_face_values.get_update_flags()),
      time_local_dof_indices(scratch.time_local_dof_indices),
      //
      um(scratch.um) {}

}  // namespace Scratch

namespace CopyData {

template <int dim>
InitialValueAssembly<dim>::InitialValueAssembly(
    const dealii::FiniteElement<dim> &fe_s,
    const dealii::FiniteElement<1> &fe_t)
    : vi_um_vector(fe_s.dofs_per_cell * fe_t.dofs_per_cell),
      local_dof_indices(fe_s.dofs_per_cell * fe_t.dofs_per_cell) {}

template <int dim>
InitialValueAssembly<dim>::InitialValueAssembly(
    const InitialValueAssembly &copydata)
    : vi_um_vector(copydata.vi_um_vector),
      local_dof_indices(copydata.local_dof_indices) {}

}  // namespace CopyData
}  // namespace Assembly

template <int dim>
void Assembler<dim>::assemble(
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> _um,   // input
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> _Mum,  // output
    const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab) {
  ////////////////////////////////////////////////////////////////////////////
  // check
  Assert(_um.use_count(), dealii::ExcNotInitialized());
  Assert(_Mum.use_count(), dealii::ExcNotInitialized());

  Assert(slab->space.primal.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->fe.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->time.primal.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->time.primal.fe_info->fe.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->time.primal.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->spacetime.primal.constraints.use_count(),
         dealii::ExcNotInitialized());

  ////////////////////////////////////////////////////////////////////////////
  // init

  // FEValuesExtractors
  convection = 0;
  //	pressure = dim;

  um = _um;
  Mum = _Mum;

  space.dof = slab->space.primal.fe_info->dof;
  space.fe = slab->space.primal.fe_info->fe;
  space.mapping = slab->space.primal.fe_info->mapping;

  time.dof = slab->time.primal.fe_info->dof;
  time.fe = slab->time.primal.fe_info->fe;
  time.mapping = slab->time.primal.fe_info->mapping;

  spacetime.constraints = slab->spacetime.primal.hanging_node_constraints;

  ////////////////////////////////////////////////////////////////////////////
  // WorkStream assemble

  const dealii::QGauss<dim> quad_space(
      std::max(
          std::max(space.fe->base_element(0).base_element(0).tensor_degree(),
                   space.fe->base_element(0).base_element(1).tensor_degree()),
          static_cast<unsigned int>(1)) +
      2);

  const dealii::QGaussLobatto<1> face_nodes(2);

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
      Assembly::Scratch::InitialValueAssembly<dim>(*space.fe, *space.mapping,
                                                   quad_space, *time.fe,
                                                   *time.mapping, face_nodes),
      Assembly::CopyData::InitialValueAssembly<dim>(*space.fe, *time.fe));
  Mum->compress(dealii::VectorOperation::add);
}

/// Local assemble on cell.
template <int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::InitialValueAssembly<dim> &scratch,
    Assembly::CopyData::InitialValueAssembly<dim> &copydata) {
  cell->get_dof_indices(scratch.space_local_dof_indices);
  scratch.space_fe_values.reinit(cell);

  auto cell_time = time.dof->begin_active();
  auto endc_time = time.dof->end();

  for (; cell_time != endc_time; ++cell_time) {
    copydata.vi_um_vector = 0;

    // dof mapping
    cell_time->get_dof_indices(scratch.time_local_dof_indices);
    for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
      for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii) {
        copydata.local_dof_indices[i + ii * space.fe->dofs_per_cell] =
            scratch.space_local_dof_indices[i] +
            scratch.time_local_dof_indices[ii] * space.dof->n_dofs();
      }

    // assemble: face (w^+ * u^-) only on the first time cell of Q_n
    scratch.time_fe_face_values.reinit(cell_time);

    for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
         ++q) {
      scratch.um = 0;
      for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
        for (unsigned int k{0}; k < dim; k++) {
          scratch.um[k] += (*um)[scratch.space_local_dof_indices[j]] *
                           scratch.space_fe_values[convection].value(j, q)[k];
        }
      }

      for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
        for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i) {
          copydata.vi_um_vector(i + ii * space.fe->dofs_per_cell) +=
              // trace operator: (+) w(x,t_0)^+ * u(x,t_0)^-
              (scratch.space_fe_values[convection].value(i,
                                                         q) *  // TODO: prefetch
               scratch.time_fe_face_values.shape_value(ii, 0) *

               scratch.um *

               scratch.space_fe_values.JxW(q)  // TODO: prefetch
              );
        }
    }  // x_q

    break;
  }
}

template <int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::InitialValueAssembly<dim> &copydata) {
  spacetime.constraints->distribute_local_to_global(
      copydata.vi_um_vector, copydata.local_dof_indices, *Mum);
}

}  // namespace Operator
}  // namespace spacetime
}  // namespace initialvalue

#include "ST_InitialValueAssembly.inst.in"
