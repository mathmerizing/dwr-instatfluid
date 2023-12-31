/**
 * @file ST_Dual_FluidAssembly.tpl.cc
 * @author Uwe Koecher (UK)
 * @author Julian Roth (JR)
 * @author G. Kanschat, W. Bangerth and the deal.II authors
 * @author Jan Philipp Thiele (JPT)
 *
 * @Date 2022-05-13, nonlinearity in dual, JR
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-12-20, dual assembler, JR
 * @date 2021-11-05, dynamics for stokes, JR
 *
 * @date 2019-11-21, space-time stokes, UK
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
#include <fluid/QRightBox.tpl.hh>
#include <fluid/assembler/ST_Dual_FluidAssembly.tpl.hh>

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

namespace fluid {
namespace spacetime {
namespace dual {
namespace Operator {

namespace Assembly {
namespace Scratch {

template <int dim>
FluidAssembly<dim>::FluidAssembly(  // @suppress("Class members should be
                                    // properly initialized")
    const dealii::DoFHandler<dim> &dof_dual,
    const dealii::DoFHandler<dim> &dof_primal,
    // space
    const dealii::FiniteElement<dim> &fe_space,
    const dealii::Mapping<dim> &mapping_space,
    const dealii::Quadrature<dim> &quad_space,
    // time
    const dealii::FiniteElement<1> &fe_time,
    const dealii::Mapping<1> &mapping_time,
    const dealii::Quadrature<1> &quad_time,
    const dealii::Quadrature<1> &face_nodes,
    // primal space
    const dealii::FiniteElement<dim> &primal_fe_space,
    const dealii::Mapping<dim> &primal_mapping_space,
    // primal time
    const dealii::FiniteElement<1> &primal_fe_time,
    const dealii::Mapping<1> &primal_mapping_time)
    :  // init space
      dof_dual(dof_dual),
      dof_primal(dof_primal),
      space_fe_values(mapping_space, fe_space, quad_space,
                      dealii::update_values | dealii::update_gradients |
                          dealii::update_JxW_values |
                          dealii::update_quadrature_points),
      space_phi(fe_space.dofs_per_cell),
      space_symgrad_phi(fe_space.dofs_per_cell),
      space_grad_phi(fe_space.dofs_per_cell),
      space_div_phi(fe_space.dofs_per_cell),
      space_psi(fe_space.dofs_per_cell),
      space_local_dof_indices(fe_space.dofs_per_cell),
      primal_space_fe_values(primal_mapping_space, primal_fe_space, quad_space,
                             dealii::update_values | dealii::update_gradients |
                                 dealii::update_quadrature_points),
      primal_space_phi(primal_fe_space.dofs_per_cell),
      primal_space_grad_phi(primal_fe_space.dofs_per_cell),
      primal_space_local_dof_indices(primal_fe_space.dofs_per_cell),
      // init time
      time_fe_values(mapping_time, fe_time, quad_time,
                     dealii::update_values | dealii::update_gradients |
                         dealii::update_JxW_values |
                         dealii::update_quadrature_points),
      primal_time_fe_values(primal_mapping_time, primal_fe_time, quad_time,
                            dealii::update_values | dealii::update_gradients |
                                dealii::update_quadrature_points),
      time_fe_face_values(
          mapping_time, fe_time, face_nodes,
          dealii::update_values | dealii::update_quadrature_points),
      time_fe_face_values_neighbor(mapping_time, fe_time, face_nodes,
                                   dealii::update_values),
      time_zeta(fe_time.dofs_per_cell),
      time_grad_zeta(fe_time.dofs_per_cell),
      time_local_dof_indices(fe_time.dofs_per_cell),
      time_local_dof_indices_neighbor(fe_time.dofs_per_cell),
      v(),
      grad_v() {}

template <int dim>
FluidAssembly<dim>::FluidAssembly(const FluidAssembly &scratch)
    : dof_dual(scratch.dof_dual),
      dof_primal(scratch.dof_primal),
      space_fe_values(scratch.space_fe_values.get_mapping(),
                      scratch.space_fe_values.get_fe(),
                      scratch.space_fe_values.get_quadrature(),
                      scratch.space_fe_values.get_update_flags()),
      space_phi(scratch.space_phi),
      space_symgrad_phi(scratch.space_symgrad_phi),
      space_grad_phi(scratch.space_grad_phi),
      space_div_phi(scratch.space_div_phi),
      space_psi(scratch.space_psi),
      space_dofs_per_cell(scratch.space_dofs_per_cell),
      space_JxW(scratch.space_JxW),
      space_local_dof_indices(scratch.space_local_dof_indices),
      primal_space_fe_values(scratch.primal_space_fe_values.get_mapping(),
                             scratch.primal_space_fe_values.get_fe(),
                             scratch.primal_space_fe_values.get_quadrature(),
                             scratch.primal_space_fe_values.get_update_flags()),
      primal_space_phi(scratch.primal_space_phi),
      primal_space_grad_phi(scratch.primal_space_grad_phi),
      primal_space_local_dof_indices(scratch.primal_space_local_dof_indices),
      //
      time_fe_values(scratch.time_fe_values.get_mapping(),
                     scratch.time_fe_values.get_fe(),
                     scratch.time_fe_values.get_quadrature(),
                     scratch.time_fe_values.get_update_flags()),
      primal_time_fe_values(scratch.primal_time_fe_values.get_mapping(),
                            scratch.primal_time_fe_values.get_fe(),
                            scratch.primal_time_fe_values.get_quadrature(),
                            scratch.primal_time_fe_values.get_update_flags()),
      time_fe_face_values(scratch.time_fe_face_values.get_mapping(),
                          scratch.time_fe_face_values.get_fe(),
                          scratch.time_fe_face_values.get_quadrature(),
                          scratch.time_fe_face_values.get_update_flags()),
      time_fe_face_values_neighbor(
          scratch.time_fe_face_values_neighbor.get_mapping(),
          scratch.time_fe_face_values_neighbor.get_fe(),
          scratch.time_fe_face_values_neighbor.get_quadrature(),
          scratch.time_fe_face_values_neighbor.get_update_flags()),
      time_zeta(scratch.time_zeta),
      time_grad_zeta(scratch.time_grad_zeta),
      time_dofs_per_cell(scratch.time_dofs_per_cell),
      time_JxW(scratch.time_JxW),
      time_local_dof_indices(scratch.time_local_dof_indices),
      time_local_dof_indices_neighbor(scratch.time_local_dof_indices_neighbor),
      //
      v(scratch.v),
      grad_v(scratch.grad_v),
      viscosity(scratch.viscosity) {}

}  // namespace Scratch

namespace CopyData {

template <int dim>
FluidAssembly<dim>::FluidAssembly(
    const dealii::FiniteElement<dim> &fe_s,
    const dealii::FiniteElement<1> &fe_t,
    const dealii::types::global_dof_index &n_global_active_cells_t)
    : local_matrix(
          n_global_active_cells_t * fe_s.dofs_per_cell * fe_t.dofs_per_cell,
          n_global_active_cells_t * fe_s.dofs_per_cell * fe_t.dofs_per_cell),
      vi_ui_matrix(
          n_global_active_cells_t,  // n_cells time
          dealii::FullMatrix<double>(fe_s.dofs_per_cell * fe_t.dofs_per_cell,
                                     fe_s.dofs_per_cell * fe_t.dofs_per_cell)),
      ve_ui_matrix(
          n_global_active_cells_t - 1,  // n_cells time -1
          dealii::FullMatrix<double>(fe_s.dofs_per_cell * fe_t.dofs_per_cell,
                                     fe_s.dofs_per_cell * fe_t.dofs_per_cell)),
      new_local_dof_indices(n_global_active_cells_t * fe_s.dofs_per_cell *
                            fe_t.dofs_per_cell),
      local_dof_indices(n_global_active_cells_t,  // n_cells time
                        std::vector<dealii::types::global_dof_index>(
                            fe_s.dofs_per_cell * fe_t.dofs_per_cell)),
      local_dof_indices_neighbor(
          n_global_active_cells_t - 1,  // n_cells time - 1
          std::vector<dealii::types::global_dof_index>(fe_s.dofs_per_cell *
                                                       fe_t.dofs_per_cell)) {}

template <int dim>
FluidAssembly<dim>::FluidAssembly(const FluidAssembly &copydata)
    : local_matrix(copydata.local_matrix),
      vi_ui_matrix(copydata.vi_ui_matrix),
      ve_ui_matrix(copydata.ve_ui_matrix),
      new_local_dof_indices(copydata.new_local_dof_indices),
      local_dof_indices(copydata.local_dof_indices),
      local_dof_indices_neighbor(copydata.local_dof_indices_neighbor) {}

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
void Assembler<dim>::set_time_quad_type(std::string _quad_type) {
  time.quad_type = _quad_type;
}

template <int dim>
void Assembler<dim>::assemble(
    std::shared_ptr<dealii::TrilinosWrappers::SparseMatrix> _L,
    const typename fluid::types::spacetime::dwr::slabs<dim>::iterator &slab,
    std::shared_ptr<dealii::TrilinosWrappers::MPI::Vector> _u, bool _nonlin) {
  ////////////////////////////////////////////////////////////////////////////
  // check
  Assert(dim == 2 || dim == 3, dealii::ExcNotImplemented());

  Assert(_L.use_count(), dealii::ExcNotInitialized());

  Assert(slab->space.dual.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->fe.use_count(), dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.dual.fe_info->constraints.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->space.primal.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->fe.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->space.primal.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->time.dual.fe_info->dof.use_count(), dealii::ExcNotInitialized());
  Assert(slab->time.dual.fe_info->fe.use_count(), dealii::ExcNotInitialized());
  Assert(slab->time.dual.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->time.primal.fe_info->dof.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->time.primal.fe_info->fe.use_count(),
         dealii::ExcNotInitialized());
  Assert(slab->time.primal.fe_info->mapping.use_count(),
         dealii::ExcNotInitialized());

  Assert(slab->spacetime.dual.constraints.use_count(),
         dealii::ExcNotInitialized());

  Assert(function.viscosity.use_count(), dealii::ExcNotInitialized());

  ////////////////////////////////////////////////////////////////////////////
  // init

  L = _L;
  u = _u;
  nonlin = _nonlin;

  space.dof = slab->space.dual.fe_info->dof;
  space.fe = slab->space.dual.fe_info->fe;
  space.mapping = slab->space.dual.fe_info->mapping;
  space.constraints = slab->space.dual.fe_info->constraints;

  primal.space.dof = slab->space.primal.fe_info->dof;
  primal.space.fe = slab->space.primal.fe_info->fe;
  primal.space.mapping = slab->space.primal.fe_info->mapping;

  time.dof = slab->time.dual.fe_info->dof;
  time.fe = slab->time.dual.fe_info->fe;
  time.mapping = slab->time.dual.fe_info->mapping;

  primal.time.dof = slab->time.primal.fe_info->dof;
  primal.time.fe = slab->time.primal.fe_info->fe;
  primal.time.mapping = slab->time.primal.fe_info->mapping;

  spacetime.constraints = slab->spacetime.dual.constraints;

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

  const dealii::QGauss<1> quad_time(time.fe->tensor_degree() + 2);

  const dealii::QGaussLobatto<1> face_nodes(2);

  time.n_global_active_cells = slab->time.tria->n_global_active_cells();

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
      Assembly::Scratch::FluidAssembly<dim>(
          *slab->space.dual.fe_info->dof, *slab->space.primal.fe_info->dof,
          *space.fe, *space.mapping, quad_space, *time.fe, *time.mapping,
          quad_time, face_nodes, *primal.space.fe, *primal.space.mapping,
          *primal.time.fe, *primal.time.mapping),
      Assembly::CopyData::FluidAssembly<dim>(*space.fe, *time.fe,
                                             time.n_global_active_cells));

  L->compress(dealii::VectorOperation::add);
}

/// Local assemble on cell.
template <int dim>
void Assembler<dim>::local_assemble_cell(
    const typename dealii::DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::FluidAssembly<dim> &scratch,
    Assembly::CopyData::FluidAssembly<dim> &copydata) {
  typename dealii::DoFHandler<dim>::active_cell_iterator cell_dual(
      &cell->get_triangulation(), cell->level(), cell->index(),
      &scratch.dof_dual);

  typename dealii::DoFHandler<dim>::active_cell_iterator cell_primal(
      &cell->get_triangulation(), cell->level(), cell->index(),
      &scratch.dof_primal);

  // reinit scratch and data to current cell
  scratch.space_fe_values.reinit(cell_dual);
  scratch.primal_space_fe_values.reinit(cell_primal);

  // fetch local dof data
  cell_dual->get_dof_indices(scratch.space_local_dof_indices);
  cell_primal->get_dof_indices(scratch.primal_space_local_dof_indices);

  auto cell_time = time.dof->begin_active();
  auto primal_cell_time = primal.time.dof->begin_active();
  auto endc_time = time.dof->end();

  unsigned int n;
  copydata.local_matrix = 0;

  unsigned int element_offset =
      time.fe->dofs_per_cell * space.fe->dofs_per_cell;
  for (; cell_time != endc_time;) {
    n = cell_time->index();
    //		copydata.vi_ui_matrix[n] = 0;

    // dof mapping
    cell_time->get_dof_indices(scratch.time_local_dof_indices);
    for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
      for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii) {
        copydata.new_local_dof_indices[i + ii * space.fe->dofs_per_cell +
                                       n * element_offset] =
            scratch.space_local_dof_indices[i] +
            scratch.time_local_dof_indices[ii] * space.dof->n_dofs();
      }

    // prefetch data

    // assemble: volume
    scratch.time_fe_values.reinit(cell_time);
    scratch.primal_time_fe_values.reinit(primal_cell_time);
    for (unsigned int qt{0}; qt < scratch.time_fe_values.n_quadrature_points;
         ++qt) {
      function.viscosity->set_time(
          scratch.time_fe_values.quadrature_point(qt)[0]);

      for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
           ++q) {
        scratch.viscosity = function.viscosity->value(
            scratch.space_fe_values.quadrature_point(q), 0);

        // prefetch dual
        for (unsigned int k{0}; k < space.fe->dofs_per_cell; ++k) {
          scratch.space_phi[k] =
              scratch.space_fe_values[convection].value(k, q);
          scratch.space_symgrad_phi[k] =
              scratch.space_fe_values[convection].symmetric_gradient(k, q);
          scratch.space_grad_phi[k] =
              scratch.space_fe_values[convection].gradient(k, q);
          scratch.space_div_phi[k] =
              scratch.space_fe_values[convection].divergence(k, q);
          scratch.space_psi[k] = scratch.space_fe_values[pressure].value(k, q);
        }

        // prefetch primal
        for (unsigned int k{0}; k < primal.space.fe->dofs_per_cell; ++k) {
          scratch.primal_space_phi[k] =
              scratch.primal_space_fe_values[convection].value(k, q);
          scratch.primal_space_grad_phi[k] =
              scratch.primal_space_fe_values[convection].gradient(k, q);
        }

        if (nonlin) {
          scratch.v = 0;
          scratch.grad_v = 0;

          for (unsigned int ii{0}; ii < primal.time.fe->dofs_per_cell; ++ii)
            for (unsigned int i{0}; i < primal.space.fe->dofs_per_cell; ++i) {
              // correct ST solution vector entry
              double u_i_ii = (*u)[scratch.primal_space_local_dof_indices[i]
                                   // time offset
                                   + primal.space.dof->n_dofs() *
                                         (n * primal.time.fe->dofs_per_cell)
                                   // local in time dof
                                   + primal.space.dof->n_dofs() * ii];

              // all other evals use shape values in time, so multiply only once
              u_i_ii *= scratch.primal_time_fe_values.shape_value(ii, qt);

              // v
              scratch.v += u_i_ii * scratch.primal_space_phi[i];

              // grad v
              scratch.grad_v += u_i_ii * scratch.primal_space_grad_phi[i];
            }

          // for Navier-Stokes assemble convection term derivatives
          for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
            for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
              for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
                for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
                  copydata.local_matrix(
                      i + ii * space.fe->dofs_per_cell + n * element_offset,
                      j + jj * space.fe->dofs_per_cell + n * element_offset) +=
                      //						copydata.vi_ui_matrix[n](
                      //							i
                      //+ ii*space.fe->dofs_per_cell, 							j +
                      //jj*space.fe->dofs_per_cell 						) +=
                      // convection C(u)_bb
                      ((scratch.space_grad_phi[i] * scratch.v +
                        scratch.grad_v * scratch.space_phi[i]) *
                       scratch.time_fe_values.shape_value(ii, qt) *

                       scratch.space_phi[j] *
                       scratch.time_fe_values.shape_value(jj, qt) *

                       scratch.space_fe_values.JxW(q) *
                       scratch.time_fe_values.JxW(qt));
                }
        }

        for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
          for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
            for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
              for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
                copydata.local_matrix(
                    i + ii * space.fe->dofs_per_cell + n * element_offset,
                    j + jj * space.fe->dofs_per_cell + n * element_offset) +=
                    //					copydata.vi_ui_matrix[n](
                    //						i +
                    //ii*space.fe->dofs_per_cell, 						j + jj*space.fe->dofs_per_cell
                    //					) +=
                    // convection M_bb: w^+(x,t) * (-\partial_t z_b^+(x,t))
                    (scratch.space_fe_values[convection].value(i, q) *
                     scratch.time_fe_values.shape_value(ii, qt) *

                     scratch.space_fe_values[convection].value(j, q) *
                     (-scratch.time_fe_values.shape_grad(jj, qt)[0]) *

                     scratch.space_fe_values.JxW(q) *
                     scratch.time_fe_values.JxW(qt))
                    // convection A_bb
                    +
                    ((symmetric_stress
                          ? (scratch.space_symgrad_phi[i] *
                             scratch.time_fe_values.shape_value(ii, qt) *

                             scratch.viscosity * 2. *
                             scratch.space_symgrad_phi[j] *
                             scratch.time_fe_values.shape_value(jj, qt))
                          : (scalar_product(
                                scratch.space_grad_phi[i] *
                                    scratch.time_fe_values.shape_value(ii, qt),

                                scratch.viscosity * scratch.space_grad_phi[j] *
                                    scratch.time_fe_values.shape_value(jj,
                                                                       qt)))) *

                     scratch.space_fe_values.JxW(q) *
                     scratch.time_fe_values.JxW(qt))
                    // pressure B_bp
                    + (scratch.space_div_phi[i] *
                       scratch.time_fe_values.shape_value(ii, qt) *

                       scratch.space_psi[j] *
                       scratch.time_fe_values.shape_value(jj, qt) *

                       scratch.space_fe_values.JxW(q) *
                       scratch.time_fe_values.JxW(qt))
                    // div-free constraint B_pb
                    - (scratch.space_psi[i] *
                       scratch.time_fe_values.shape_value(ii, qt) *

                       scratch.space_div_phi[j] *
                       scratch.time_fe_values.shape_value(jj, qt) *

                       scratch.space_fe_values.JxW(q) *
                       scratch.time_fe_values.JxW(qt));
              }
      }  // x_q
    }    // t_q

    // prepare [.]_t_n trace operator
    scratch.time_fe_face_values.reinit(cell_time);
    // assemble: face (w^+ * z^+)
    for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
         ++q) {
      for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
        for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
          for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
            for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
              copydata.local_matrix(
                  i + ii * space.fe->dofs_per_cell + n * element_offset,
                  j + jj * space.fe->dofs_per_cell + n * element_offset) +=
                  // 				copydata.vi_ui_matrix[n](
                  // 					i +
                  // ii*space.fe->dofs_per_cell, 					j + jj*space.fe->dofs_per_cell
                  // 				) +=
                  // trace operator:  w(x,t_n)^+ * z(x,t_n)^+
                  scratch.space_fe_values[convection].value(i, q) *
                  scratch.time_fe_face_values.shape_value(ii, 1) *

                  scratch.space_fe_values[convection].value(j, q) *
                  scratch.time_fe_face_values.shape_value(jj, 1) *

                  scratch.space_fe_values.JxW(q);
            }
    }

    // assemble: face (w^+ * z^-)
    //
    // NOTE: sparsity pattern for n_global_active_cells = 3:
    //       [ ++ +-    ]
    //       [    ++ +- ]
    //       [       ++ ]
    ++cell_time;
    ++primal_cell_time;
    if ((time.n_global_active_cells > 1) &&
        (n < (time.n_global_active_cells - 1))) {
      // init time next element
      auto next_cell_time{cell_time};

      Assert((next_cell_time != endc_time),
             dealii::ExcMessage(
                 "next time cell is invalid. This is an internal error."));

      scratch.time_fe_face_values_neighbor.reinit(next_cell_time);

      // assemble: face (w^+ * z^-)
      for (unsigned int q{0}; q < scratch.space_fe_values.n_quadrature_points;
           ++q) {
        for (unsigned int ii{0}; ii < time.fe->dofs_per_cell; ++ii)
          for (unsigned int jj{0}; jj < time.fe->dofs_per_cell; ++jj)
            for (unsigned int i{0}; i < space.fe->dofs_per_cell; ++i)
              for (unsigned int j{0}; j < space.fe->dofs_per_cell; ++j) {
                copydata.local_matrix(
                    i + ii * space.fe->dofs_per_cell + n * element_offset,
                    j + jj * space.fe->dofs_per_cell +
                        (n + 1) * element_offset) -=
                    // trace operator: - w(x,t_n)^+ * z(x,t_m)^-
                    scratch.space_fe_values[convection].value(i, q) *
                    scratch.time_fe_face_values.shape_value(ii, 1) *

                    scratch.space_fe_values[convection].value(j, q) *
                    scratch.time_fe_face_values_neighbor.shape_value(jj, 0) *

                    scratch.space_fe_values.JxW(q);
              }
      }
    }
  }
}

/// Copy local assembly to global matrix.
template <int dim>
void Assembler<dim>::copy_local_to_global_cell(
    const Assembly::CopyData::FluidAssembly<dim> &copydata) {
  spacetime.constraints->distribute_local_to_global(
      copydata.local_matrix, copydata.new_local_dof_indices, *L);
}

}  // namespace Operator
}  // namespace dual
}  // namespace spacetime
}  // namespace fluid

#include "ST_Dual_FluidAssembly.inst.in"
