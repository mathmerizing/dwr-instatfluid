/**
 * @file Grid_Schaefer_Turek_2D.tpl.cc
 * @author Julian Roth (JR)
 * @author Jan Philipp Thiele (JPT)
 * @author Uwe Koecher (UK)
 * @author Marius Paul Bruchhaeuser (MPB)
 * @author Jan Philipp Thiele (JPT)
 *
 * @Date 2022-01-14, Fluid, JPT
 * @date 2021-11-05, Schaefer/Turek, UK
 * @date 2021-09-27, Schaefer/Turek, JR
 * @date 2020-11-25, Schaefer/Turek, JPT
 *
 * @date 2019-11-11, UK
 * @date 2018-07-26, UK
 * @date 2018-03-06, UK
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

// PROJECT includes
#include <deal.II/grid/manifold_lib.h>

#include <fluid/grid/Grid_Schaefer_Turek_2D.tpl.hh>
#include <fluid/types/boundary_id.hh>

namespace fluid {
namespace grid {

template <int dim>
void Grid_Schaefer_Turek_2D<dim>::set_manifolds() {
  Assert((dim == 2), dealii::ExcNotImplemented());

  // Obstacle: SphericalManifold
  dealii::Point<dim> p_obstacle_mid;
  for (unsigned int d{0}; d < dim; ++d) {
    p_obstacle_mid[d] = .2;
  }
  {
    auto slab(this->slabs.begin());
    auto ends(this->slabs.end());

    for (; slab != ends; ++slab) {
      slab->space.tria->reset_all_manifolds();
      slab->space.tria->set_all_manifold_ids(0);

      auto cell(slab->space.tria->begin_active());
      auto endc(slab->space.tria->end());

      for (; cell != endc; ++cell)
        if (cell->at_boundary())
          for (unsigned int face(0);
               face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
              auto center{cell->face(face)->center()};

              // obstacle cylinder
              if ((center - p_obstacle_mid).norm() < 0.11) {
                cell->face(face)->set_manifold_id(42  // SphericalManifold
                );
              }
            }
          }

      slab->space.tria->set_manifold(
          42, dealii::SphericalManifold<dim>(p_obstacle_mid));
    }
  }
}

template <int dim>
void Grid_Schaefer_Turek_2D<dim>::set_boundary_indicators() {
  // set boundary indicators (space)
  {
    auto slab(this->slabs.begin());
    auto ends(this->slabs.end());

    for (; slab != ends; ++slab) {
      auto cell(slab->space.tria->begin_active());
      auto endc(slab->space.tria->end());

      for (; cell != endc; ++cell)
        if (cell->at_boundary())
          for (unsigned int face(0);
               face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
              auto center{cell->face(face)->center()};

              // walls: no-flow / no-slip conditions for y=0 and y=0.41
              // prescribed homog. Dirichlet for convection
              if ((std::abs(center[1] - 0.) < 1.e-14) ||
                  (std::abs(center[1] - 0.41) < 1.e-14)) {
                cell->face(face)->set_boundary_id(
                    static_cast<dealii::types::boundary_id>(
                        fluid::types::space::boundary_id::prescribed_no_slip));
              }
              // parabolic inflow boundary for x=0
              else if (std::abs(center[0] - 0.) < 1.e-14) {
                cell->face(face)->set_boundary_id(static_cast<
                                                  dealii::types::boundary_id>(
                    fluid::types::space::boundary_id::prescribed_convection_c1 +
                    fluid::types::space::boundary_id::prescribed_convection_c2 +
                    fluid::types::space::boundary_id::
                        prescribed_convection_c3));
              }
              // outflow boundary (do nothing) for x=2.2
              else if (std::abs(center[0] - 2.2) < 1.e-14) {
                cell->face(face)->set_boundary_id(static_cast<
                                                  dealii::types::boundary_id>(
                    fluid::types::space::boundary_id::prescribed_do_nothing));
              }
              // walls of the obstacle cylinder
              else {
                // walls OF CYLINDER: no-flow / no-slip conditions:
                //   prescribed homog. Dirichlet for convection
                // TODO: maybe use additional color obstacle for something
                cell->face(face)->set_boundary_id(
                    static_cast<dealii::types::boundary_id>(
                        fluid::types::space::boundary_id::prescribed_no_slip +
                        fluid::types::space::boundary_id::prescribed_obstacle));
              }
            }  // face at boundary
          }    // loop over cells at boundary and faces of those cells
    }          // for slab
  }

  // set Sigma_0
  {
    if (this->slabs.size()) {
      auto slab_Q1(this->slabs.begin());

      auto cell(slab_Q1->time.tria->begin_active());
      auto endc(slab_Q1->time.tria->end());

      for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
          for (unsigned int face(0);
               face < dealii::GeometryInfo<1>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary() && face == 0) {
              cell->face(face)->set_boundary_id(
                  static_cast<dealii::types::boundary_id>(
                      fluid::types::time::boundary_id::Sigma0));
            }
          }
        }
      }
    }
  }

  // set Sigma_T
  {
    if (this->slabs.size()) {
      auto slab_QN(this->slabs.rbegin());

      auto cell(slab_QN->time.tria->begin_active());
      auto endc(slab_QN->time.tria->end());

      for (; cell != endc; ++cell) {
        if (cell->at_boundary()) {
          for (unsigned int face(0);
               face < dealii::GeometryInfo<1>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary() && face == 1) {
              cell->face(face)->set_boundary_id(
                  static_cast<dealii::types::boundary_id>(
                      fluid::types::time::boundary_id::SigmaT));
            }
          }
        }
      }
    }
  }
}

}  // namespace grid
}  // namespace fluid

#include "Grid_Schaefer_Turek_2D.inst.in"
